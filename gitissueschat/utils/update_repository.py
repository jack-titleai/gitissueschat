#!/usr/bin/env python3
"""
Update Repository Script

This script updates an existing repository database with new or modified issues and comments.
It checks the most recent API call from the SQLite database and fetches any issues/comments
that have been added or updated since then. It updates both the SQLite database and the 
ChromaDB instances accordingly.

Usage:
    python update_repository.py owner/repo [--token TOKEN] [--chunk-size SIZE] [--chunk-overlap OVERLAP]
"""

import os
import argparse
import logging
import time
import sqlite3
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dotenv import load_dotenv

from gitissueschat.github_issues import GitHubIssuesFetcher
from gitissueschat.sqlite_storage.sqlite_storage import SQLiteIssueStorage
from gitissueschat.embed.embed_database_to_chromadb import embed_database_to_chromadb, process_issue_with_comments
from gitissueschat.embed.chroma_database import ChunksDatabase
from gitissueschat.embed.llamaindex_chunker import LlamaIndexChunker
from gitissueschat.utils.db_path_manager import get_sqlite_db_path, get_chroma_db_path
from gitissueschat.utils.process_repository import normalize_repo_input

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_most_recent_api_call(storage: SQLiteIssueStorage, repo_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent API call for a repository.
    
    Args:
        storage: SQLite issue storage instance.
        repo_name: Repository name in the format 'owner/repo'.
        
    Returns:
        The most recent API call log or None if no logs exist.
    """
    logs = storage.get_api_logs(repo_name, limit=1)
    if logs:
        return logs[0]
    return None


def delete_issue_chunks_from_chromadb(
    chroma_db: ChunksDatabase, 
    issue_id: int
) -> None:
    """
    Delete all chunks related to an issue from ChromaDB.
    
    Args:
        chroma_db: ChromaDB instance.
        issue_id: Issue ID to delete.
    """
    # Get all chunks for this issue
    results = chroma_db.collection.get(where={"issue_id": issue_id})
    
    if results and results["ids"]:
        # Delete chunks by IDs
        chroma_db.collection.delete(ids=results["ids"])
        logger.info(f"Deleted {len(results['ids'])} chunks for issue ID {issue_id} from ChromaDB")


def update_repository(
    repo_name: str,
    github_token: str,
    sqlite_db_path: str,
    chroma_db_path: str,
    collection_name: str = "github_issues",
    chunk_size: int = 250,
    chunk_overlap: int = 50,
    project_id: Optional[str] = None,
    api_key: Optional[str] = None,
    credentials_path: Optional[str] = None,
    disable_buffer: bool = False,
) -> Dict[str, Any]:
    """
    Update a repository with new or modified issues and comments.
    
    Args:
        repo_name: Repository name in the format 'owner/repo'.
        github_token: GitHub API token.
        sqlite_db_path: Path to the SQLite database.
        chroma_db_path: Path to the ChromaDB database.
        collection_name: Name of the ChromaDB collection.
        chunk_size: Size of chunks in tokens.
        chunk_overlap: Overlap between chunks in tokens.
        project_id: Google Cloud project ID.
        api_key: Google API key.
        credentials_path: Path to the Google Cloud service account key file.
        disable_buffer: Disable the buffer period for the last update time.
        
    Returns:
        Statistics about the update.
    """
    logger.info(f"Updating repository: {repo_name}")
    
    # Initialize the SQLite issue storage
    storage = SQLiteIssueStorage(sqlite_db_path)
    
    # Get the most recent API call
    most_recent_call = get_most_recent_api_call(storage, repo_name)
    
    if not most_recent_call:
        logger.warning(f"No previous API calls found for repository {repo_name}. Consider running process_repository.py first.")
        return {"error": "No previous API calls found"}
    
    # Get the timestamp of the most recent API call
    last_update_time = most_recent_call.get("timestamp", "")
    logger.info(f"Last update time from database: {last_update_time}")
    
    # Ensure the timestamp is in the correct format for the GitHub API
    # The GitHub API expects ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)
    try:
        # Parse the timestamp from the database (which is in UTC)
        last_update_datetime = datetime.strptime(last_update_time, "%Y-%m-%d %H:%M:%S")
        
        # Default buffer hours
        buffer_hours = 1
        
        if not disable_buffer:
            # Subtract a buffer period to ensure we don't miss any issues
            # due to slight timing differences between GitHub and our database
            last_update_datetime = last_update_datetime - timedelta(hours=buffer_hours)
        
        # Format in ISO 8601 format with UTC timezone indicator (Z)
        last_update_time = last_update_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info(f"Adjusted last update time (with {buffer_hours if not disable_buffer else 0}h buffer): {last_update_time}")
    except Exception as e:
        logger.warning(f"Error parsing timestamp: {e}. Using original timestamp.")
    
    # Initialize the GitHub issues fetcher
    fetcher = GitHubIssuesFetcher(github_token)
    
    # Get existing issue numbers
    existing_issue_numbers = set(storage.get_issue_numbers(repo_name))
    existing_issue_codes = storage.get_issue_codes(repo_name)
    
    # Track statistics
    stats = {
        "total_issues": 0,
        "new_issues": 0,
        "updated_issues": 0,
        "redundant_issues": 0,
        "issues_before": storage.get_issue_count(repo_name),
        "execution_time": 0,
    }
    
    # Start timing the execution
    start_time = time.time()
    
    # Initialize ChromaDB
    chroma_db = ChunksDatabase(
        db_path=chroma_db_path,
        collection_name=collection_name,
        project_id=project_id,
        api_key=api_key,
        credentials_path=credentials_path
    )
    
    # Initialize chunker
    chunker = LlamaIndexChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Set of issues that need to be re-embedded
    issues_to_update = set()
    
    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Configure the fetcher to use a callback for batch processing
    def process_batch(batch_issues, batch_stats):
        nonlocal stats, issues_to_update
        
        # Debug print to see what issues are being processed
        print(f"DEBUG - Processing batch with {len(batch_issues)} issues: {[issue['number'] for issue in batch_issues]}")
        print(f"DEBUG - Batch stats: {batch_stats}")
        
        # Update counts for total and new issues
        stats["total_issues"] += len(batch_issues)
        stats["new_issues"] += batch_stats.get("new_count", 0)
        
        # Process each issue in the batch
        issues_to_store = []
        
        for issue in batch_issues:
            issue_id = issue["id"]
            issue_number = issue["number"]
            issue_updated_at = issue.get("updated_at", "")
            
            # Check if this issue is already in the database
            cursor.execute(
                "SELECT updated_at FROM issues WHERE id = ?",
                (issue_id,)
            )
            row = cursor.fetchone()
            
            if row:
                # Issue exists in database
                db_updated_at = row[0] if row[0] else ""
                
                # If the issue's updated_at timestamp has changed, it's been updated
                if db_updated_at != issue_updated_at:
                    stats["updated_issues"] += 1
                    issues_to_update.add(issue_id)
                    issues_to_store.append(issue)
                    logger.debug(f"Issue #{issue_number} (ID: {issue_id}) has been updated")
                else:
                    # Issue hasn't changed
                    stats["redundant_issues"] += 1
                    logger.debug(f"Issue #{issue_number} (ID: {issue_id}) is redundant - no changes detected")
            else:
                # New issue
                stats["new_issues"] += 1
                issues_to_update.add(issue_id)
                issues_to_store.append(issue)
                logger.debug(f"Issue #{issue_number} (ID: {issue_id}) is new")
        
        # Store only the issues that need to be updated
        if issues_to_store:
            storage.store_issues(issues_to_store, repo_name)
    
    # Fetch issues updated since the last API call
    result = fetcher.fetch_issues(
        repo_name=repo_name,
        state="all",
        include_comments=True,
        existing_issue_numbers=existing_issue_numbers,
        existing_issue_codes=existing_issue_codes,
        batch_size=100,
        batch_callback=process_batch,
        since=last_update_time  # Only fetch issues updated since the last API call
    )
    
    # Debug print to see what the fetcher is returning
    print(f"DEBUG - Fetcher returned: {result}")
    
    logger.info(f"Fetched {stats['total_issues']} issues: {stats['new_issues']} new, {stats['updated_issues']} updated, {stats['redundant_issues']} redundant")
    
    # Process updated issues
    if issues_to_update:
        logger.info(f"Processing {len(issues_to_update)} issues for embedding")
        
        # Process each issue
        for issue_id in issues_to_update:
            # Get the issue from the database
            cursor.execute(
                "SELECT * FROM issues WHERE id = ?",
                (issue_id,)
            )
            issue = dict(cursor.fetchone())
            
            # Get the issue's updated_at timestamp
            issue_updated_at = issue.get("updated_at", "")
            
            # Check if this issue is already in ChromaDB
            results = chroma_db.collection.get(where={"issue_id": issue_id})
            
            # If the issue is already in ChromaDB and hasn't been updated, skip it
            if results and results["ids"] and results["metadatas"] and len(results["metadatas"]) > 0:
                # Get the updated_at timestamp from the first chunk
                chunk_updated_at = results["metadatas"][0].get("updated_at", "")
                
                # If the timestamps match, skip this issue
                if issue_updated_at == chunk_updated_at:
                    logger.info(f"Skipping issue #{issue['number']} (ID: {issue_id}) - no changes detected")
                    # Update the stats to reflect that this is a redundant issue, not an updated one
                    stats["updated_issues"] -= 1
                    stats["redundant_issues"] += 1
                    continue
            
            # Delete existing chunks for this issue
            delete_issue_chunks_from_chromadb(chroma_db, issue_id)
            
            # Process the issue and its comments
            chunks = process_issue_with_comments(sqlite_db_path, repo_name, issue, chunker)
            
            # Add chunks to ChromaDB
            if chunks:
                chroma_db.add_chunks(chunks)
                logger.info(f"Added {len(chunks)} chunks for issue #{issue['number']} to ChromaDB")
        
        # Close the connection
        conn.close()
    
    # Get API rate limit info
    rate_limit_info = fetcher.get_rate_limit_info()
    remaining = rate_limit_info.get("remaining", 0)
    limit = rate_limit_info.get("limit", 0)
    reset_time = rate_limit_info.get("reset_time", "")
    
    logger.info(f"API Rate Limit: {remaining}/{limit}")
    logger.info(f"Rate limit resets at: {reset_time}")
    
    # Calculate execution time
    stats["execution_time"] = time.time() - start_time
    
    # Log the API call
    storage.log_api_call(
        repo_name=repo_name,
        new_issues_count=stats["new_issues"],
        updated_issues_count=stats["updated_issues"],
        redundant_issues_count=stats["redundant_issues"],
        issues_before_count=stats["issues_before"],
        issues_after_count=storage.get_issue_count(repo_name),
        api_rate_limit_remaining=remaining,
        api_rate_limit_total=limit,
        execution_time_seconds=stats["execution_time"],
    )
    
    logger.info(f"Total execution time for updating: {stats['execution_time']:.2f} seconds")
    
    return stats


def main():
    """
    Main function to update a GitHub repository.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Update a GitHub repository with new or modified issues and comments")
    parser.add_argument("repo", help="GitHub repository in the format 'owner/repo' or as a URL")
    parser.add_argument("--token", help="GitHub API token (if not provided, will use GITHUB_TOKEN environment variable)")
    parser.add_argument("--collection-name", default="github_issues", help="Name of the ChromaDB collection")
    parser.add_argument("--chunk-size", type=int, default=500, help="Size of chunks in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Overlap between chunks in tokens")
    parser.add_argument("--disable-buffer", action="store_true", help="Disable the buffer period for the last update time")
    args = parser.parse_args()
    
    # Normalize repository input
    repo_name = normalize_repo_input(args.repo)
    logger.info(f"Using repository: {repo_name}")
    
    # Get GitHub token
    github_token = args.token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        logger.error("GitHub token not provided. Please provide a token with --token or set the GITHUB_TOKEN environment variable.")
        exit(1)
    
    # Get Google Cloud credentials
    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    if not project_id:
        logger.error("GOOGLE_PROJECT_ID environment variable not set")
        exit(1)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Get database paths
    sqlite_db_path = get_sqlite_db_path(repo_name)
    chroma_db_path = get_chroma_db_path(repo_name)
    
    logger.info(f"SQLite database path: {sqlite_db_path}")
    logger.info(f"ChromaDB database path: {chroma_db_path}")
    
    # Update the repository
    stats = update_repository(
        repo_name=repo_name,
        github_token=github_token,
        sqlite_db_path=sqlite_db_path,
        chroma_db_path=chroma_db_path,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        project_id=project_id,
        api_key=api_key,
        credentials_path=credentials_path,
        disable_buffer=args.disable_buffer,
    )
    
    logger.info("Repository Update Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
