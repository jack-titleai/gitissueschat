#!/usr/bin/env python3
"""
Process Repository Script

This script takes a GitHub repository (either as a URL or in the format 'owner/repo'), 
downloads all issues to an SQLite database, chunks them with specified parameters, 
and embeds the chunks into ChromaDB.

Usage:
    python process_repository.py owner/repo [--token TOKEN] [--chunk-size SIZE] [--chunk-overlap OVERLAP]
    python process_repository.py https://github.com/owner/repo [--token TOKEN] [--chunk-size SIZE] [--chunk-overlap OVERLAP]
"""

import os
import argparse
import logging
import time
import re
from typing import Optional
from urllib.parse import urlparse
from dotenv import load_dotenv

from gitissueschat.github_issues import GitHubIssuesFetcher
from gitissueschat.sqlite_storage.sqlite_storage import SQLiteIssueStorage
from gitissueschat.embed.embed_database_to_chromadb import embed_database_to_chromadb
from gitissueschat.utils.db_path_manager import get_sqlite_db_path, get_chroma_db_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_repo_from_url(url: str) -> str:
    """
    Extract the repository name from a GitHub URL.
    
    Args:
        url: GitHub repository URL.
        
    Returns:
        Repository name in the format 'owner/repo'.
        
    Raises:
        ValueError: If the URL is not a valid GitHub repository URL.
    """
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Check if it's a GitHub URL
    if parsed_url.netloc != 'github.com':
        raise ValueError(f"Not a GitHub URL: {url}")
    
    # Extract the path
    path = parsed_url.path.strip('/')
    
    # Remove .git suffix if present
    if path.endswith('.git'):
        path = path[:-4]  # Remove the last 4 characters ('.git')
    
    # Split the path and check if it has at least two components (owner and repo)
    path_components = path.split('/')
    if len(path_components) < 2:
        raise ValueError(f"Invalid GitHub repository URL: {url}")
    
    # Return the owner/repo format
    return f"{path_components[0]}/{path_components[1]}"

def normalize_repo_input(repo_input: str) -> str:
    """
    Normalize the repository input, which can be either a URL or a repository name.
    
    Args:
        repo_input: Repository input, either a URL or a name in the format 'owner/repo'.
        
    Returns:
        Repository name in the format 'owner/repo'.
        
    Raises:
        ValueError: If the input is not a valid GitHub repository URL or name.
    """
    # Check if it's a URL
    if repo_input.startswith('http'):
        return extract_repo_from_url(repo_input)
    
    # If it's in the format 'owner/repo.git', remove the .git suffix
    if repo_input.endswith('.git'):
        repo_input = repo_input[:-4]
    
    # Check if it's in the format 'owner/repo'
    if re.match(r'^[^/]+/[^/]+$', repo_input):
        return repo_input
    
    raise ValueError(f"Invalid repository format: {repo_input}. Expected a GitHub URL or 'owner/repo' format.")

def download_issues(repo_name: str, github_token: str, db_path: str, resume: bool = True) -> None:
    """
    Download issues from a GitHub repository and store them in an SQLite database.
    
    Args:
        repo_name: Repository name in the format 'owner/repo'.
        github_token: GitHub API token.
        db_path: Path to the SQLite database.
        resume: Whether to resume from the last successful batch.
    """
    logger.info(f"Downloading issues for repository: {repo_name}")
    
    # Initialize the GitHub issues fetcher
    fetcher = GitHubIssuesFetcher(github_token)
    
    # Initialize the SQLite issue storage
    storage = SQLiteIssueStorage(db_path)
    
    # Create tables if they don't exist
    storage.create_tables()
    
    # Get existing issue numbers
    existing_issue_numbers = set(storage.get_issue_numbers(repo_name))
    logger.info(f"Found {len(existing_issue_numbers)} existing issues in the database")
    
    # If we have existing issues and we're not explicitly asking to resume, skip downloading
    if existing_issue_numbers and not resume:
        logger.info(f"Skipping download as {len(existing_issue_numbers)} issues already exist and resume=False")
        return
    
    # Start timing the execution
    start_time = time.time()
    
    # Track statistics
    total_issues = 0
    total_new = 0
    total_updated = 0
    total_redundant = 0
    
    # Configure the fetcher to use a callback for batch processing
    def process_batch(batch_issues, batch_stats):
        nonlocal total_issues, total_new, total_updated, total_redundant
        
        # Update counts
        total_issues += len(batch_issues)
        total_new += batch_stats.get("new_count", 0)
        total_updated += batch_stats.get("updated_count", 0)
        total_redundant += batch_stats.get("redundant_count", 0)
        
        # Store this batch in the database
        if batch_issues:
            storage.store_issues(batch_issues, repo_name)
            
            # Log progress
            if total_issues % 100 == 0:
                logger.info(f"Processed {total_issues} issues: {total_new} new, {total_updated} updated, {total_redundant} redundant")
    
    # Fetch issues with batch processing
    fetcher.fetch_issues(
        repo_name=repo_name,
        state="all",
        include_comments=True,
        existing_issue_numbers=existing_issue_numbers,
        batch_size=100,
        batch_callback=process_batch
    )
    
    logger.info(f"Fetched {total_issues} issues: {total_new} new, {total_updated} updated, {total_redundant} redundant")
    
    # Get API rate limit info
    rate_limit_info = fetcher.get_rate_limit_info()
    remaining = rate_limit_info.get("remaining", 0)
    limit = rate_limit_info.get("limit", 0)
    reset_time = rate_limit_info.get("reset_time", "")
    
    logger.info(f"API Rate Limit: {remaining}/{limit}")
    logger.info(f"Rate limit resets at: {reset_time}")
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Log the API call
    storage.log_api_call(
        repo_name=repo_name,
        new_issues_count=total_new,
        updated_issues_count=total_updated,
        redundant_issues_count=total_redundant,
        issues_before_count=storage.get_issue_count(repo_name),
        issues_after_count=storage.get_issue_count(repo_name),
        api_rate_limit_remaining=remaining,
        api_rate_limit_total=limit,
        execution_time_seconds=execution_time,
    )
    
    logger.info(f"Total execution time for downloading: {execution_time:.2f} seconds")

def analyze_repository_data(sqlite_db_path: str, chroma_db_path: str, collection_name: str) -> dict:
    """
    Analyze the repository data to provide statistics about issues, comments, and chunks.
    
    Args:
        sqlite_db_path: Path to the SQLite database.
        chroma_db_path: Path to the ChromaDB database.
        collection_name: Name of the ChromaDB collection.
        
    Returns:
        Dictionary containing statistics about the repository data.
    """
    import sqlite3
    import chromadb
    import numpy as np
    from collections import Counter
    
    logger = logging.getLogger(__name__)
    logger.info("Analyzing repository data...")
    
    stats = {}
    
    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()
    
    # Get issue statistics
    cursor.execute("SELECT COUNT(*) FROM issues")
    issue_count = cursor.fetchone()[0]
    stats["issue_count"] = issue_count
    
    # Get comment statistics
    cursor.execute("SELECT COUNT(*) FROM comments")
    comment_count = cursor.fetchone()[0]
    stats["comment_count"] = comment_count
    
    # Calculate average comments per issue
    if issue_count > 0:
        stats["avg_comments_per_issue"] = comment_count / issue_count
    else:
        stats["avg_comments_per_issue"] = 0
    
    # Get issue with most comments
    cursor.execute("""
        SELECT i.number, i.title, COUNT(c.id) as comment_count
        FROM issues i
        LEFT JOIN comments c ON i.id = c.issue_id
        GROUP BY i.id
        ORDER BY comment_count DESC
        LIMIT 1
    """)
    most_commented = cursor.fetchone()
    if most_commented:
        stats["most_commented_issue"] = {
            "number": most_commented[0],
            "title": most_commented[1],
            "comment_count": most_commented[2]
        }
    
    # Get distribution of issue states
    cursor.execute("SELECT state, COUNT(*) FROM issues GROUP BY state")
    state_counts = cursor.fetchall()
    stats["issue_states"] = {state: count for state, count in state_counts}
    
    # Get issues by month
    cursor.execute("""
        SELECT strftime('%Y-%m', created_at) as month, COUNT(*) as count
        FROM issues
        GROUP BY month
        ORDER BY month
    """)
    monthly_counts = cursor.fetchall()
    stats["issues_by_month"] = {month: count for month, count in monthly_counts}
    
    # Check if user_login column exists in issues table
    cursor.execute("PRAGMA table_info(issues)")
    columns = [column[1] for column in cursor.fetchall()]
    
    # Get top issue authors if user_login column exists
    if "user_login" in columns:
        cursor.execute("""
            SELECT user_login, COUNT(*) as count
            FROM issues
            GROUP BY user_login
            ORDER BY count DESC
            LIMIT 5
        """)
        top_authors = cursor.fetchall()
        stats["top_issue_authors"] = {author: count for author, count in top_authors}
    
    # Close SQLite connection
    conn.close()
    
    # Get ChromaDB statistics
    try:
        # Initialize ChromaDB client directly
        client = chromadb.PersistentClient(path=chroma_db_path)
        collection = client.get_collection(collection_name)
        
        # Get chunk count
        chunk_count = collection.count()
        stats["chunk_count"] = chunk_count
        
        # Get metadata for all chunks to analyze
        if chunk_count > 0:
            all_chunks = collection.get()
            
            # Print out metadata fields from the first few chunks
            logger.info("Checking metadata fields in the first few chunks:")
            for i, metadata in enumerate(all_chunks["metadatas"][:5]):
                logger.info(f"Chunk {i} metadata: {metadata.keys()}")
                if metadata:
                    logger.info(f"Chunk {i} metadata values: {metadata}")
            
            # Extract chunk sizes and token counts from metadata
            chunk_sizes = []
            token_counts = []
            char_lengths = []
            issue_ids = []
            
            for i, (metadata, document) in enumerate(zip(all_chunks["metadatas"], all_chunks["documents"])):
                if metadata:
                    if "chunk_size" in metadata:
                        chunk_sizes.append(metadata["chunk_size"])
                    if "token_count" in metadata:
                        token_counts.append(metadata["token_count"])
                    if "issue_id" in metadata:
                        issue_ids.append(metadata["issue_id"])
                    
                    # Calculate character length from document
                    if document:
                        char_lengths.append(len(document))
            
            # Analyze chunk sizes
            if chunk_sizes:
                stats["chunk_size_stats"] = {
                    "min": min(chunk_sizes),
                    "max": max(chunk_sizes),
                    "avg": sum(chunk_sizes) / len(chunk_sizes),
                    "median": np.median(chunk_sizes),
                    "std_dev": np.std(chunk_sizes),
                    "total": sum(chunk_sizes),
                    "count": len(chunk_sizes)
                }
                
                # Get distribution of chunk sizes
                chunk_size_counter = Counter(chunk_sizes)
                stats["chunk_size_distribution"] = dict(chunk_size_counter.most_common(10))
            
            # Analyze token counts
            if token_counts:
                stats["token_count_stats"] = {
                    "min": min(token_counts),
                    "max": max(token_counts),
                    "avg": sum(token_counts) / len(token_counts),
                    "median": np.median(token_counts),
                    "std_dev": np.std(token_counts),
                    "total": sum(token_counts),
                    "count": len(token_counts)
                }
            else:
                # If no token_count in metadata, estimate tokens using character count
                if char_lengths:
                    # Rough estimate: 1 token ≈ 4 characters
                    estimated_tokens = [max(1, int(length / 4)) for length in char_lengths]
                    stats["estimated_token_count_stats"] = {
                        "min": min(estimated_tokens),
                        "max": max(estimated_tokens),
                        "avg": sum(estimated_tokens) / len(estimated_tokens),
                        "median": np.median(estimated_tokens),
                        "std_dev": np.std(estimated_tokens),
                        "total": sum(estimated_tokens),
                        "count": len(estimated_tokens)
                    }
            
            # Analyze character lengths
            if char_lengths:
                stats["char_length_stats"] = {
                    "min": min(char_lengths),
                    "max": max(char_lengths),
                    "avg": sum(char_lengths) / len(char_lengths),
                    "median": np.median(char_lengths),
                    "std_dev": np.std(char_lengths),
                    "total": sum(char_lengths),
                    "count": len(char_lengths)
                }
            
            # Analyze chunks per issue
            if issue_ids:
                chunks_per_issue = Counter(issue_ids)
                chunks_per_issue_values = list(chunks_per_issue.values())
                stats["chunks_per_issue_stats"] = {
                    "min": min(chunks_per_issue_values),
                    "max": max(chunks_per_issue_values),
                    "avg": sum(chunks_per_issue_values) / len(chunks_per_issue),
                    "median": np.median(chunks_per_issue_values),
                    "std_dev": np.std(chunks_per_issue_values),
                    "total_issues_with_chunks": len(chunks_per_issue)
                }
                
                # Issues with most chunks
                stats["issues_with_most_chunks"] = dict(chunks_per_issue.most_common(5))
    except Exception as e:
        logger.error(f"Error analyzing ChromaDB: {e}")
        stats["chroma_error"] = str(e)
    
    return stats

def main():
    """
    Main function to process a repository.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Process a GitHub repository")
    parser.add_argument("repository", help="GitHub repository URL or 'owner/repo' format")
    parser.add_argument("--token", help="GitHub API token")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap in tokens")
    parser.add_argument("--limit", type=int, help="Limit the number of issues to process")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading issues")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding issues")
    parser.add_argument("--collection-name", default="github_issues", help="Collection name for ChromaDB (default: github_issues)")
    args = parser.parse_args()
    
    # Extract repository name from URL if needed
    if args.repository.startswith("http"):
        repo_name = extract_repo_from_url(args.repository)
    else:
        repo_name = args.repository
    
    logger.info(f"Using repository: {repo_name}")
    
    # Get GitHub token from environment if not provided
    github_token = args.token or os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.error("GitHub API token not provided. Use --token or set GITHUB_TOKEN environment variable.")
        return 1
    
    # Get database paths
    sqlite_db_path = get_sqlite_db_path(repo_name)
    chroma_db_path = get_chroma_db_path(repo_name)
    
    logger.info(f"SQLite database path: {sqlite_db_path}")
    logger.info(f"ChromaDB database path: {chroma_db_path}")
    
    # Download issues if not skipped
    if not args.skip_download:
        logger.info(f"Downloading issues for repository: {repo_name}")
        download_issues(repo_name, github_token, sqlite_db_path)
    
    # Embed issues if not skipped
    if not args.skip_embed:
        logger.info(f"Embedding issues for repository: {repo_name}")
        embed_database_to_chromadb(
            repo_name=repo_name,
            sqlite_db_path=sqlite_db_path,
            chroma_db_path=chroma_db_path,
            collection_name=args.collection_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            limit_issues=args.limit
        )
    
    # Analyze the repository data
    logger.info(f"Analyzing repository data: {repo_name}")
    stats = analyze_repository_data(
        sqlite_db_path=sqlite_db_path,
        chroma_db_path=chroma_db_path,
        collection_name=args.collection_name
    )
    
    # Print the statistics
    logger.info(f"Repository statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    return 0

if __name__ == "__main__":
    main()
