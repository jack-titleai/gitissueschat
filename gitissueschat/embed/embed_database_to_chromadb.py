"""
Embed a full database of issues into ChromaDB.

This script processes all issues from a SQLite database, chunks them,
and embeds the chunks into ChromaDB in batches.
"""

import argparse
import json
import logging
import os
import sqlite3
import time
from typing import Dict, List, Any, Optional, Tuple

from tqdm import tqdm

from gitissueschat.embed.chroma_database import ChunksDatabase
from gitissueschat.embed.llamaindex_chunker import LlamaIndexChunker
from gitissueschat.utils.db_path_manager import get_sqlite_db_path, get_chroma_db_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_repositories(db_path: str) -> List[Dict[str, Any]]:
    """
    Get all repositories from the database.
    
    Args:
        db_path: Path to the SQLite database.
        
    Returns:
        List of repositories.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM repositories")
    repositories = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return repositories


def get_issues_for_repository(db_path: str, repo_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get all issues for a repository.
    
    Args:
        db_path: Path to the SQLite database.
        repo_id: Repository ID.
        limit: Optional limit on the number of issues to retrieve.
        
    Returns:
        List of issues.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = "SELECT * FROM issues WHERE repo_id = ?"
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query, (repo_id,))
    issues = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return issues


def get_comments_for_issue(db_path: str, issue_id: int) -> List[Dict[str, Any]]:
    """
    Get all comments for an issue.
    
    Args:
        db_path: Path to the SQLite database.
        issue_id: Issue ID.
        
    Returns:
        List of comments.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM comments WHERE issue_id = ?", (issue_id,))
    comments = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return comments


def process_issue_with_comments(
    db_path: str, 
    repo_name: str, 
    issue: Dict[str, Any], 
    chunker: LlamaIndexChunker
) -> List[Dict[str, Any]]:
    """
    Process an issue with its comments and chunk them.
    
    Args:
        db_path: Path to the SQLite database.
        repo_name: Repository name.
        issue: Issue data.
        chunker: Chunker instance.
        
    Returns:
        List of chunks.
    """
    # Get comments for the issue
    comments = get_comments_for_issue(db_path, issue["id"])
    
    # Prepare issue data for chunking
    issue_data = {
        "id": issue["id"],
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"],
        "created_at": issue["created_at"],
        "updated_at": issue["updated_at"],
        "html_url": issue["html_url"],
        "repository": repo_name,
        "comments": []
    }
    
    # Add comments to issue data
    for comment in comments:
        issue_data["comments"].append({
            "id": comment["id"],
            "body": comment["body"],
            "created_at": comment["created_at"],
            "updated_at": comment["updated_at"],
            "user": {"login": comment["author"]}
        })
    
    # Process the issue and its comments
    chunks = chunker.process_issue_with_comments(issue_data)
    
    return chunks


def embed_database_to_chromadb(
    repo_name: str,
    sqlite_db_path: Optional[str] = None,
    chroma_db_path: Optional[str] = None,
    collection_name: str = "github_issues",
    chunk_size: int = 250,
    chunk_overlap: int = 50,
    batch_size: int = 100,
    project_id: Optional[str] = None,
    api_key: Optional[str] = None,
    credentials_path: Optional[str] = None,
    limit_issues: Optional[int] = None,
    resume: bool = True
) -> Dict[str, Any]:
    """
    Process all issues from a SQLite database, chunk them, and embed them into ChromaDB.
    
    Args:
        repo_name: Name of the repository.
        sqlite_db_path: Path to the SQLite database. If None, uses the default path.
        chroma_db_path: Path to the ChromaDB database. If None, uses the default path.
        collection_name: Name of the collection to use.
        chunk_size: Size of chunks in tokens.
        chunk_overlap: Overlap between chunks in tokens.
        batch_size: Number of chunks to process at once.
        project_id: Google Cloud project ID for Vertex AI.
        api_key: Google Cloud API key for Vertex AI.
        credentials_path: Path to Google Cloud credentials file.
        limit_issues: Maximum number of issues to process.
        resume: Whether to resume from the last processed issue.
        
    Returns:
        Dictionary with statistics about the embedding process.
    """
    # Set up paths
    if sqlite_db_path is None:
        db_path = get_sqlite_db_path(repo_name)
    else:
        db_path = sqlite_db_path
    
    if chroma_db_path is None:
        chroma_path = get_chroma_db_path(repo_name)
    else:
        chroma_path = chroma_db_path
    
    # Initialize the embedding client
    embedding_client = ChunksDatabase(
        db_path=chroma_path,
        collection_name=collection_name,
        project_id=project_id,
        api_key=api_key,
        credentials_path=credentials_path
    )
    
    # Initialize the chunker
    chunker = LlamaIndexChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Get all repositories
    repositories = get_repositories(db_path)
    logger.info(f"Found {len(repositories)} repositories in the database")
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get the repository ID
    cursor.execute("SELECT id FROM repositories WHERE name = ?", (repo_name,))
    repo_row = cursor.fetchone()
    if not repo_row:
        logger.error(f"Repository {repo_name} not found in the database")
        conn.close()
        return {"error": "Repository not found"}
    
    repo_id = repo_row["id"]
    
    # Get existing document IDs from ChromaDB to avoid re-embedding
    existing_doc_ids = set()
    try:
        # Get all document IDs from the collection
        existing_docs = embedding_client.get_collection().get()
        if existing_docs and "ids" in existing_docs:
            existing_doc_ids = set(existing_docs["ids"])
            logger.info(f"Found {len(existing_doc_ids)} existing documents in ChromaDB")
    except Exception as e:
        logger.warning(f"Error getting existing documents from ChromaDB: {e}")
    
    # Process each repository
    total_chunks = 0
    total_issues = 0
    total_embedded = 0
    total_skipped = 0
    
    # Batch processing variables
    current_batch = []
    batch_count = 0
    
    # Get all issues for the repository
    query = """
        SELECT i.id, i.number, i.title, i.body, i.created_at, i.updated_at, i.state,
               i.author, i.html_url
        FROM issues i
        WHERE i.repo_id = ?
    """
    
    if limit_issues:
        query += f" LIMIT {limit_issues}"
    
    cursor.execute(query, (repo_id,))
    issues = cursor.fetchall()
    
    logger.info(f"Found {len(issues)} issues for repository {repo_name}")
    
    # Process each issue
    for issue in tqdm(issues, desc="Processing issues"):
        issue_id = issue["id"]
        issue_number = issue["number"]
        
        # Generate a unique ID for this issue
        issue_doc_id = f"issue-{repo_name}-{issue_number}"
        
        # Skip if this issue has already been embedded
        if resume and issue_doc_id in existing_doc_ids:
            logger.debug(f"Skipping issue #{issue_number} (already embedded)")
            total_skipped += 1
            continue
        
        # Process the issue text
        issue_doc_id = f"issue-{repo_name}-{issue_number}"
        
        # Format issue text with clear structure and add updated timestamp at the end
        issue_text = f"## Issue #{issue_number}: {issue['title']}\n\nCreated by {issue['author']} on {issue['created_at']}\n\n{issue['body']}\n\nLast updated: {issue['updated_at']}"
        
        # Get the first 100 characters of the issue body for context in comments
        issue_context = issue['body'][:100] + "..." if issue['body'] and len(issue['body']) > 100 else issue['body'] or ""
        
        # Chunk the issue text
        issue_chunks = chunker._split_text_into_chunks(issue_text)
        
        # Create metadata for each issue chunk
        for i, chunk in enumerate(issue_chunks):
            chunk_id = f"{issue_doc_id}-issue-chunk-{i}"
            metadata = {
                "repo_name": repo_name,
                "issue_number": issue_number,
                "issue_title": issue["title"],
                "issue_state": issue["state"],
                "issue_author": issue["author"],
                "issue_url": issue["html_url"],
                "chunk_index": i,
                "total_chunks": len(issue_chunks),
                "content_type": "issue"
            }
            
            # Create chunk object
            chunk_object = {
                "id": chunk_id,
                "text": chunk,
                "type": "issue",
                "repository": repo_name,
                "issue_number": issue_number,
                "issue_id": issue_id,
                "metadata": {
                    **metadata,
                    "created_at": issue["created_at"],
                    "updated_at": issue["updated_at"]
                }
            }
            current_batch.append(chunk_object)
            total_chunks += 1
            
            # If we've reached the batch size, process the batch
            if len(current_batch) >= batch_size:
                embedding_client.add_chunks(current_batch)
                logger.debug(f"Processed batch {batch_count} with {len(current_batch)} chunks")
                batch_count += 1
                current_batch = []
        
        # Get comments for this issue
        cursor.execute(
            """
            SELECT c.id, c.body, c.created_at, c.updated_at, c.author
            FROM comments c
            WHERE c.issue_id = ?
            ORDER BY c.created_at ASC
            """,
            (issue_id,)
        )
        comments = cursor.fetchall()
        
        # Process each comment separately
        for comment_idx, comment in enumerate(comments):
            # Include issue context but avoid timestamp duplication
            # The comment already has a timestamp in its header, so we don't need to repeat it in the content
            comment_text = f"## Comment by {comment['author']} on {comment['created_at']}\n\nContext from issue #{issue_number}:\n{issue_context}\n\n{comment['body']}\n\nLast updated: {comment['updated_at']}"
            
            # Chunk the comment text if needed
            comment_chunks = chunker._split_text_into_chunks(comment_text)
            
            # Create metadata for each comment chunk
            for i, chunk in enumerate(comment_chunks):
                chunk_id = f"{issue_doc_id}-comment-{comment_idx}-chunk-{i}"
                metadata = {
                    "repo_name": repo_name,
                    "issue_number": issue_number,
                    "issue_title": issue["title"],
                    "issue_state": issue["state"],
                    "comment_author": comment["author"],
                    "chunk_index": i,
                    "total_chunks": len(comment_chunks),
                    "comment_index": comment_idx,
                    "content_type": "comment"
                }
                
                # Create chunk object
                chunk_object = {
                    "id": chunk_id,
                    "text": chunk,
                    "type": "comment",
                    "repository": repo_name,
                    "issue_number": issue_number,
                    "issue_id": issue_id,
                    "metadata": {
                        **metadata,
                        "created_at": comment["created_at"],
                        "updated_at": comment["updated_at"] if "updated_at" in comment else comment["created_at"]
                    }
                }
                current_batch.append(chunk_object)
                total_chunks += 1
                
                # If we've reached the batch size, process the batch
                if len(current_batch) >= batch_size:
                    embedding_client.add_chunks(current_batch)
                    logger.debug(f"Processed batch {batch_count} with {len(current_batch)} chunks")
                    batch_count += 1
                    current_batch = []
        
        total_embedded += 1
        
        # Log progress
        if total_embedded % 10 == 0:
            logger.debug(f"Embedded {total_embedded} issues ({total_chunks} chunks)")
    
    # Process any remaining chunks in the last batch
    if current_batch:
        embedding_client.add_chunks(current_batch)
        logger.debug(f"Processed final batch with {len(current_batch)} chunks")
    
    # Close the database connection
    conn.close()
    
    # Log the results
    logger.info(f"Embedded {total_embedded} issues ({total_chunks} chunks)")
    logger.info(f"Skipped {total_skipped} issues (already embedded)")
    
    return {
        "total_issues": len(issues),
        "total_embedded": total_embedded,
        "total_skipped": total_skipped,
        "total_chunks": total_chunks,
    }


def main():
    """
    Main function to embed a database of issues into ChromaDB.
    """
    parser = argparse.ArgumentParser(description="Embed a database of issues into ChromaDB")
    parser.add_argument("repo_name", help="Repository name (e.g., 'username/repo')")
    parser.add_argument("--sqlite-db-path", help="Path to the SQLite database (overrides default path)")
    parser.add_argument("--chroma-db-path", help="Path to the ChromaDB database (overrides default path)")
    parser.add_argument("--collection-name", default="github_issues", help="Name of the collection to use")
    parser.add_argument("--chunk-size", type=int, default=250, help="Size of chunks in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between chunks in tokens")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of chunks to process at once")
    parser.add_argument("--limit-issues", type=int, help="Limit on the number of issues to process")
    parser.add_argument("--resume", action="store_true", help="Resume from the last processed issue")
    
    args = parser.parse_args()
    
    # Get project ID from environment
    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    if not project_id:
        logger.error("GOOGLE_PROJECT_ID environment variable not set")
        return
    
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Get credentials from environment
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    
    # Embed the database
    stats = embed_database_to_chromadb(
        repo_name=args.repo_name,
        sqlite_db_path=args.sqlite_db_path,
        chroma_db_path=args.chroma_db_path,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        project_id=project_id,
        api_key=api_key,
        credentials_path=credentials_path,
        limit_issues=args.limit_issues,
        resume=args.resume
    )
    
    logger.info(f"Database statistics: {stats}")


if __name__ == "__main__":
    main()
