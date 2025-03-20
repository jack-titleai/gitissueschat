"""
Process GitHub Issues into Chunks

This script processes GitHub issues from a SQLite database into chunks for embedding.
"""

import argparse
import sqlite3
import json
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from gitissueschat.embed.llamaindex_chunker import LlamaIndexChunker


def get_repository_issues(db_path: str, repo_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve all issues and their comments from a repository in the SQLite database.
    
    Args:
        db_path: Path to the SQLite database.
        repo_name: Repository name in the format 'owner/repo'.
        limit: Optional limit on the number of issues to retrieve.
        
    Returns:
        A list of issue dictionaries with comments.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get repository ID
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM repositories WHERE name = ?", (repo_name,))
    repo_row = cursor.fetchone()
    
    if not repo_row:
        raise ValueError(f"Repository {repo_name} not found in database")
    
    repo_id = repo_row[0]
    
    # Get issues
    limit_clause = f"LIMIT {limit}" if limit else ""
    cursor.execute(
        f"""
        SELECT * FROM issues 
        WHERE repo_id = ?
        ORDER BY number
        {limit_clause}
        """, 
        (repo_id,)
    )
    issue_rows = cursor.fetchall()
    
    issues = []
    for issue_row in tqdm(issue_rows, desc="Processing issues"):
        # Convert row to dict
        issue = dict(issue_row)
        issue["repository"] = repo_name
        
        # Get comments for this issue
        cursor.execute(
            """
            SELECT * FROM comments
            WHERE issue_id = ?
            ORDER BY created_at
            """,
            (issue["id"],)
        )
        comment_rows = cursor.fetchall()
        
        # Convert rows to dicts
        comments = [dict(row) for row in comment_rows]
        issue["comments"] = comments
        
        issues.append(issue)
    
    conn.close()
    
    return issues


def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save chunks to a JSONL file.
    
    Args:
        chunks: List of chunk dictionaries.
        output_path: Path to the output JSONL file.
    """
    with open(output_path, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')


def main():
    """
    Main function to process GitHub issues into chunks.
    """
    parser = argparse.ArgumentParser(description="Process GitHub issues into chunks for embedding")
    parser.add_argument("--db-path", required=True, help="Path to the SQLite database")
    parser.add_argument("--repo-name", required=True, help="Repository name in the format 'owner/repo'")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file")
    parser.add_argument("--chunk-size", type=int, default=250, help="Target size of each chunk in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Number of tokens to overlap between chunks")
    parser.add_argument("--issue-context", type=int, default=100, 
                       help="Number of characters from the issue to include in comment chunks")
    parser.add_argument("--limit", type=int, help="Limit the number of issues to process")
    
    args = parser.parse_args()
    
    # Create chunker
    chunker = LlamaIndexChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        issue_context_chars=args.issue_context
    )
    
    # Get issues from database
    print(f"Retrieving issues from {args.repo_name}...")
    issues = get_repository_issues(args.db_path, args.repo_name, args.limit)
    print(f"Retrieved {len(issues)} issues with their comments")
    
    # Process issues and comments into chunks
    all_chunks = []
    issue_chunk_count = 0
    comment_chunk_count = 0
    
    for issue in tqdm(issues, desc="Chunking issues"):
        chunks = chunker.process_issue_with_comments(issue)
        
        # Count chunks by type
        for chunk in chunks:
            if chunk["type"] == "issue":
                issue_chunk_count += 1
            else:
                comment_chunk_count += 1
                
        all_chunks.extend(chunks)
    
    # Save chunks to JSONL file
    save_chunks_to_jsonl(all_chunks, args.output)
    
    print(f"Processed {len(issues)} issues into {len(all_chunks)} chunks:")
    print(f"- Issue chunks: {issue_chunk_count}")
    print(f"- Comment chunks: {comment_chunk_count}")
    print(f"Saved chunks to {args.output}")


if __name__ == "__main__":
    main()
