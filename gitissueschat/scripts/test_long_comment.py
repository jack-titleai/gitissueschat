"""
Test script for chunking a specific issue with long comments.
"""

import sqlite3
import json
import tiktoken
from typing import Dict, List, Any
from gitissueschat.embed.llamaindex_chunker import LlamaIndexChunker


def get_issue_with_comments(db_path: str, repo_name: str, issue_number: int) -> Dict[str, Any]:
    """
    Retrieve an issue and its comments from the SQLite database.
    
    Args:
        db_path: Path to the SQLite database.
        repo_name: Repository name in the format 'owner/repo'.
        issue_number: Issue number to retrieve.
        
    Returns:
        A dictionary containing the issue data with comments.
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
    
    # Get issue
    cursor.execute(
        """
        SELECT * FROM issues 
        WHERE repo_id = ? AND number = ?
        """, 
        (repo_id, issue_number)
    )
    issue_row = cursor.fetchone()
    
    if not issue_row:
        raise ValueError(f"Issue #{issue_number} not found in repository {repo_name}")
    
    # Convert to dictionary
    issue = {key: issue_row[key] for key in issue_row.keys()}
    issue["repository"] = repo_name
    
    # Get comments for the issue
    cursor.execute(
        """
        SELECT * FROM comments 
        WHERE issue_id = ? 
        ORDER BY created_at
        """, 
        (issue["id"],)
    )
    comments = []
    
    for row in cursor:
        comment = {key: row[key] for key in row.keys()}
        comments.append(comment)
    
    # Add comments to the issue
    issue["comments"] = comments
    
    conn.close()
    
    return issue


def analyze_chunks(chunks, chunker_name):
    """Analyze the token counts of chunks."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    token_counts = []
    for chunk in chunks:
        token_count = len(tokenizer.encode(chunk["text"]))
        token_counts.append(token_count)
    
    if token_counts:
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        avg_tokens = sum(token_counts) / len(token_counts)
    else:
        min_tokens = max_tokens = avg_tokens = 0
    
    print(f"\n{chunker_name} Results:")
    print(f"Total chunks: {len(chunks)}")
    print(f"Token counts - Min: {min_tokens}, Max: {max_tokens}, Avg: {avg_tokens:.1f}")
    
    # Count chunks by type
    issue_chunks = [c for c in chunks if c["type"] == "issue"]
    comment_chunks = [c for c in chunks if c["type"] == "comment"]
    
    print(f"Issue chunks: {len(issue_chunks)}")
    print(f"Comment chunks: {len(comment_chunks)}")
    
    # Print chunks per comment
    comment_ids = set(c.get("comment_id") for c in comment_chunks)
    for comment_id in comment_ids:
        comment_chunks_count = len([c for c in comment_chunks if c.get("comment_id") == comment_id])
        print(f"  - Comment {comment_id}: {comment_chunks_count} chunks")


def main():
    """
    Main function to test chunking on a specific issue with long comments.
    """
    # Configuration
    db_path = "fastai-issues.db"
    repo_name = "fastai/fastai"
    issue_number = 2769  # This issue has a very long comment
    
    # Get the issue with comments
    issue = get_issue_with_comments(db_path, repo_name, issue_number)
    
    print(f"Retrieved issue #{issue_number} with {len(issue['comments'])} comments")
    
    # Print comment sizes
    for i, comment in enumerate(issue["comments"]):
        print(f"Comment {i+1}: {len(comment['body'])} characters")
    
    # Test different chunk sizes
    for chunk_size in [100, 250, 500]:
        print(f"\n=== Testing LlamaIndex chunker with chunk_size={chunk_size} ===")
        
        # Initialize chunker
        chunker = LlamaIndexChunker(
            chunk_size=chunk_size,
            chunk_overlap=50
        )
        
        # Process the issue
        chunks = chunker.process_issue_with_comments(issue.copy())
        
        # Analyze the chunks
        analyze_chunks(chunks, f"LlamaIndex Chunker (size={chunk_size})")
        
        # Save chunks to file
        output_file = f"issue2769-chunks-{chunk_size}.jsonl"
        with open(output_file, 'w') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + '\n')
        
        print(f"Saved {len(chunks)} chunks to {output_file}")


if __name__ == "__main__":
    main()
