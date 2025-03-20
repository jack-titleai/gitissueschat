"""
Test script for the LlamaIndexChunker class.
"""

import json
import sqlite3
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


def main():
    """
    Main function to test the LlamaIndexChunker.
    """
    # Configuration
    db_path = "./fastai-issues.db"
    repo_name = "fastai/fastai"
    issue_number = 2769  # Issue with a very long comment
    output_path = "./chunks.jsonl"
    
    # Get issue with comments
    try:
        issue = get_issue_with_comments(db_path, repo_name, issue_number)
        print(f"Retrieved issue #{issue_number} with {len(issue.get('comments', []))} comments")
        
        # Create chunker
        chunker = LlamaIndexChunker(
            chunk_size=250,
            chunk_overlap=50,
            issue_context_chars=100
        )
        
        # Process issue and comments
        chunks = chunker.process_issue_with_comments(issue)
        
        # Count chunks by type
        issue_chunks = [c for c in chunks if c["type"] == "issue"]
        comment_chunks = [c for c in chunks if c["type"] == "comment"]
        
        print(f"Total chunks: {len(chunks)}")
        print(f"Issue chunks: {len(issue_chunks)}")
        print(f"Comment chunks: {len(comment_chunks)}")
        
        # Save chunks to file
        with open(output_path, 'w') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + '\n')
        
        print(f"Saved chunks to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
