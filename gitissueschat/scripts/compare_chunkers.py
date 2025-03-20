"""
Compare different configurations of the LlamaIndexChunker on a GitHub issue with long comments.
"""

import sqlite3
import json
import argparse
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


def analyze_chunks(chunks: List[Dict[str, Any]], chunker_name: str):
    """
    Analyze the chunks and print statistics.
    
    Args:
        chunks: List of chunks to analyze.
        chunker_name: Name of the chunker used.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Count tokens in each chunk
    token_counts = []
    char_counts = []
    
    for chunk in chunks:
        text = chunk["text"]
        token_count = len(tokenizer.encode(text))
        char_count = len(text)
        
        token_counts.append(token_count)
        char_counts.append(char_count)
    
    # Calculate statistics
    if token_counts:
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        avg_tokens = sum(token_counts) / len(token_counts)
        
        min_chars = min(char_counts)
        max_chars = max(char_counts)
        avg_chars = sum(char_counts) / len(char_counts)
    else:
        min_tokens = max_tokens = avg_tokens = 0
        min_chars = max_chars = avg_chars = 0
    
    # Count chunks by type
    issue_chunks = [c for c in chunks if c["type"] == "issue"]
    comment_chunks = [c for c in chunks if c["type"] == "comment"]
    
    # Print results
    print(f"\n{chunker_name} results:")
    print(f"- Total chunks: {len(chunks)}")
    print(f"- Issue chunks: {len(issue_chunks)}")
    print(f"- Comment chunks: {len(comment_chunks)}")
    print(f"- Token count - Min: {min_tokens}, Max: {max_tokens}, Avg: {avg_tokens:.1f}")
    print(f"- Character count - Min: {min_chars}, Max: {max_chars}, Avg: {avg_chars:.1f}")
    
    # Print chunks per comment
    comment_ids = set(c["comment_id"] for c in comment_chunks)
    for comment_id in comment_ids:
        comment_chunks_count = len([c for c in comment_chunks if c["comment_id"] == comment_id])
        print(f"  - Comment {comment_id}: {comment_chunks_count} chunks")


def save_chunks_to_file(chunks: List[Dict[str, Any]], output_file: str):
    """
    Save chunks to a JSONL file.
    
    Args:
        chunks: List of chunks to save.
        output_file: Path to the output file.
    """
    with open(output_file, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')
    
    print(f"Saved {len(chunks)} chunks to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare different configurations of LlamaIndexChunker")
    parser.add_argument("--db", required=True, help="Path to the SQLite database")
    parser.add_argument("--repo", required=True, help="Repository name in the format 'owner/repo'")
    parser.add_argument("--issue", type=int, required=True, help="Issue number to process")
    parser.add_argument("--output-prefix", default="chunks", help="Prefix for output files")
    
    args = parser.parse_args()
    
    # Get the issue with comments
    issue = get_issue_with_comments(args.db, args.repo, args.issue)
    
    print(f"Retrieved issue #{args.issue} with {len(issue['comments'])} comments")
    
    # Print comment sizes
    for i, comment in enumerate(issue["comments"]):
        print(f"Comment {i+1}: {len(comment['body'])} characters")
    
    # Test different configurations
    configurations = [
        {"name": "Small Chunks", "chunk_size": 100, "chunk_overlap": 20},
        {"name": "Medium Chunks", "chunk_size": 250, "chunk_overlap": 50},
        {"name": "Large Chunks", "chunk_size": 500, "chunk_overlap": 100},
        {"name": "Large Overlap", "chunk_size": 250, "chunk_overlap": 125},
    ]
    
    for config in configurations:
        name = config["name"]
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        
        print(f"\n=== Testing {name} (size={chunk_size}, overlap={chunk_overlap}) ===")
        
        # Initialize chunker
        chunker = LlamaIndexChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Process with LlamaIndex chunker
        chunks = chunker.process_issue_with_comments(issue.copy())
        
        # Analyze the chunks
        analyze_chunks(chunks, name)
        
        # Save chunks if output file is specified
        if args.output_prefix:
            output_file = f"{args.output_prefix}-{chunk_size}-{chunk_overlap}.jsonl"
            save_chunks_to_file(chunks, output_file)


if __name__ == "__main__":
    main()
