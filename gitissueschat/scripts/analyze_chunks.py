"""
Analyze chunk sizes in a JSONL file.
"""

import json
import sys
import tiktoken
import statistics
from collections import defaultdict

def analyze_chunks(jsonl_file, target_size=None):
    """Analyze the token counts of chunks in a JSONL file."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Analyzing {len(lines)} chunks in {jsonl_file}")
    print("-" * 80)
    
    # Track statistics
    token_counts = []
    char_counts = []
    type_counts = defaultdict(int)
    id_to_tokens = {}
    
    # Track chunks by comment ID
    comment_chunks = defaultdict(list)
    
    for i, line in enumerate(lines):
        data = json.loads(line)
        text = data["text"]
        chunk_type = data["type"]
        chunk_id = data["id"]
        
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        char_count = len(text)
        
        token_counts.append(token_count)
        char_counts.append(char_count)
        type_counts[chunk_type] += 1
        id_to_tokens[chunk_id] = token_count
        
        # Group chunks by their base ID (before the _X suffix)
        if chunk_type == "comment":
            base_id = chunk_id.split("_comment_")[0]
            comment_chunks[base_id].append((chunk_id, token_count, char_count))
    
    # Print statistics
    print("Token count statistics:")
    print(f"  Min: {min(token_counts)}")
    print(f"  Max: {max(token_counts)}")
    print(f"  Mean: {statistics.mean(token_counts):.1f}")
    print(f"  Median: {statistics.median(token_counts)}")
    print(f"  Standard deviation: {statistics.stdev(token_counts):.1f}")
    print()
    
    print("Character count statistics:")
    print(f"  Min: {min(char_counts)}")
    print(f"  Max: {max(char_counts)}")
    print(f"  Mean: {statistics.mean(char_counts):.1f}")
    print(f"  Median: {statistics.median(char_counts)}")
    print(f"  Standard deviation: {statistics.stdev(char_counts):.1f}")
    print()
    
    print("Chunk type counts:")
    for chunk_type, count in type_counts.items():
        print(f"  {chunk_type}: {count}")
    print()
    
    # Check for chunks exceeding target size
    if target_size:
        oversized_chunks = [(id, count) for id, count in id_to_tokens.items() if count > target_size]
        if oversized_chunks:
            print(f"WARNING: Found {len(oversized_chunks)} chunks exceeding target size of {target_size} tokens:")
            for chunk_id, token_count in sorted(oversized_chunks, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {chunk_id}: {token_count} tokens (exceeds by {token_count - target_size})")
        else:
            print(f"All chunks are within the target size of {target_size} tokens.")
    print()
    
    # Print information about the longest comments
    print("Comments with the most chunks:")
    for base_id, chunks in sorted(comment_chunks.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        total_tokens = sum(tc for _, tc, _ in chunks)
        print(f"  Comment {base_id}: {len(chunks)} chunks, {total_tokens} total tokens")
        for chunk_id, token_count, char_count in chunks:
            print(f"    - {chunk_id}: {token_count} tokens, {char_count} characters")
    
    # Print information about specific comments if requested
    if len(sys.argv) > 2:
        comment_id = sys.argv[2]
        if comment_id in comment_chunks:
            print(f"\nDetailed analysis of comment {comment_id}:")
            chunks = comment_chunks[comment_id]
            for chunk_id, token_count, char_count in chunks:
                print(f"  {chunk_id}: {token_count} tokens, {char_count} characters")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_chunks.py <jsonl_file> [comment_id] [target_size]")
        sys.exit(1)
    
    target_size = None
    if len(sys.argv) > 3:
        target_size = int(sys.argv[3])
    
    analyze_chunks(sys.argv[1], target_size)
