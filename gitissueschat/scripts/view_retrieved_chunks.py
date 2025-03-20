"""
Script to view retrieved chunks from ChromaDB without using Gemini API.
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Any, Optional

from gitissueschat.embed.chroma_database import ChunksDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="View retrieved chunks from ChromaDB")
    
    # Database arguments
    parser.add_argument("--db-path", type=str, default="./chroma_db",
                        help="Path to the ChromaDB database")
    parser.add_argument("--collection-name", type=str, default="github_issues",
                        help="Name of the ChromaDB collection")
    
    # Query arguments
    parser.add_argument("--query", type=str, required=True,
                        help="Query to search for")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of chunks to retrieve")
    
    return parser.parse_args()

def format_chunk(i: int, chunk: Dict[str, Any]) -> str:
    """Format a chunk for display."""
    # Format metadata
    metadata_str = json.dumps(chunk.get("metadata", {}), indent=2)
    
    # Format content
    content = chunk.get("document", "")
    
    # Format similarity
    similarity = chunk.get("similarity", 0.0)
    
    return f"""
CHUNK {i+1} [similarity: {similarity:.4f}]
{'=' * 80}
METADATA:
{metadata_str}
{'=' * 80}
CONTENT:
{content}
{'=' * 80}
"""

def main():
    """Main entry point."""
    args = parse_args()
    
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return
    
    # Get project ID from environment
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")
    if not project_id:
        logger.error("GOOGLE_PROJECT_ID environment variable not set")
        return
    
    # Get credentials from environment
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    try:
        # Initialize the database
        db = ChunksDatabase(
            db_path=args.db_path,
            collection_name=args.collection_name,
            project_id=project_id,
            credentials_path=credentials
        )
        
        # Query the database
        logger.info(f"Querying database with: {args.query}")
        results = db.query(args.query, args.top_k)
        
        # Process results
        if "ids" in results and results["ids"] and len(results["ids"]) > 0:
            # ChromaDB returns nested lists for query results
            ids = results["ids"][0] if isinstance(results["ids"][0], list) else results["ids"]
            documents = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
            metadatas = results["metadatas"][0] if isinstance(results["metadatas"][0], list) else results["metadatas"]
            distances = results["distances"][0] if isinstance(results["distances"][0], list) else results["distances"]
            
            print(f"\nFound {len(ids)} chunks for query: '{args.query}'")
            
            # Format and display chunks
            for i in range(len(ids)):
                # Convert distance to similarity score (1 - distance)
                similarity = 1 - distances[i]
                
                chunk = {
                    "id": ids[i],
                    "document": documents[i],
                    "metadata": metadatas[i],
                    "similarity": similarity
                }
                
                print(format_chunk(i, chunk))
                
            # Save results to file
            output_file = "retrieved_chunks.json"
            with open(output_file, "w") as f:
                json.dump({
                    "query": args.query,
                    "chunks": [
                        {
                            "id": ids[i],
                            "document": documents[i],
                            "metadata": metadatas[i],
                            "similarity": 1 - distances[i]
                        } for i in range(len(ids))
                    ]
                }, f, indent=2)
            
            logger.info(f"Saved results to {output_file}")
        else:
            logger.warning("No results found")
            print("No results found for the query.")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
