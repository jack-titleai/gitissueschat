#!/usr/bin/env python
"""
Embed chunks using Google's text-embeddings-005 model and store them in ChromaDB.

This script takes chunks from a JSONL file, embeds them using Google's text-embeddings-005 model,
and stores them in ChromaDB for later retrieval.
"""

import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from gitissueschat.scripts.chunk_database_processor import process_chunks_to_db

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main function to embed chunks and store them in ChromaDB.
    """
    parser = argparse.ArgumentParser(description="Embed chunks and store them in ChromaDB")
    parser.add_argument("--chunks-file", required=True, help="Path to the JSONL file with chunks")
    parser.add_argument("--db-path", required=True, help="Path to the ChromaDB database")
    parser.add_argument("--collection-name", default="github_issues", help="Name of the collection to use")
    parser.add_argument("--google-project-id", help="Google Cloud project ID")
    parser.add_argument("--google-api-key", help="Google API key")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of chunks to process at once")
    
    args = parser.parse_args()
    
    # Check for authentication methods
    project_id = args.google_project_id or os.environ.get("GOOGLE_PROJECT_ID")
    api_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY")
    
    if not project_id and not api_key:
        logger.warning("Neither Google Project ID nor API key is provided. "
                      "Will use default embedding function.")
    
    # Create the database directory if it doesn't exist
    db_dir = Path(args.db_path).parent
    os.makedirs(db_dir, exist_ok=True)
    
    # Process chunks and store them in ChromaDB
    stats = process_chunks_to_db(
        chunks_file=args.chunks_file,
        db_path=args.db_path,
        collection_name=args.collection_name,
        project_id=project_id,
        api_key=api_key,
        credentials=None,
        batch_size=args.batch_size
    )
    
    logger.info(f"Successfully embedded chunks and stored them in ChromaDB at {args.db_path}")
    logger.info(f"Database statistics: {stats}")


if __name__ == "__main__":
    main()
