"""
Test script for embed_database_to_chromadb.py

This script tests the embed_database_to_chromadb.py script on fastai-issues-sample.db.
"""

import argparse
import logging
import os
import time
from typing import Dict, Any, Optional

from gitissueschat.scripts.embed_database_to_chromadb import embed_database_to_chromadb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_embed_database(
    db_path: str,
    chroma_db_path: str,
    collection_name: str,
    chunk_size: int = 250,
    chunk_overlap: int = 50,
    batch_size: int = 20,
    project_id: Optional[str] = None,
    api_key: Optional[str] = None,
    credentials: Optional[str] = None
) -> Dict[str, Any]:
    """
    Test the embed_database_to_chromadb function on a sample database.
    
    Args:
        db_path: Path to the SQLite database.
        chroma_db_path: Path to the ChromaDB database.
        collection_name: Name of the collection to use.
        chunk_size: Size of chunks in tokens.
        chunk_overlap: Overlap between chunks in tokens.
        batch_size: Number of chunks to process at once.
        project_id: Google Cloud project ID.
        api_key: Google API key.
        credentials: Path to the Google Cloud service account key file.
        
    Returns:
        Statistics about the database.
    """
    logger.info(f"Testing embed_database_to_chromadb on {db_path}")
    logger.info(f"Using chunk size: {chunk_size}, chunk overlap: {chunk_overlap}, batch size: {batch_size}")
    
    # Start timer
    start_time = time.time()
    
    # Process the database
    stats = embed_database_to_chromadb(
        db_path=db_path,
        chroma_db_path=chroma_db_path,
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
        project_id=project_id,
        api_key=api_key,
        credentials=credentials
    )
    
    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print statistics
    logger.info(f"Embedding completed in {elapsed_time:.2f} seconds")
    logger.info(f"Database statistics: {stats}")
    
    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test embed_database_to_chromadb.py on fastai-issues-sample.db")
    parser.add_argument("--db-path", default="./fastai-issues-sample.db", help="Path to the SQLite database")
    parser.add_argument("--chroma-db-path", default="./chroma_db_sample", help="Path to the ChromaDB database")
    parser.add_argument("--collection-name", default="fastai_issues_sample", help="Name of the collection to use")
    parser.add_argument("--chunk-size", type=int, default=250, help="Size of chunks in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between chunks in tokens")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of chunks to process at once")
    parser.add_argument("--project-id", help="Google Cloud project ID")
    parser.add_argument("--api-key", help="Google API key")
    parser.add_argument("--credentials", help="Path to the Google Cloud service account key file")
    
    args = parser.parse_args()
    
    # Get Google Cloud project ID from environment variable if not provided
    project_id = args.project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
    
    # Get Google API key from environment variable if not provided
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    
    # Get Google Cloud credentials from environment variable if not provided
    credentials = args.credentials or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Test the function
    test_embed_database(
        db_path=args.db_path,
        chroma_db_path=args.chroma_db_path,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        project_id=project_id,
        api_key=api_key,
        credentials=credentials
    )


if __name__ == "__main__":
    main()
