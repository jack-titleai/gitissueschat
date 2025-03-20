#!/usr/bin/env python
"""
Test querying the ChromaDB database with a custom embedding function.
"""

import os
import logging
import argparse
from dotenv import load_dotenv
from gitissueschat.embed.chroma_database import ChunksDatabase

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Test querying the ChromaDB database.
    """
    parser = argparse.ArgumentParser(description="Test querying the ChromaDB database")
    parser.add_argument("--db-path", default="./chroma_db", help="Path to the ChromaDB database")
    parser.add_argument("--collection-name", default="github_issues", help="Name of the collection to use")
    parser.add_argument("--query", default="How to use PyTorch with fastai?", help="Query text")
    parser.add_argument("--n-results", type=int, default=3, help="Number of results to return")
    
    args = parser.parse_args()
    
    # Initialize database
    db = ChunksDatabase(
        db_path=args.db_path, 
        collection_name=args.collection_name,
        project_id=os.environ.get("GOOGLE_PROJECT_ID"),
        credentials=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
    
    # Query the database
    results = db.query(args.query, n_results=args.n_results)
    
    # Print results
    logger.info(f"Query: {args.query}")
    logger.info(f"Found {len(results['documents'][0])} results")
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )):
        logger.info(f"Result {i+1}:")
        logger.info(f"Distance: {distance}")
        logger.info(f"Metadata: {metadata}")
        logger.info(f"Document: {doc[:200]}...")
        logger.info("-" * 80)


if __name__ == "__main__":
    main()
