"""
Debug script for querying ChromaDB directly.

This script tests querying the ChromaDB collection directly and prints the structure of the results.
"""

import os
import logging
import argparse
import json
import chromadb
from gitissueschat.embed.google_vertex_embedding_function import GoogleVertexEmbeddingFunctionCustom

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to debug ChromaDB query results.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Debug ChromaDB query results")
    parser.add_argument("--db-path", type=str, default="./chroma_db_sample",
                        help="Path to the ChromaDB database")
    parser.add_argument("--collection-name", type=str, default="fastai_issues_sample",
                        help="Name of the ChromaDB collection")
    parser.add_argument("--query", type=str, default="How to use fastai with PyTorch?",
                        help="Query to test")
    args = parser.parse_args()
    
    # Get project ID from environment
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")
    if not project_id:
        logger.error("GOOGLE_PROJECT_ID environment variable not set")
        return
    
    # Get credentials from environment
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        return
    
    logger.info(f"Testing ChromaDB query with db_path={args.db_path}, collection_name={args.collection_name}")
    
    try:
        # Initialize the embedding function
        embedding_function = GoogleVertexEmbeddingFunctionCustom(
            project_id=project_id,
            credentials_path=credentials
        )
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=args.db_path)
        
        # Get collection
        collection = client.get_collection(
            name=args.collection_name,
            embedding_function=embedding_function
        )
        
        logger.info(f"Collection count: {collection.count()}")
        
        # Query the collection
        results = collection.query(
            query_texts=[args.query],
            n_results=5
        )
        
        # Print the structure of the results
        logger.info("Results structure:")
        for key, value in results.items():
            logger.info(f"  {key}: {type(value)} of length {len(value)}")
            if isinstance(value, list) and len(value) > 0:
                logger.info(f"    First element type: {type(value[0])}")
                if isinstance(value[0], (list, dict)):
                    logger.info(f"    First element: {value[0]}")
                else:
                    logger.info(f"    First element: {value[0]}")
        
        # Print the full results as JSON
        try:
            logger.info("Full results:")
            logger.info(json.dumps(results, indent=2))
        except Exception as e:
            logger.error(f"Error serializing results to JSON: {e}")
            # Try to print the results directly
            logger.info(f"Raw results: {results}")
    
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")

if __name__ == "__main__":
    main()
