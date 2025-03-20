"""
Simple script to query ChromaDB directly.

This script uses the custom Google Vertex AI embedding function to query the database.
"""

import os
import json
import logging
import argparse
import chromadb
from gitissueschat.embed.google_vertex_embedding_function import GoogleVertexEmbeddingFunctionCustom

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to query ChromaDB.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Query ChromaDB")
    parser.add_argument("--db-path", type=str, default="./chroma_db_sample",
                        help="Path to the ChromaDB database")
    parser.add_argument("--collection-name", type=str, default="fastai_issues_sample",
                        help="Name of the ChromaDB collection")
    parser.add_argument("--query", type=str, default="where is my config data stored?",
                        help="Query to test")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results to return")
    parser.add_argument("--output-file", type=str, default="query_results.json",
                        help="Path to output JSON file")
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
    
    logger.info(f"Querying ChromaDB with query: '{args.query}'")
    
    try:
        # Initialize the custom embedding function
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
            n_results=args.top_k
        )
        
        # Print results
        if not results["ids"] or len(results["ids"][0]) == 0:
            logger.warning("No results found")
            return
            
        logger.info(f"Found {len(results['ids'][0])} results:")
        
        # Prepare results for JSON output
        json_results = []
        
        for i in range(len(results["ids"][0])):
            # Get document and metadata
            document_id = results["ids"][0][i]
            document = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            # Convert distance to similarity score (1 - distance)
            # ChromaDB uses cosine distance, so similarity = 1 - distance
            similarity = 1 - distance
            
            # Add to JSON results
            json_results.append({
                "id": document_id,
                "similarity": similarity,
                "metadata": metadata,
                "document": document
            })
            
            # Log summary
            logger.info(f"Result {i+1}:")
            logger.info(f"  ID: {document_id}")
            logger.info(f"  Similarity: {similarity:.4f}")
            logger.info(f"  Metadata: {metadata}")
            logger.info(f"  Document: {document[:200]}...")
        
        # Write results to JSON file
        with open(args.output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Full results written to {args.output_file}")
    
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
