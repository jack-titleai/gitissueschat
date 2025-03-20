"""
Test script for the RAG system.

This script tests the RAG system with a sample query.
"""

import argparse
import logging
import os
import sys

from gitissueschat.rag.rag_orchestrator import RAGOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main function to test the RAG system.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the RAG system")
    parser.add_argument("--db-path", type=str, default="./chroma_db_sample",
                        help="Path to the ChromaDB database")
    parser.add_argument("--collection-name", type=str, default="fastai_issues_sample",
                        help="Name of the ChromaDB collection")
    parser.add_argument("--query", type=str, default="How to use fastai with PyTorch?",
                        help="Query to test")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Google API key (if not provided, will try to get from environment)")
    args = parser.parse_args()
    
    # Get API key from arguments or environment
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("API key not provided and GOOGLE_API_KEY environment variable not set")
        logger.error("Please provide an API key with --api-key or set the GOOGLE_API_KEY environment variable")
        sys.exit(1)
    
    # Get project ID from environment
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")
    
    # Get credentials from environment
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    logger.info(f"Testing RAG system with query: {args.query}")
    
    # Initialize the RAG orchestrator
    orchestrator = RAGOrchestrator(
        db_path=args.db_path,
        collection_name=args.collection_name,
        project_id=project_id,
        api_key=api_key,
        credentials=credentials,
        top_k=10,
        relevance_threshold=0.75
    )
    
    # Process the query
    result = orchestrator.process_query(args.query)
    
    # Print the results
    print("\n" + "="*80)
    print(f"QUERY: {args.query}")
    print("="*80)
    print(f"RESPONSE: (based on {result['num_chunks']} chunks)")
    print("-"*80)
    print(result["response"])
    print("="*80)
    
    # Print the chunks used
    print("\nCHUNKS USED:")
    for i, chunk in enumerate(result["chunks"]):
        print(f"\nChunk {i+1} (Similarity: {chunk['similarity']:.4f}):")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Text: {chunk['document'][:100]}...")
    
    logger.info("Test completed")


if __name__ == "__main__":
    main()
