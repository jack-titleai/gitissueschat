"""
Test script for querying ChromaDB using FAISS with Google embeddings.

This script tests querying the ChromaDB collection using FAISS with Google embeddings.
"""

import os
import logging
import argparse
import json
import numpy as np
import faiss
import chromadb
from gitissueschat.embed.google_vertex_embedding_function import GoogleVertexEmbeddingFunctionCustom

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to test querying ChromaDB using FAISS.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test querying ChromaDB using FAISS")
    parser.add_argument("--db-path", type=str, default="./chroma_db_sample",
                        help="Path to the ChromaDB database")
    parser.add_argument("--collection-name", type=str, default="fastai_issues_sample",
                        help="Name of the ChromaDB collection")
    parser.add_argument("--query", type=str, default="How to use fastai with PyTorch?",
                        help="Query to test")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results to return")
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
    
    logger.info(f"Testing FAISS query with db_path={args.db_path}, collection_name={args.collection_name}")
    
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
        
        # Get all documents from the collection
        all_results = collection.get(include=["embeddings", "documents", "metadatas"])
        
        # Check if we have embeddings
        if "embeddings" not in all_results or not all_results["embeddings"]:
            logger.error("No embeddings found in the collection")
            logger.info(f"Available keys in results: {list(all_results.keys())}")
            logger.info(f"Sample of results: {json.dumps(all_results, indent=2)[:500]}...")
            return
        
        # Convert embeddings to numpy array
        embeddings = np.array(all_results["embeddings"])
        
        # Create FAISS index
        dimension = len(embeddings[0])
        logger.info(f"Embedding dimension: {dimension}")
        
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Generate embedding for query
        query_embedding = embedding_function([args.query])[0]
        
        # Normalize query embedding
        query_embedding_np = np.array([query_embedding])
        faiss.normalize_L2(query_embedding_np)
        
        # Search index
        distances, indices = index.search(query_embedding_np, args.top_k)
        
        # Print results
        logger.info(f"Query results: {len(indices[0])} matches")
        
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            # Convert distance to similarity (for inner product, higher is better)
            similarity = distance
            
            # Get document and metadata
            document = all_results["documents"][idx]
            metadata = all_results["metadatas"][idx]
            
            logger.info(f"Result {i+1}:")
            logger.info(f"  ID: {all_results['ids'][idx]}")
            logger.info(f"  Similarity: {similarity}")
            logger.info(f"  Metadata: {metadata}")
            logger.info(f"  Document: {document[:100]}...")
    
    except Exception as e:
        logger.error(f"Error querying with FAISS: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
