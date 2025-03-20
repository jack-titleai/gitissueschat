"""
Test script for the embedding function.

This script tests the embedding function directly to ensure it works correctly.
"""

import os
import logging
from gitissueschat.embed.google_vertex_embedding_function import GoogleVertexEmbeddingFunctionCustom

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to test the embedding function.
    """
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
    
    logger.info(f"Testing embedding function with project_id={project_id} and credentials={credentials}")
    
    # Initialize the embedding function
    embedding_function = GoogleVertexEmbeddingFunctionCustom(
        project_id=project_id,
        credentials_path=credentials
    )
    
    # Test texts
    test_texts = [
        "How to use fastai with PyTorch?",
        "What is the best way to train a model?",
        "How to handle errors in Python?"
    ]
    
    # Generate embeddings
    try:
        embeddings = embedding_function(test_texts)
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        
        # Print embedding dimensions
        for i, embedding in enumerate(embeddings):
            logger.info(f"Embedding {i+1}: dimension={len(embedding)}")
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")

if __name__ == "__main__":
    main()
