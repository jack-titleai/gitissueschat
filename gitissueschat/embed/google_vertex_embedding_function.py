"""
Google Vertex AI embedding function for GitHub issues and comments.

This module provides a custom embedding function for Google's text-embedding-005 model
that is compatible with ChromaDB's embedding function interface.
"""

import logging
from typing import List, Optional
import re

import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.oauth2 import service_account
from google.api_core import exceptions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GoogleVertexEmbeddingFunctionCustom:
    """
    Custom embedding function for Google's text-embedding-005 model using service account credentials.
    This class is designed to be compatible with ChromaDB's embedding function interface.
    """
    
    def __init__(
        self, 
        project_id: str,
        location: str = "us-central1",
        model_name: str = "text-embedding-005",
        credentials_path: Optional[str] = None
    ):
        """
        Initialize the Google Vertex AI embedding function.
        
        Args:
            project_id: Google Cloud project ID.
            location: Google Cloud location.
            model_name: Model name to use for embeddings.
            credentials_path: Path to service account credentials file.
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        
        # Initialize Vertex AI with service account credentials if provided
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            vertexai.init(project=project_id, location=location, credentials=credentials)
        else:
            # Use Application Default Credentials
            vertexai.init(project=project_id, location=location)
        
        # Load the embedding model
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        logger.info(f"Initialized Google Vertex AI with model {model_name}")
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            input: List of texts to embed.
            
        Returns:
            List of embeddings.
        """
        embeddings = []
        
        # Process in batches to avoid API limits
        batch_size = 35  # Adjust based on API limits
        for i in range(0, len(input), batch_size):
            batch_texts = input[i:i+batch_size]
            
            # Get embeddings for the batch using a single API call
            try:
                embedding_results = self.model.get_embeddings(batch_texts)
                batch_embeddings = [result.values for result in embedding_results]
                embeddings.extend(batch_embeddings)
            except exceptions.InvalidArgument as e:
                # Check if the error is related to token count exceeding the limit
                if "Unable to submit request because the input token count" in str(e):
                    logger.warning(f"Token count exceeded limit, splitting batch: {str(e)}")
                    sub_batch = [batch_texts[:batch_size//2], batch_texts[batch_size//2:]]
                    for sub_batch_texts in sub_batch:
                        sub_embedding_results = self.model.get_embeddings(sub_batch_texts)
                        sub_batch_embeddings = [result.values for result in sub_embedding_results]
                        embeddings.extend(sub_batch_embeddings)
                    print("Successfully embeded sub-batches")
                else:
                    # For other InvalidArgument errors
                    raise ValueError(f"API error: {str(e)}")

        return embeddings
