"""
Database module for storing and retrieving chunks using ChromaDB.

This module provides functionality for storing and retrieving chunks using ChromaDB
with various embedding functions.
"""

import os
import json
import uuid
import logging
from typing import Dict, List, Any, Optional

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

from gitissueschat.embed.google_vertex_embedding_function import GoogleVertexEmbeddingFunctionCustom

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChunksDatabase:
    """
    Database class for storing and retrieving chunks using ChromaDB.
    """
    
    def __init__(
        self, 
        db_path: str, 
        collection_name: str = "github_issues",
        project_id: Optional[str] = None,
        api_key: Optional[str] = None,
        credentials_path: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize the database.
        
        Args:
            db_path: Path to the ChromaDB database.
            collection_name: Name of the collection to use.
            project_id: Google Cloud project ID. If None, will try to use environment variable.
            api_key: Google API key. If None, will try to use environment variable.
            credentials_path: Optional path to service account credentials file. If None, will try to use environment variable.
            embedding_function: Optional custom embedding function. If None, will use GoogleVertexEmbeddingFunctionCustom.
        """
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Determine embedding function to use
        if embedding_function:
            logger.info("Using provided embedding function")
            self.embedding_function = embedding_function
        else:
            # Get project ID from environment if not provided
            project_id = project_id or os.environ.get("GOOGLE_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            
            # Get credentials path from environment if not provided
            credentials_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            
            # Always use GoogleVertexEmbeddingFunctionCustom if project_id is available
            if project_id:
                logger.info(f"Using GoogleVertexEmbeddingFunctionCustom with project_id: {project_id}")
                self.embedding_function = GoogleVertexEmbeddingFunctionCustom(
                    project_id=project_id,
                    credentials_path=credentials_path
                )
            else:
                logger.warning("No project_id available. Falling back to default embedding function.")
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB at {db_path} with collection {collection_name}")
    
    def get_collection(self):
        """
        Get the current collection.
        
        Returns:
            The ChromaDB collection object.
        """
        return self.collection
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to the database.
        
        Args:
            chunks: List of chunks to add.
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            # Generate a unique ID for the chunk
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            
            # Add the document text (using the "text" field from the chunker)
            # This field is expected to contain the text content of the chunk
            documents.append(chunk["text"])
            
            # Prepare metadata
            metadata = {
                "id": chunk["id"],
                "type": chunk["type"],
                "repository": chunk["repository"],
                "issue_number": chunk["issue_number"],
                "issue_id": chunk["issue_id"],
            }
            
            # Add created_at and updated_at dates from metadata if available
            if "metadata" in chunk and "created_at" in chunk["metadata"]:
                metadata["created_at"] = chunk["metadata"]["created_at"]
            elif "metadata" in chunk and "issue_created_at" in chunk["metadata"]:
                metadata["created_at"] = chunk["metadata"]["issue_created_at"]
                
            if "metadata" in chunk and "updated_at" in chunk["metadata"]:
                metadata["updated_at"] = chunk["metadata"]["updated_at"]
            elif "metadata" in chunk and "issue_updated_at" in chunk["metadata"]:
                metadata["updated_at"] = chunk["metadata"]["issue_updated_at"]
            
            # Add issue-specific metadata
            if chunk["type"] == "issue" and "metadata" in chunk:
                if "title" in chunk["metadata"]:
                    metadata["issue_title"] = chunk["metadata"]["title"]
                if "url" in chunk["metadata"]:
                    metadata["issue_url"] = chunk["metadata"]["url"]
            
            # Add comment-specific metadata
            if chunk["type"] == "comment" and "metadata" in chunk:
                if "issue_title" in chunk["metadata"]:
                    metadata["issue_title"] = chunk["metadata"]["issue_title"]
                if "issue_url" in chunk["metadata"]:
                    metadata["issue_url"] = chunk["metadata"]["issue_url"]
                if "author" in chunk["metadata"]:
                    metadata["comment_author"] = chunk["metadata"]["author"]
                if "url" in chunk["metadata"]:
                    metadata["comment_url"] = chunk["metadata"]["url"]
            
            metadatas.append(metadata)
        
        # Add data to ChromaDB
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"Error adding chunks to ChromaDB: {e}")
            raise
    
    def query(
        self, 
        query_text: str, 
        n_results: int = 5, 
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the database for similar chunks.
        
        Args:
            query_text: Query text.
            n_results: Number of results to return.
            filter_criteria: Optional filter criteria.
            
        Returns:
            Query results.
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=filter_criteria
            )
            
            return results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            # Return empty results structure
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
                "distances": []
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with statistics.
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection.name,
            "count": count
        }
