"""
ChromaDB Retriever for RAG System.

This module provides a retriever that integrates with ChromaDB to retrieve relevant chunks.
"""

import logging
from typing import Dict, List, Any, Optional

from gitissueschat.embed.chroma_database import ChunksDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChromaRetriever:
    """
    A retriever that integrates with ChromaDB to retrieve relevant chunks.
    """
    
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        project_id: Optional[str] = None,
        api_key: Optional[str] = None,
        credentials_path: Optional[str] = None,
        top_k: int = 10,
        relevance_threshold: float = 0.75
    ):
        """
        Initialize the ChromaDB retriever.
        
        Args:
            db_path: Path to the ChromaDB database.
            collection_name: Name of the collection to use.
            project_id: Google Cloud project ID.
            api_key: Google API key.
            credentials_path: Path to the Google Cloud service account key file.
            top_k: Number of chunks to retrieve.
            relevance_threshold: Minimum relevance score for chunks to be included.
        """
        self.db = ChunksDatabase(
            db_path=db_path,
            collection_name=collection_name,
            project_id=project_id,
            api_key=api_key,
            credentials_path=credentials_path
        )
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        logger.info(f"Initialized ChromaRetriever with top_k={top_k} and relevance_threshold={relevance_threshold}")
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: Query text.
            
        Returns:
            List of relevant chunks.
        """
        logger.info(f"Retrieving chunks for query: {query}")
        
        try:
            # Query the database
            results = self.db.query(query, self.top_k)
            
            # Log the structure of the results for debugging
            logger.info("Results structure:")
            for key in results:
                if isinstance(results[key], list):
                    if len(results[key]) > 0 and isinstance(results[key][0], list):
                        # Handle nested lists (ChromaDB returns nested lists)
                        logger.info(f"  {key}: {type(results[key])} of length {len(results[key])} with inner length {len(results[key][0])}")
                    else:
                        logger.info(f"  {key}: {type(results[key])} of length {len(results[key])}")
                else:
                    logger.info(f"  {key}: {type(results[key])}")
            
            # Extract the relevant chunks
            chunks = []
            
            # Check if we have results
            if "ids" in results and results["ids"] and len(results["ids"]) > 0:
                # ChromaDB returns nested lists for query results
                ids = results["ids"][0] if isinstance(results["ids"][0], list) else results["ids"]
                documents = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
                metadatas = results["metadatas"][0] if isinstance(results["metadatas"][0], list) else results["metadatas"]
                distances = results["distances"][0] if isinstance(results["distances"][0], list) else results["distances"]
                
                for i in range(len(ids)):
                    # Convert distance to similarity score (1 - distance)
                    # ChromaDB uses cosine distance, so similarity = 1 - distance
                    similarity = 1 - distances[i]
                    
                    # Only include chunks with similarity above the threshold
                    if similarity >= self.relevance_threshold:
                        chunk = {
                            "id": ids[i],
                            "content": documents[i],
                            "metadata": metadatas[i],
                            "similarity": similarity
                        }
                        chunks.append(chunk)
            
            if not chunks:
                logger.warning("No results found in the database")
            else:
                logger.info(f"Retrieved {len(chunks)} chunks with similarity >= {self.relevance_threshold}")
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
