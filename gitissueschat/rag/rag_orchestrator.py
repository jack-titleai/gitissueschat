"""
RAG Orchestrator for GitHub Issues.

This module provides the main orchestrator for the RAG system, integrating
the retriever and generator components.
"""

import logging
from typing import Dict, List, Any, Optional

from gitissueschat.rag.chroma_retriever import ChromaRetriever
from gitissueschat.rag.gemini_generator import GeminiGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Main orchestrator for the RAG system, integrating the retriever and generator components.
    """
    
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        project_id: Optional[str] = None,
        api_key: Optional[str] = None,
        credentials_path: Optional[str] = None,
        top_k: int = 10,
        relevance_threshold: float = 0.75,
        model_name: str = "gemini-2.0-flash-001",
        temperature: float = 0.2
    ):
        """
        Initialize the RAG orchestrator.
        
        Args:
            db_path: Path to the ChromaDB database.
            collection_name: Name of the collection to use.
            project_id: Google Cloud project ID.
            api_key: Google API key.
            credentials_path: Path to the Google Cloud service account key file.
            top_k: Number of chunks to retrieve.
            relevance_threshold: Minimum relevance score for chunks to be included.
            model_name: Name of the Gemini model to use.
            temperature: Temperature for generation.
        """
        # Initialize the retriever
        self.retriever = ChromaRetriever(
            db_path=db_path,
            collection_name=collection_name,
            project_id=project_id,
            api_key=api_key,
            credentials_path=credentials_path,
            top_k=top_k,
            relevance_threshold=relevance_threshold
        )
        
        # Initialize the generator
        self.generator = GeminiGenerator(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature
        )
        
        logger.info(f"Initialized RAGOrchestrator with top_k={top_k}, relevance_threshold={relevance_threshold}")
    
    def process_query(self, query: str) -> str:
        """
        Process a query and generate a response.
        
        Args:
            query: Query text.
            
        Returns:
            Response text.
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(query)
        
        # Generate response
        result = self.generator.generate(query, chunks)
        
        # Extract response text from result
        if isinstance(result, dict) and "response" in result:
            return result["response"]
        
        # For backward compatibility
        return result
