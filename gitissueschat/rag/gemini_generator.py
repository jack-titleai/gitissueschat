"""
Gemini Flash 2.0 Generator for RAG System.

This module provides a generator that uses Gemini Flash 2.0 to generate responses.
"""

import logging
import os
import time
from typing import Dict, List, Any, Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeminiGenerator:
    """
    A generator that uses Gemini Flash 2.0 to generate responses.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-001",
        temperature: float = 0.2,
        max_output_tokens: int = 2048
    ):
        """
        Initialize the Gemini generator.
        
        Args:
            api_key: Google API key. If None, will try to get from environment.
            model_name: Name of the Gemini model to use.
            temperature: Temperature for generation.
            max_output_tokens: Maximum number of tokens to generate.
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key is None:
                logger.warning("API key not provided and not found in environment, will try to use service account credentials")
        
        # Configure the Gemini API
        if api_key:
            logger.info("Configuring Gemini API with API key")
            genai.configure(api_key=api_key)
        else:
            # Try to use service account credentials
            credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if credentials_path:
                logger.info(f"Configuring Gemini API with service account credentials from {credentials_path}")
                import google.auth
                from google.auth import credentials as auth_credentials
                from google.auth.transport import requests
                
                # Load credentials from the service account file
                try:
                    credentials, project_id = google.auth.load_credentials_from_file(credentials_path)
                    genai.configure(credentials=credentials)
                    logger.info(f"Successfully configured Gemini API with service account for project {project_id}")
                except Exception as e:
                    logger.error(f"Error loading service account credentials: {e}")
                    raise ValueError("Failed to load service account credentials")
            else:
                raise ValueError("Neither API key nor service account credentials provided")
        
        # Set model parameters
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "top_p": 0.95,
                "top_k": 40
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        logger.info(f"Initialized GeminiGenerator with model={model_name}, temperature={temperature}")
    
    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response for a query using retrieved context chunks.
        
        Args:
            query: User query.
            context_chunks: List of context chunks with document text and metadata.
            
        Returns:
            Dictionary containing the generated response and timing information.
        """
        logger.info(f"Generating response for query: {query}")
        
        # Check if we have any context chunks
        if not context_chunks:
            logger.warning("No context chunks provided for generation")
            return {
                "response": "I couldn't find any relevant information to answer your question. Please try a different query or check if the database contains the information you're looking for.",
                "api_call_time": 0.0,
                "processing_time": 0.0
            }
        
        # Start timing for processing
        processing_start = time.time()
        
        # Format context for the prompt
        formatted_context = self._format_context(context_chunks)
        
        # Create the prompt
        prompt = self._create_prompt(query, formatted_context)
        
        # Calculate preprocessing time
        preprocessing_time = time.time() - processing_start
        
        # Generate the response
        try:
            # Start timing for API call
            api_call_start = time.time()
            
            # Make the API call
            response = self.model.generate_content(prompt)
            
            # Calculate API call time
            api_call_time = time.time() - api_call_start
            
            # Start timing for post-processing
            postprocessing_start = time.time()
            
            # Extract text from response
            response_text = response.text
            logger.info(f"Generated response of length {len(response_text)}")
            
            # Calculate post-processing time
            postprocessing_time = time.time() - postprocessing_start
            
            # Calculate total processing time (pre + post, excluding API call)
            processing_time = preprocessing_time + postprocessing_time
            
            return {
                "response": response_text,
                "api_call_time": api_call_time,
                "processing_time": processing_time
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": f"Error generating response: {str(e)}",
                "api_call_time": 0.0,
                "processing_time": time.time() - processing_start
            }
    
    def _format_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format context chunks for the prompt.
        
        Args:
            context_chunks: List of context chunks.
            
        Returns:
            Formatted context string.
        """
        if not context_chunks:
            return ""
        
        # Sort chunks by similarity (highest first)
        sorted_chunks = sorted(context_chunks, key=lambda x: x.get("similarity", 0.0), reverse=True)
        
        context_parts = []
        for i, chunk in enumerate(sorted_chunks):
            # Extract content and metadata
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            similarity = chunk.get("similarity", 0.0)
            
            # Extract key metadata fields
            repo = metadata.get("repository", "Unknown repository")
            issue_number = metadata.get("issue_number", "Unknown")
            issue_title = metadata.get("issue_title", "Unknown title")
            chunk_type = metadata.get("type", "Unknown type")
            created_at = metadata.get("created_at", "Unknown date")
            updated_at = metadata.get("updated_at", "Unknown date")
            
            # Format header based on chunk type
            if chunk_type == "issue":
                header = f"ISSUE #{issue_number} in {repo}: \"{issue_title}\""
            elif chunk_type == "comment":
                author = metadata.get("comment_author", "Unknown author")
                header = f"COMMENT on ISSUE #{issue_number} in {repo} by {author}"
            else:
                header = f"CONTENT from {repo} related to issue #{issue_number}"
            
            # Add to context parts with clear formatting
            context_parts.append(
                f"CHUNK {i+1} [similarity: {similarity:.4f}]:\n"
                f"{header}\n"
                f"Created: {created_at} | Last updated: {updated_at}\n"
                f"CONTENT:\n{content}\n"
            )
        
        return "\n" + "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the Gemini model.
        
        Args:
            query: User query.
            context: Formatted context string.
            
        Returns:
            Complete prompt string.
        """
        return f"""You are an AI assistant specialized in answering questions about GitHub issues. 
You will be given a user query and relevant context from GitHub issues and comments.
Use the provided context to answer the user's question accurately and concisely.
Pay attention to the timestamps in the context to provide the most up-to-date information.
When referencing comments, note that they include context from the original issue to help you understand the discussion.

If the context doesn't contain enough information to answer the question, say so clearly.
Do not make up information that is not in the context.

CONTEXT:
{context}

USER QUERY: {query}

RESPONSE:"""
