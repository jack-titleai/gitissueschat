#!/usr/bin/env python3
"""
Detailed test script for the RAG system.

This script tests the RAG system with a sample query and shows detailed information
about the retrieved documents, their similarity scores, how they are composed
to send to the LLM, and the final LLM output.
"""

import argparse
import logging
import os
import sys
import json
from typing import Dict, List, Any
from dotenv import load_dotenv

from gitissueschat.rag.rag_orchestrator import RAGOrchestrator
from gitissueschat.rag.gemini_generator import GeminiGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main function to test the RAG system with detailed output.
    """
    # Load environment variables from temp.env file
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp.env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        logger.info(f"Loaded environment variables from {dotenv_path}")
    else:
        logger.warning(f"Environment file {dotenv_path} not found")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the RAG system with detailed output")
    parser.add_argument("--db-path", type=str, default="./chroma_db_sample",
                        help="Path to the ChromaDB database")
    parser.add_argument("--collection-name", type=str, default="github_issues",
                        help="Name of the ChromaDB collection")
    parser.add_argument("--query", type=str, default="why am I getting a file not found error",
                        help="Query to test")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of chunks to retrieve")
    parser.add_argument("--relevance-threshold", type=float, default=0.5,
                        help="Minimum relevance score for chunks to be included")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-001",
                        help="Name of the Gemini model to use")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for generation")
    args = parser.parse_args()
    
    # Get project ID from environment
    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    if not project_id:
        logger.error("GOOGLE_PROJECT_ID environment variable not set")
        sys.exit(1)
    
    # Get credentials from environment
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        logger.error("Please set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of your service account key file")
        sys.exit(1)
    
    logger.info(f"Testing RAG system with query: {args.query}")
    logger.info(f"Using service account credentials from: {credentials_path}")
    logger.info(f"Project ID: {project_id}")
    
    try:
        # Initialize the RAG orchestrator
        orchestrator = RAGOrchestrator(
            db_path=args.db_path,
            collection_name=args.collection_name,
            project_id=project_id,
            api_key=None,  # We're using service account credentials instead
            credentials_path=credentials_path,
            top_k=args.top_k,
            relevance_threshold=args.relevance_threshold,
            model_name=args.model,
            temperature=args.temperature
        )
        
        # Step 1: Retrieve relevant chunks
        print("\n" + "="*80)
        print(f"QUERY: {args.query}")
        print("="*80)
        
        print("\nSTEP 1: RETRIEVING RELEVANT CHUNKS")
        print("-"*80)
        chunks = orchestrator.retriever.retrieve(args.query)
        
        print(f"Retrieved {len(chunks)} chunks with similarity >= {args.relevance_threshold}")
        
        # Print detailed information about each chunk
        for i, chunk in enumerate(chunks):
            print(f"\nCHUNK {i+1}:")
            print(f"  Similarity Score: {chunk['similarity']:.4f}")
            print(f"  ID: {chunk['id']}")
            print(f"  Metadata: {json.dumps(chunk['metadata'], indent=2)}")
            print(f"  Content Preview: {chunk['content'][:150]}...")
        
        # Step 2: Format context for the LLM
        print("\n" + "="*80)
        print("STEP 2: FORMATTING CONTEXT FOR THE LLM")
        print("-"*80)
        formatted_context = orchestrator.generator._format_context(chunks)
        print(formatted_context)
        
        # Step 3: Create the prompt
        print("\n" + "="*80)
        print("STEP 3: CREATING THE PROMPT")
        print("-"*80)
        prompt = orchestrator.generator._create_prompt(args.query, formatted_context)
        print(prompt)
        
        # Step 4: Generate the response
        print("\n" + "="*80)
        print("STEP 4: GENERATING THE RESPONSE")
        print("-"*80)
        response = orchestrator.generator.generate(args.query, chunks)
        print(response)
        
        print("\n" + "="*80)
        print("RAG PROCESS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
