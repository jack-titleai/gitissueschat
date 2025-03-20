"""
Command-line interface for the RAG system.

This module provides a command-line interface to interact with the RAG system.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Any

from gitissueschat.rag.rag_orchestrator import RAGOrchestrator
from gitissueschat.utils.db_path_manager import get_chroma_db_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> Dict[str, Any]:
    """
    Parse command-line arguments.
    
    Returns:
        Dictionary of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="GitHub Issues RAG System CLI")
    
    # Database arguments
    parser.add_argument("--db-path", type=str, help="Path to the ChromaDB database (overrides default path)")
    parser.add_argument("--collection-name", type=str, default="github_issues",
                        help="Name of the ChromaDB collection")
    
    # Repository argument
    parser.add_argument("--repository", type=str, required=True,
                        help="Repository name (e.g., 'username/repo')")
    
    # API arguments
    parser.add_argument("--api-key", type=str, default=None,
                        help="Google API key (if not provided, will try to get from environment)")
    
    # Retrieval arguments
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of chunks to retrieve")
    parser.add_argument("--relevance-threshold", type=float, default=0.5,
                        help="Minimum relevance score for chunks to be included")
    
    # Generation arguments
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-001",
                        help="Name of the Gemini model to use")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for generation")
    
    # Filter arguments
    parser.add_argument("--issue-number", type=int, default=None,
                        help="Filter by issue number")
    
    # Query argument
    parser.add_argument("--query", type=str, default=None,
                        help="Query to process (if not provided, will enter interactive mode)")
    
    # Parse arguments
    args = parser.parse_args()
    return vars(args)


def get_filter_criteria(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get filter criteria from arguments.
    
    Args:
        args: Parsed command-line arguments.
        
    Returns:
        Filter criteria dictionary.
    """
    filter_criteria = {}
    
    # Add repository filter if provided
    if args["repository"]:
        filter_criteria["repository"] = args["repository"]
    
    # Add issue number filter if provided
    if args["issue_number"]:
        filter_criteria["issue_number"] = args["issue_number"]
    
    return filter_criteria if filter_criteria else None


def process_single_query(orchestrator: RAGOrchestrator, query: str, filter_criteria: Dict[str, Any]) -> None:
    """
    Process a single query and print the response.
    
    Args:
        orchestrator: RAG orchestrator instance.
        query: Query to process.
        filter_criteria: Filter criteria for retrieval.
    """
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    
    # Process the query
    response = orchestrator.process_query(query)
    
    # Get the number of chunks from the retriever
    num_chunks = len(orchestrator.retriever.retrieve(query))
    
    # Print the response
    print(f"RESPONSE: (based on {num_chunks} chunks)")
    print("-" * 80)
    print(response)
    print("=" * 80)


def process_query(orchestrator: RAGOrchestrator, query: str) -> None:
    """
    Process a single query and print the response.
    
    Args:
        orchestrator: RAG orchestrator instance.
        query: Query to process.
    """
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    
    # Process the query
    response = orchestrator.process_query(query)
    
    # Print the response
    print(f"RESPONSE:")
    print("-" * 80)
    print(response)
    print("=" * 80)


def interactive_mode(orchestrator: RAGOrchestrator, filter_criteria: Dict[str, Any]) -> None:
    """
    Run the CLI in interactive mode.
    
    Args:
        orchestrator: RAG orchestrator.
        filter_criteria: Filter criteria for retrieval.
    """
    print("Interactive mode. Type 'exit' or 'quit' to exit.")
    
    while True:
        # Get user input
        query = input("\nEnter your query: ")
        
        # Check if the user wants to exit
        if query.lower() in ["exit", "quit"]:
            break
        
        # Process the query
        process_query(orchestrator, query)


def main() -> None:
    """
    Main entry point for the CLI.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.info("GOOGLE_API_KEY environment variable not set, will try to use service account credentials")
    
    # Get project ID from environment
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_PROJECT_ID")
    if not project_id:
        logger.error("GOOGLE_PROJECT_ID environment variable not set")
        return
    
    # Get credentials from environment
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Get the ChromaDB path
    db_path = args["db_path"] or get_chroma_db_path(args["repository"])
    
    # Create filter criteria
    filter_criteria = get_filter_criteria(args)
    
    # Initialize the RAG system
    try:
        # Initialize the orchestrator
        orchestrator = RAGOrchestrator(
            db_path=db_path,
            collection_name=args["collection_name"],
            project_id=project_id,
            api_key=api_key,
            credentials_path=credentials,
            top_k=args["top_k"],
            relevance_threshold=args["relevance_threshold"],
            model_name=args["model"],
            temperature=args["temperature"]
        )
        
        # Process query or run interactive mode
        if args["query"]:
            # Process a single query
            process_query(orchestrator, args["query"])
        else:
            # Run interactive mode
            interactive_mode(orchestrator, filter_criteria)
            
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
