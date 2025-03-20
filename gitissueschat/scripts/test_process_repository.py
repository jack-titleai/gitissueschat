#!/usr/bin/env python3
"""
Test script for process_repository.py

This script tests the process_repository.py script with a small repository
to verify that it works correctly.
"""

import os
import argparse
import logging
from dotenv import load_dotenv

from gitissueschat.scripts.process_repository import download_issues, embed_database_to_chromadb, normalize_repo_input
from gitissueschat.utils.db_path_manager import get_sqlite_db_path, get_chroma_db_path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to test the process_repository.py script.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test the process_repository.py script")
    parser.add_argument("--repository", default="microsoft/vscode-extension-samples", 
                        help="GitHub repository to test with (URL or 'owner/repo' format, default: microsoft/vscode-extension-samples)")
    parser.add_argument("--max-issues", type=int, default=10, 
                        help="Maximum number of issues to download (default: 10)")
    parser.add_argument("--chunk-size", type=int, default=500, 
                        help="Size of chunks in tokens (default: 500)")
    parser.add_argument("--chunk-overlap", type=int, default=100, 
                        help="Overlap between chunks in tokens (default: 100)")
    
    args = parser.parse_args()
    
    try:
        # Normalize the repository input
        repo_name = normalize_repo_input(args.repository)
        logger.info(f"Using repository: {repo_name}")
        
        # Get GitHub token from environment variable
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            logger.error("GITHUB_TOKEN environment variable not set")
            exit(1)
        
        # Get Google Cloud credentials from environment
        project_id = os.environ.get("GOOGLE_PROJECT_ID")
        if not project_id:
            logger.error("GOOGLE_PROJECT_ID environment variable not set")
            exit(1)
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Get database paths
        sqlite_db_path = get_sqlite_db_path(repo_name)
        chroma_db_path = get_chroma_db_path(repo_name)
        
        logger.info(f"Testing with repository: {repo_name}")
        logger.info(f"SQLite database path: {sqlite_db_path}")
        logger.info(f"ChromaDB database path: {chroma_db_path}")
        
        # Step 1: Download issues
        download_issues(repo_name, github_token, sqlite_db_path)
        
        # Step 2: Chunk and embed issues
        logger.info(f"Processing issues into chunks and embedding them")
        stats = embed_database_to_chromadb(
            repo_name=repo_name,
            sqlite_db_path=sqlite_db_path,
            chroma_db_path=chroma_db_path,
            collection_name="github_issues",
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=100,
            project_id=project_id,
            api_key=api_key,
            credentials_path=credentials_path,
            limit_issues=args.max_issues
        )
        
        logger.info(f"Database statistics: {stats}")
        logger.info(f"Test completed successfully")
        
    except ValueError as e:
        logger.error(str(e))
        exit(1)

    # Test cases for normalize_repo_input
    test_cases = [
        "fastai/fastai",
        "https://github.com/fastai/fastai",
        "https://github.com/fastai/fastai.git",
        "fastai/fastai.git"
    ]

    # If a repository is provided, add it to the test cases
    if args.repository:
        test_cases.append(args.repository)

    # Test normalize_repo_input with each test case
    for test_case in test_cases:
        try:
            normalized_repo = normalize_repo_input(test_case)
            logger.info(f"Input: {test_case} -> Normalized: {normalized_repo}")
        except ValueError as e:
            logger.error(f"Error normalizing {test_case}: {e}")

if __name__ == "__main__":
    main()
