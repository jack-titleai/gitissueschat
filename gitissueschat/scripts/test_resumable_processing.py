#!/usr/bin/env python3
"""
Test Resumable Processing Script

This script demonstrates how to use the resumable processing feature of the process_repository.py script.
It will process a repository in two steps:
1. Process the first 10 issues and then exit
2. Resume processing from where it left off

Usage:
    python test_resumable_processing.py owner/repo [--token TOKEN]
"""

import os
import argparse
import logging
import sys
import time
import signal
from dotenv import load_dotenv

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gitissueschat.utils.process_repository import (
    normalize_repo_input,
    download_issues,
    embed_database_to_chromadb,
    analyze_repository_data,
    get_sqlite_db_path,
    get_chroma_db_path
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to track the number of issues processed
issues_processed = 0
max_issues_first_run = 10  # Process only 10 issues in the first run

def signal_handler(sig, frame):
    """Handle interrupt signals to simulate an interruption."""
    logger.info(f"Process interrupted after processing {issues_processed} issues")
    sys.exit(0)

def process_batch_callback(batch_issues, batch_stats):
    """Callback function to track the number of issues processed and simulate an interruption."""
    global issues_processed
    issues_processed += len(batch_issues)
    logger.info(f"Processed {len(batch_issues)} issues, total: {issues_processed}")
    
    # Simulate an interruption after processing a certain number of issues
    if issues_processed >= max_issues_first_run:
        logger.info(f"Simulating interruption after processing {issues_processed} issues")
        os.kill(os.getpid(), signal.SIGINT)

def main():
    """Main function to test resumable processing."""
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Test resumable processing of a GitHub repository")
    parser.add_argument("repository", help="GitHub repository (URL or 'owner/repo' format)")
    parser.add_argument("--token", help="GitHub API token (or set GITHUB_TOKEN env var)")
    parser.add_argument("--resume", action="store_true", help="Resume processing from where it left off")
    args = parser.parse_args()
    
    try:
        # Normalize the repository input
        repo_name = normalize_repo_input(args.repository)
        logger.info(f"Using repository: {repo_name}")
        
        # Get GitHub token from args or environment variable
        github_token = args.token or os.environ.get("GITHUB_TOKEN")
        if not github_token:
            logger.error("GitHub token is required. Provide it with --token or set GITHUB_TOKEN environment variable.")
            sys.exit(1)
        
        # Get Google Cloud credentials from environment
        project_id = os.environ.get("GOOGLE_PROJECT_ID")
        if not project_id:
            logger.error("GOOGLE_PROJECT_ID environment variable not set")
            sys.exit(1)
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Get database paths
        sqlite_db_path = get_sqlite_db_path(repo_name)
        chroma_db_path = get_chroma_db_path(repo_name)
        
        logger.info(f"SQLite database path: {sqlite_db_path}")
        logger.info(f"ChromaDB database path: {chroma_db_path}")
        
        if args.resume:
            logger.info("Resuming processing from where it left off")
            # No need to set up the signal handler for the resume run
            download_issues(repo_name, github_token, sqlite_db_path, resume=True)
            
            # Process issues into chunks and embed them
            logger.info(f"Processing issues into chunks and embedding them")
            stats = embed_database_to_chromadb(
                repo_name=repo_name,
                sqlite_db_path=sqlite_db_path,
                chroma_db_path=chroma_db_path,
                collection_name="github_issues",
                chunk_size=500,
                chunk_overlap=100,
                batch_size=100,
                project_id=project_id,
                api_key=api_key,
                credentials_path=credentials_path,
                resume=True
            )
            
            logger.info(f"Database statistics: {stats}")
            logger.info(f"Repository processing complete: {repo_name}")
            
            # Analyze the repository data
            logger.info("Analyzing repository data...")
            stats = analyze_repository_data(sqlite_db_path, chroma_db_path, "github_issues")
            logger.info("Repository Analysis:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info("Starting initial processing (will be interrupted after 10 issues)")
            # Set up the signal handler to catch the interrupt
            signal.signal(signal.SIGINT, signal_handler)
            
            # Start downloading issues with a custom callback
            from gitissueschat.github_issues import GitHubIssuesFetcher
            from gitissueschat.sqlite_storage.sqlite_storage import SQLiteIssueStorage
            
            # Initialize the GitHub issues fetcher
            fetcher = GitHubIssuesFetcher(github_token)
            
            # Initialize the SQLite issue storage
            storage = SQLiteIssueStorage(sqlite_db_path)
            
            # Get existing issue numbers
            existing_issue_numbers = set(storage.get_issue_numbers(repo_name))
            
            # Fetch issues with batch processing and our custom callback
            fetcher.fetch_issues(
                repo_name=repo_name,
                state="all",
                include_comments=True,
                existing_issue_numbers=existing_issue_numbers,
                batch_size=5,  # Smaller batch size for demonstration
                batch_callback=process_batch_callback
            )
            
            # This code will not be reached due to the interrupt
            logger.info("This line should not be printed")
    
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        logger.info("Run the script again with --resume to continue from where it left off")
        sys.exit(0)

if __name__ == "__main__":
    main()
