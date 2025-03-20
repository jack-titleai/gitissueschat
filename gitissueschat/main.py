"""
Main script for GitHub Issues RAG

This script provides a command-line interface to fetch GitHub issues
and store them in a SQLite database.
"""

import os
import argparse
import time
from typing import Optional
from dotenv import load_dotenv

from gitissueschat.github_issues import GitHubIssuesFetcher
from gitissueschat.sqlite_storage.sqlite_storage import SQLiteIssueStorage
from gitissueschat.utils.db_path_manager import get_sqlite_db_path


def main():
    """
    Main function to fetch GitHub issues and store them in SQLite.
    """
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fetch GitHub issues and store them in SQLite")
    parser.add_argument("repo", help="GitHub repository name (e.g., 'username/repo')")
    parser.add_argument("--token", help="GitHub personal access token")
    parser.add_argument(
        "--state",
        choices=["open", "closed", "all"],
        default="all",
        help="State of issues to fetch (default: all)",
    )
    parser.add_argument("--max-issues", type=int, help="Maximum number of issues to fetch")
    parser.add_argument(
        "--db-path", help="Path to SQLite database file (overrides default path)"
    )
    parser.add_argument(
        "--include-comments",
        action="store_true",
        default=True,
        help="Include comments in fetched issues",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        default=False,
        help="Force refresh all issues, ignoring last update timestamp",
    )
    parser.add_argument(
        "--update-mode",
        action="store_true",
        default=False,
        help="Use update mode to only fetch issues updated since last API call",
    )

    args = parser.parse_args()

    # Get GitHub token from arguments or environment variable
    github_token = args.token or os.getenv("GITHUB_TOKEN")

    # Initialize GitHub issue fetcher
    fetcher = GitHubIssuesFetcher(token=github_token)

    # Get the SQLite database path
    db_path = args.db_path or get_sqlite_db_path(args.repo)
    
    # Initialize SQLite storage
    storage = SQLiteIssueStorage(db_path=db_path)

    # Get the latest API call timestamp and existing issue numbers
    last_api_call = None

    # Only use timestamps in update mode
    if args.update_mode and not args.force_refresh:
        last_api_call = storage.get_latest_api_call_timestamp(args.repo)

        # If no API call log exists, fall back to the latest issue update timestamp
        if not last_api_call:
            last_api_call = storage.get_latest_update_timestamp(args.repo)

    existing_issues = storage.get_issue_numbers(args.repo)

    if last_api_call and not args.force_refresh and args.update_mode:
        print(f"Last API call timestamp: {last_api_call} (UTC)")
        print(f"Found {len(existing_issues)} existing issues in the database")
        print(f"Fetching issues updated since {last_api_call}...")
    else:
        if args.force_refresh:
            print("Force refresh enabled. Fetching all issues...")
        elif args.update_mode:
            print("No existing issues found. Fetching all issues...")
        else:
            print("Normal mode enabled. Fetching all issues...")

    # Fetch issues
    start_time = time.time()

    issues = fetcher.fetch_updated_issues(
        repo_name=args.repo,
        state=args.state,
        max_issues=args.max_issues,
        include_comments=args.include_comments,
        last_update_timestamp=last_api_call,
        existing_issue_numbers=existing_issues,
    )

    fetch_time = time.time() - start_time

    # Get API rate limit information
    rate_limit_info = fetcher.get_rate_limit_info()
    rate_limit_remaining = rate_limit_info.get("remaining")
    rate_limit_total = rate_limit_info.get("limit")
    rate_limit_reset = rate_limit_info.get("reset")

    if rate_limit_remaining is not None and rate_limit_total is not None:
        print(f"API Rate Limit: {rate_limit_remaining}/{rate_limit_total}")
        if rate_limit_reset:
            print(f"Rate limit resets at: {rate_limit_reset}")

    if issues:
        print(f"Fetched {len(issues)} issues in {fetch_time:.2f} seconds")

        # Store issues in SQLite
        print(f"Storing issues in SQLite database at {db_path}...")
        store_start_time = time.time()

        # Store issues and get counts
        issue_counts = storage.store_issues(issues, args.repo)

        store_time = time.time() - store_start_time
        total_execution_time = fetch_time + store_time
        print(f"Stored {len(issues)} issues in {store_time:.2f} seconds")

        # Log the API call with rate limit and execution time
        storage.log_api_call(
            repo_name=args.repo,
            new_issues_count=issue_counts["new_issues_count"],
            updated_issues_count=issue_counts["updated_issues_count"],
            redundant_issues_count=issue_counts["redundant_issues_count"],
            issues_before_count=issue_counts["issues_before_count"],
            issues_after_count=issue_counts["issues_after_count"],
            api_rate_limit_remaining=rate_limit_remaining,
            api_rate_limit_total=rate_limit_total,
            execution_time_seconds=total_execution_time,
        )
    else:
        print("No new or updated issues to store")

        # Log the API call even if no issues were found
        storage.log_api_call(
            repo_name=args.repo,
            new_issues_count=0,
            updated_issues_count=0,
            redundant_issues_count=0,
            issues_before_count=len(existing_issues),
            issues_after_count=len(existing_issues),
            api_rate_limit_remaining=rate_limit_remaining,
            api_rate_limit_total=rate_limit_total,
            execution_time_seconds=fetch_time,
        )

    # Show total count of issues in the database
    total_issues = len(storage.get_issues(repo_name=args.repo))
    print(f"Total issues in database for {args.repo}: {total_issues}")

    print("Done!")


if __name__ == "__main__":
    main()
