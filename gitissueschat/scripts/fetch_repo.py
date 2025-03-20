#!/usr/bin/env python3
"""
Script to fetch issues from a specific GitHub repository and store them in the database.
"""

import os
import argparse
import time
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from gitissueschat.github_issues import GitHubIssuesFetcher
from gitissueschat.sqlite_storage.sqlite_storage import SQLiteIssueStorage


def main():
    """Main function to fetch issues from a GitHub repository."""
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Fetch issues from a GitHub repository")
    parser.add_argument("repo", help="GitHub repository in the format 'owner/repo'")
    parser.add_argument("--token", help="GitHub API token (or set GITHUB_TOKEN env var)")
    parser.add_argument(
        "--state",
        default="all",
        choices=["open", "closed", "all"],
        help="State of issues to fetch (default: all)",
    )
    parser.add_argument("--max-issues", type=int, help="Maximum number of issues to fetch")
    parser.add_argument(
        "--include-comments", action="store_true", help="Include comments in fetched issues"
    )
    parser.add_argument(
        "--update-mode",
        action="store_true",
        help="Only fetch issues updated since the last API call",
    )
    parser.add_argument(
        "--db-path",
        default="github_issues.db",
        help="Path to SQLite database (default: github_issues.db)",
    )
    parser.add_argument("--show-api-logs", action="store_true", help="Show API logs after fetching")

    args = parser.parse_args()

    # Get GitHub token from args or environment variable
    github_token = args.token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print(
            "Error: GitHub token is required. Provide it with --token or set GITHUB_TOKEN environment variable."
        )
        exit(1)

    # Initialize the GitHub issues fetcher
    fetcher = GitHubIssuesFetcher(github_token)

    # Initialize the SQLite issue storage
    storage = SQLiteIssueStorage(args.db_path)

    # Get existing issue numbers
    existing_issue_numbers = set(storage.get_issue_numbers(args.repo))

    # Get the latest API call timestamp
    last_update_timestamp = None
    if args.update_mode:
        last_api_call = storage.get_latest_api_call_timestamp(args.repo)
        if last_api_call:
            last_update_timestamp = last_api_call
            print(
                f"Update mode enabled. Only fetching issues updated since {last_update_timestamp} (UTC)"
            )
        else:
            print("No previous API calls found. Fetching all issues...")
    else:
        print("Normal mode enabled. Fetching all issues...")

    # Fetch issues with batch processing
    batch_size = 100  # Process 100 issues at a time
    total_issues = 0
    total_new = 0
    total_updated = 0
    total_redundant = 0
    
    # Start timing the execution
    start_time = time.time()
    
    # Configure the fetcher to use a callback for batch processing
    def process_batch(batch_issues, batch_stats):
        nonlocal total_issues, total_new, total_updated, total_redundant
        
        # Update counts
        total_issues += len(batch_issues)
        total_new += batch_stats.get("new_count", 0)
        total_updated += batch_stats.get("updated_count", 0)
        total_redundant += batch_stats.get("redundant_count", 0)
        
        # Store this batch in the database
        if batch_issues:
            storage.store_issues(batch_issues, args.repo)
    
    # Fetch issues with batch processing
    fetcher.fetch_updated_issues(
        repo_name=args.repo,
        state=args.state,
        max_issues=args.max_issues,
        include_comments=args.include_comments,
        last_update_timestamp=last_update_timestamp,
        existing_issue_numbers=existing_issue_numbers,
        update_mode=args.update_mode,
        batch_size=batch_size,
        batch_callback=process_batch
    )
    
    print(
        f"Fetched {total_issues} issues: {total_new} new, {total_updated} updated, {total_redundant} redundant"
    )

    # Get API rate limit info
    rate_limit_info = fetcher.get_rate_limit_info()
    remaining = rate_limit_info.get("remaining", 0)
    limit = rate_limit_info.get("limit", 0)
    reset_time = rate_limit_info.get("reset_time", "")

    print(f"API Rate Limit: {remaining}/{limit}")
    print(f"Rate limit resets at: {reset_time}")

    # Calculate execution time
    execution_time = time.time() - start_time

    # Log the API call
    storage.log_api_call(
        repo_name=args.repo,
        new_issues_count=total_new,
        updated_issues_count=total_updated,
        redundant_issues_count=total_redundant,
        issues_before_count=storage.get_issue_count(args.repo),
        issues_after_count=storage.get_issue_count(args.repo),
        api_rate_limit_remaining=remaining,
        api_rate_limit_total=limit,
        execution_time_seconds=execution_time,
    )

    print(f"Total execution time: {execution_time:.2f} seconds")

    # Show API logs if requested
    if args.show_api_logs:
        print("\nAPI Logs:\n")
        logs = storage.get_api_logs(args.repo, limit=3)
        for i, log in enumerate(logs):
            print(f"Log #{i+1} - {log['timestamp']} (UTC):")
            print(f"  Repository: {args.repo}")
            print(f"  New issues: {log['new_issues_count']}")
            print(f"  Updated issues: {log['updated_issues_count']}")
            print(f"  Redundant issues: {log['redundant_issues_count']}")
            print(f"  Issues before: {log['issues_before_count']}")
            print(f"  Issues after: {log['issues_after_count']}")
            print(
                f"  API Rate Limit: {log['api_rate_limit_remaining']}/{log['api_rate_limit_total']}"
            )
            print(f"  Execution time: {log['execution_time_seconds']:.2f} seconds")
            print()

        # Show database statistics
        print("Database Statistics:")
        total_issues = storage.get_issue_count(args.repo)
        open_issues = storage.get_issue_count(args.repo, state="open")
        closed_issues = storage.get_issue_count(args.repo, state="closed")
        total_comments = storage.get_comment_count(args.repo)

        print(f"Total issues for {args.repo}: {total_issues}")
        print(f"Open issues: {open_issues}")
        print(f"Closed issues: {closed_issues}")
        print(f"Total comments: {total_comments}")
        print()

        # Show most recent issues
        print("Most recent issues:")
        recent_issues = storage.get_issues(args.repo, limit=3)
        for i, issue in enumerate(recent_issues):
            print(f"{i+1}. #{issue['number']}: {issue['title']} ({issue['state']})")
        print()

        # Show issues with most comments
        print("Issues with most comments:")
        issues_with_comments = storage.get_issues_with_most_comments(args.repo, limit=3)
        for i, issue in enumerate(issues_with_comments):
            comment_count = len(issue.get("comments", []))
            print(f"{i+1}. #{issue['number']}: {issue['title']} - {comment_count} comments")


if __name__ == "__main__":
    main()
