"""
GitHub Issues Fetcher

This module provides functionality to fetch issues from a GitHub repository.
"""

import os
import time
import datetime
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from github import Github, Issue, Repository, PaginatedList
from github.GithubException import RateLimitExceededException
from tqdm import tqdm
from datetime import datetime
import pytz
from dateutil import parser


class GitHubIssuesFetcher:
    """
    A class to fetch issues from a GitHub repository.
    """

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHub issue fetcher.

        Args:
            token: GitHub personal access token. If None, uses anonymous access with rate limits.
        """
        self.github = Github(token) if token else Github()
        self.rate_limit_cooldown = 60  # seconds to wait when rate limit is hit

    def get_repository(self, repo_name: str) -> Repository.Repository:
        """
        Get a GitHub repository by name.

        Args:
            repo_name: The full name of the repository (e.g., "username/repo").

        Returns:
            The GitHub repository object.
        """
        # Cache repositories to avoid redundant API calls
        if not hasattr(self, "_repo_cache"):
            self._repo_cache = {}

        if repo_name not in self._repo_cache:
            self._repo_cache[repo_name] = self.github.get_repo(repo_name)

        return self._repo_cache[repo_name]

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """
        Get the current rate limit information from GitHub API.

        Returns:
            Dictionary containing rate limit information with keys:
            - remaining: Number of requests remaining in the current rate limit window
            - limit: Total number of requests allowed per rate limit window
            - reset: Datetime when the rate limit will reset
        """
        try:
            rate_limit = self.github.get_rate_limit()
            return {
                "remaining": rate_limit.core.remaining,
                "limit": rate_limit.core.limit,
                "reset": rate_limit.core.reset,
            }
        except Exception as e:
            print(f"Error getting rate limit information: {str(e)}")
            return {"remaining": None, "limit": None, "reset": None}

    def fetch_issues(
        self,
        repo_name: str,
        state: str = "all",
        max_issues: Optional[int] = None,
        include_comments: bool = True,
        since: Optional[str] = None,
        existing_issue_numbers: Optional[Set[int]] = None,
        existing_issue_codes: Optional[Set[str]] = None,
        batch_size: int = 100,
        batch_callback: Optional[Callable[[List[Dict[str, Any]], Dict[str, int]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Fetch issues from a GitHub repository.

        Args:
            repo_name: Name of the repository in the format 'owner/repo'.
            state: State of issues to fetch ('open', 'closed', 'all').
            max_issues: Maximum number of issues to fetch.
            include_comments: Whether to include comments in the fetched issues.
            since: Only fetch issues updated after this timestamp.
            existing_issue_numbers: Set of existing issue numbers to check against.
            batch_size: Number of issues to process in each batch.
            batch_callback: Callback function to process each batch of issues.

        Returns:
            Dictionary with structured issues and counts.
        """
        try:
            # Get the repository
            repo = self.get_repository(repo_name)

            # Debug information
            if since:
                print(f"DEBUG: Fetching issues updated since {since} (UTC)")

            # Get issues from the repository
            kwargs = {"state": state, "sort": "updated", "direction": "desc"}

            if since:
                # Convert the ISO format timestamp string to a datetime object
                # The GitHub API expects a datetime object for the since parameter
                try:
                    since_dt = parser.parse(since)
                    since_dt = since_dt.astimezone(pytz.utc)
                    kwargs["since"] = since_dt
                except ValueError as e:
                    print(f"Error parsing timestamp: {since}")
                    print(f"Error details: {str(e)}")
                    # If we can't parse the timestamp, don't use it
                    pass

            # Get issues with pagination
            issues = repo.get_issues(**kwargs)

            # Try to get the first page of issues to check if there are any
            try:
                # Get the first issue to check if there are any
                first_page = list(issues[:1])
                if not first_page:
                    print(f"No issues found in {repo_name} matching the criteria")
                    return {"issues": [], "new_count": 0, "updated_count": 0, "redundant_count": 0}

                # Reset the iterator
                issues = repo.get_issues(**kwargs)

                # Determine how many issues to fetch
                total_to_fetch = max_issues if max_issues else issues.totalCount

                # Prepare to fetch issues in batches
                structured_issues = []
                new_count = 0
                updated_count = 0
                redundant_count = 0

                # Use a progress bar for better visibility
                with tqdm(total=total_to_fetch, desc="Fetching all issues") as pbar:
                    # Process issues in batches of batch_size
                    batch_start = 0
                    while batch_start < total_to_fetch:
                        # Calculate how many issues to fetch in this batch
                        current_batch_size = min(batch_size, total_to_fetch - batch_start)
                        
                        # Get a batch of issues
                        batch_end = batch_start + current_batch_size
                        issue_batch = list(issues[batch_start:batch_end])
                        
                        # Process each issue in the batch
                        batch_structured_issues = []
                        batch_new_count = 0
                        batch_updated_count = 0
                        batch_redundant_count = 0
                        
                        # Process the filtered batch
                        for issue in issue_batch:
                            github_updated_at = issue.updated_at.strftime("%Y-%m-%d %H:%M:%S")
                            issue_code = f"{issue.number}_{github_updated_at}"
                            # Check if the issue is new or updated
                            if existing_issue_numbers and existing_issue_codes:
                                if issue_code in existing_issue_codes:
                                    # this is a redundant issue
                                    batch_redundant_count += 1
                                    pbar.update(1)
                                    continue
                                elif issue.number in existing_issue_numbers:
                                    # this is an updated issue
                                    batch_updated_count += 1
                                else:
                                    # this is a new issue
                                    batch_new_count += 1
                            elif existing_issue_numbers:
                                if issue.number in existing_issue_numbers:
                                    # this is a redundant issue
                                    batch_redundant_count += 1
                                    pbar.update(1)
                                    continue
                                else:
                                    # this is a new issue
                                    batch_new_count += 1
                            elif existing_issue_codes:
                                if issue_code in existing_issue_codes:
                                    # this is a redundant issue
                                    batch_redundant_count += 1
                                    pbar.update(1)
                                    continue
                                else:
                                    # this is a new issue
                                    batch_new_count += 1
                            else:
                                # this is a new issue
                                batch_new_count += 1

                            # Structure the issue data
                            structured_issue = self._structure_issue(issue, include_comments)
                            batch_structured_issues.append(structured_issue)

                            pbar.update(1)

                        # Call the batch callback if provided
                        if batch_callback and batch_structured_issues:
                            batch_callback(batch_structured_issues, {
                                "new_count": batch_new_count,
                                "updated_count": batch_updated_count,
                                "redundant_count": batch_redundant_count,
                            })

                        # Update the counts
                        new_count += batch_new_count
                        updated_count += batch_updated_count
                        redundant_count += batch_redundant_count
                        structured_issues.extend(batch_structured_issues)

                        # Move to the next batch
                        batch_start += current_batch_size

                print(
                    f"Fetched {len(structured_issues)} issues: {new_count} new, {updated_count} updated, {redundant_count} redundant"
                )
                return {
                    "issues": structured_issues,
                    "new_count": new_count,
                    "updated_count": updated_count,
                    "redundant_count": redundant_count,
                }

            except IndexError:
                # No issues found
                print(f"No issues found in {repo_name} matching the criteria")
                return {"issues": [], "new_count": 0, "updated_count": 0, "redundant_count": 0}

        except RateLimitExceededException:
            reset_time = self.github.get_rate_limit().core.reset
            current_time = datetime.now()
            wait_time = (reset_time - current_time).total_seconds()

            print(f"Rate limit exceeded. Reset at {reset_time} (in {wait_time:.2f} seconds)")

            if wait_time > 0 and wait_time < 7200:  # Wait if less than 2 hours
                print(f"Waiting {wait_time:.2f} seconds for rate limit to reset...")
                time.sleep(wait_time + 5)  # Add 5 seconds buffer
                return self.fetch_issues(repo_name, state, max_issues, include_comments, since)
            else:
                raise
        except Exception as e:
            print(f"Error fetching issues: {str(e)}")
            raise

    def fetch_issue_by_number(
        self, repo_name: str, issue_number: int, include_comments: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific issue by its number.

        Args:
            repo_name: The full name of the repository (e.g., "username/repo").
            issue_number: The issue number to fetch.
            include_comments: Whether to include comments in the fetched issue.

        Returns:
            A dictionary containing issue data, or None if the issue doesn't exist.
        """
        repo = self.get_repository(repo_name)

        try:
            issue = repo.get_issue(issue_number)
            return self._structure_issue(issue, include_comments)
        except RateLimitExceededException:
            reset_time = self.github.get_rate_limit().core.reset
            current_time = datetime.now()
            wait_time = (reset_time - current_time).total_seconds()

            print(f"Rate limit exceeded. Reset at {reset_time} (in {wait_time:.2f} seconds)")

            if wait_time > 0 and wait_time < 7200:  # Wait if less than 2 hours
                print(f"Waiting {wait_time:.2f} seconds for rate limit to reset...")
                time.sleep(wait_time + 5)  # Add 5 seconds buffer
                return self.fetch_issue_by_number(repo_name, issue_number, include_comments)
            else:
                raise
        except Exception as e:
            print(f"Error fetching issue #{issue_number}: {str(e)}")
            return None

    def fetch_updated_issues(
        self,
        repo_name: str,
        state: str = "all",
        max_issues: Optional[int] = None,
        include_comments: bool = True,
        last_update_timestamp: Optional[str] = None,
        existing_issue_numbers: Optional[Set[int]] = None,
        update_mode: bool = True,
        batch_size: int = 100,
        batch_callback: Optional[Callable[[List[Dict[str, Any]], Dict[str, int]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Fetch only new or updated issues from a GitHub repository.

        Args:
            repo_name: Name of the repository in the format 'owner/repo'.
            state: State of issues to fetch ('open', 'closed', 'all').
            max_issues: Maximum number of issues to fetch.
            include_comments: Whether to include comments in the fetched issues.
            last_update_timestamp: Only fetch issues updated after this timestamp.
            existing_issue_numbers: Set of issue numbers that already exist in the database.
            update_mode: If True, only fetch issues updated since last_update_timestamp.
                        If False, fetch all issues regardless of last_update_timestamp.
            batch_size: Number of issues to process in each batch.
            batch_callback: Callback function to process each batch of issues.

        Returns:
            Dictionary containing:
            - issues: List of structured issue data
            - new_count: Number of new issues
            - updated_count: Number of updated issues
            - redundant_count: Number of redundant issues
        """
        # If update mode is enabled and we have a last update timestamp, use it
        since = last_update_timestamp if update_mode and last_update_timestamp else None

        # Fetch issues
        return self.fetch_issues(
            repo_name=repo_name,
            state=state,
            max_issues=max_issues,
            include_comments=include_comments,
            since=since,
            existing_issue_numbers=existing_issue_numbers,
            batch_size=batch_size,
            batch_callback=batch_callback,
        )

    def _structure_issue(self, issue: Issue.Issue, include_comments: bool = True) -> Dict[str, Any]:
        """
        Structure a GitHub issue into a dictionary format.

        Args:
            issue: The GitHub issue object.
            include_comments: Whether to include comments in the structured issue.

        Returns:
            A dictionary containing structured issue data.
        """
        # Fetch all comments for the issue if requested
        comments = []
        if include_comments:
            for comment in issue.get_comments():
                comments.append(
                    {
                        "id": comment.id,
                        "user": (
                            {"login": comment.user.login} if comment.user else {"login": "Unknown"}
                        ),
                        "created_at": comment.created_at.isoformat(),
                        "updated_at": (
                            comment.updated_at.isoformat() if comment.updated_at else None
                        ),
                        "body": comment.body,
                    }
                )

        # Structure the issue data
        structured_issue = {
            "id": issue.id,
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "created_at": issue.created_at.isoformat(),
            "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
            "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
            "user": {"login": issue.user.login} if issue.user else {"login": "Unknown"},
            "body": issue.body,
            "labels": [{"name": label.name} for label in issue.labels],
            "comments": comments,
            "comments_count": issue.comments,
            "html_url": issue.html_url,
            "repository": issue.repository.full_name,
        }

        return structured_issue
