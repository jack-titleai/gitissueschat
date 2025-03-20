"""
SQLite Issue Storage

This module provides the main SQLiteIssueStorage class that integrates all storage components.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from .connection_manager import SQLiteConnectionManager
from .repository_manager import RepositoryManager
from .issue_manager import IssueManager
from .api_log_manager import APILogManager


def dict_factory(cursor, row):
    """Convert a row to a dictionary."""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


class SQLiteIssueStorage:
    """A class for storing GitHub issues in a SQLite database."""

    def __init__(self, db_path: str = "./github_issues.db", timeout: int = 30):
        """
        Initialize the SQLite issue storage.

        Args:
            db_path: Path to the SQLite database file.
            timeout: Timeout in seconds for database operations.
        """
        self.connection_manager = SQLiteConnectionManager(db_path, timeout)
        self.repository_manager = RepositoryManager(self.connection_manager)
        self.issue_manager = IssueManager(self.connection_manager, self.repository_manager)
        self.api_log_manager = APILogManager(self.connection_manager, self.repository_manager)
    
    def create_tables(self):
        """Create all necessary tables if they don't exist."""
        # Create tables in each manager
        self.repository_manager.create_tables()
        self.issue_manager.create_tables()
        self.api_log_manager.create_tables()
    
    def store_issues(self, issues: List[Dict[str, Any]], repo_name: str) -> Dict[str, int]:
        """
        Store GitHub issues in the SQLite database.

        Args:
            issues: List of GitHub issues to store.
            repo_name: Name of the repository the issues belong to.

        Returns:
            Dictionary with counts of new, updated, redundant, and total issues before and after the update.
        """
        return self.issue_manager.store_issues(issues, repo_name)

    def get_issue_numbers(self, repo_name: str) -> List[int]:
        """
        Get the issue numbers for a repository.

        Args:
            repo_name: Name of the repository.

        Returns:
            List of issue numbers.
        """
        return self.issue_manager.get_issue_numbers(repo_name)
    
    def get_issue_codes(self, repo_name: str) -> List[str]:
        """
        Get the issue numbers for a repository.

        Args:
            repo_name: Name of the repository.

        Returns:
            List of strings where each string is a combination of issue number and updated_at date
        """
        return self.issue_manager.get_issue_codes(repo_name)

    def get_issues(
        self, repo_name: str, state: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get issues for a repository.

        Args:
            repo_name: Name of the repository.
            state: State of issues to get ('open', 'closed', or None for all).
            limit: Maximum number of issues to get.

        Returns:
            List of issues.
        """
        return self.issue_manager.get_issues(repo_name, state, limit)

    def get_issue_count(self, repo_name: str, state: Optional[str] = None) -> int:
        """
        Get the number of issues for a repository.

        Args:
            repo_name: Name of the repository.
            state: State of issues to count ('open', 'closed', or None for all).

        Returns:
            Number of issues.
        """
        return self.issue_manager.get_issue_count(repo_name, state)

    def get_comment_count(self, repo_name: str) -> int:
        """
        Get the number of comments for a repository.

        Args:
            repo_name: Name of the repository.

        Returns:
            Number of comments.
        """
        return self.issue_manager.get_comment_count(repo_name)

    def get_issues_with_most_comments(self, repo_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get issues with the most comments.

        Args:
            repo_name: Name of the repository.
            limit: Maximum number of issues to get.

        Returns:
            List of issues with comment counts.
        """
        return self.issue_manager.get_issues_with_most_comments(repo_name, limit)

    def log_api_call(
        self,
        repo_name: str,
        new_issues_count: int = 0,
        updated_issues_count: int = 0,
        redundant_issues_count: int = 0,
        issues_before_count: int = 0,
        issues_after_count: int = 0,
        api_rate_limit_remaining: Optional[int] = None,
        api_rate_limit_total: Optional[int] = None,
        execution_time_seconds: Optional[float] = None,
    ) -> int:
        """
        Log an API call.

        Args:
            repo_name: Name of the repository.
            new_issues_count: Number of new issues.
            updated_issues_count: Number of updated issues.
            redundant_issues_count: Number of redundant issues.
            issues_before_count: Number of issues before the update.
            issues_after_count: Number of issues after the update.
            api_rate_limit_remaining: Remaining API rate limit.
            api_rate_limit_total: Total API rate limit.
            execution_time_seconds: Execution time in seconds.

        Returns:
            The log ID.
        """
        return self.api_log_manager.log_api_call(
            repo_name,
            new_issues_count,
            updated_issues_count,
            redundant_issues_count,
            issues_before_count,
            issues_after_count,
            api_rate_limit_remaining,
            api_rate_limit_total,
            execution_time_seconds,
        )

    def get_api_logs(self, repo_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get API logs for a repository.

        Args:
            repo_name: Name of the repository.
            limit: Maximum number of logs to get.

        Returns:
            List of API logs.
        """
        return self.api_log_manager.get_api_logs(repo_name, limit)

    def has_api_call_log(self, repo_name: str, timestamp: str) -> bool:
        """
        Check if an API call log exists for a repository at a specific timestamp.

        Args:
            repo_name: Name of the repository.
            timestamp: Timestamp to check.

        Returns:
            True if an API call log exists, False otherwise.
        """
        return self.api_log_manager.has_api_call_log(repo_name, timestamp)

    def get_latest_api_call_timestamp(self, repo_name: str) -> Optional[str]:
        """
        Get the timestamp of the latest API call for a repository.

        Args:
            repo_name: Name of the repository.

        Returns:
            Timestamp of the latest API call, or None if there are no API calls.
        """
        return self.issue_manager.get_latest_api_call_timestamp(repo_name)

    def execute_query(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a custom SQL query.

        Args:
            query: SQL query string.
            params: Query parameters.

        Returns:
            List of results as dictionaries.
        """
        return self.connection_manager.execute_query(query, params)
