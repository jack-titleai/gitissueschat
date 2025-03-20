"""
API Log Manager

This module provides a class for managing API log data in the SQLite database.
"""

from typing import Any, Dict, List, Optional
from .connection_manager import SQLiteConnectionManager
from .repository_manager import RepositoryManager


class APILogManager:
    """A class for managing API log data in the SQLite database."""

    def __init__(
        self, connection_manager: SQLiteConnectionManager, repository_manager: RepositoryManager
    ):
        """
        Initialize the API log manager.

        Args:
            connection_manager: The SQLite connection manager.
            repository_manager: The repository manager.
        """
        self.connection_manager = connection_manager
        self.repository_manager = repository_manager

    def create_tables(self):
        """Create the api_logs table if it doesn't exist."""
        conn = self.connection_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY,
                repo_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                new_issues_count INTEGER,
                updated_issues_count INTEGER,
                redundant_issues_count INTEGER,
                issues_before_count INTEGER,
                issues_after_count INTEGER,
                api_rate_limit_remaining INTEGER,
                api_rate_limit_total INTEGER,
                execution_time_seconds REAL,
                FOREIGN KEY (repo_id) REFERENCES repositories(id)
            )
        """)
        
        conn.commit()

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
        repo_id = self.repository_manager.get_or_create_repo(repo_name)

        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO api_logs (
                    repo_id, new_issues_count, updated_issues_count, redundant_issues_count,
                    issues_before_count, issues_after_count,
                    api_rate_limit_remaining, api_rate_limit_total, execution_time_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    repo_id,
                    new_issues_count,
                    updated_issues_count,
                    redundant_issues_count,
                    issues_before_count,
                    issues_after_count,
                    api_rate_limit_remaining,
                    api_rate_limit_total,
                    execution_time_seconds,
                ),
            )

            return cursor.lastrowid

    def get_api_logs(self, repo_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get API logs for a repository.

        Args:
            repo_name: Name of the repository.
            limit: Maximum number of logs to get.

        Returns:
            List of API logs.
        """
        repo_id = self.repository_manager.get_repo_id(repo_name)
        if not repo_id:
            return []

        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM api_logs WHERE repo_id = ? ORDER BY timestamp DESC"
            params = [repo_id]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)

            return [dict(row) for row in cursor.fetchall()]

    def has_api_call_log(self, repo_name: str, timestamp: str) -> bool:
        """
        Check if an API call log exists for a repository at a specific timestamp.

        Args:
            repo_name: Name of the repository.
            timestamp: Timestamp to check.

        Returns:
            True if an API call log exists, False otherwise.
        """
        repo_id = self.repository_manager.get_repo_id(repo_name)
        if not repo_id:
            return False

        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM api_logs WHERE repo_id = ? AND timestamp = ?",
                (repo_id, timestamp),
            )

            return cursor.fetchone()[0] > 0
