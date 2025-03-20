"""
SQLite Connection Manager

This module provides a class for managing SQLite database connections.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Tuple


class SQLiteConnectionManager:
    """A class for managing SQLite database connections."""

    def __init__(self, db_path: str = "./github_issues.db", timeout: int = 30):
        """
        Initialize the SQLite connection manager.

        Args:
            db_path: Path to the SQLite database file.
            timeout: Timeout in seconds for database operations.
        """
        self.db_path = db_path
        self.timeout = timeout
        self._create_tables()

    def get_connection(self):
        """
        Get a database connection with proper settings.

        Returns:
            SQLite connection object.
        """
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        """
        Create the necessary tables if they don't exist.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Create repositories table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS repositories (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            )

            # Create issues table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY,
                repo_id INTEGER,
                number INTEGER,
                title TEXT,
                body TEXT,
                state TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                closed_at TIMESTAMP,
                author TEXT,
                html_url TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (repo_id) REFERENCES repositories(id),
                UNIQUE (repo_id, number)
            )
            """
            )

            # Create labels table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE
            )
            """
            )

            # Create issue_labels table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS issue_labels (
                issue_id INTEGER,
                label_id INTEGER,
                PRIMARY KEY (issue_id, label_id),
                FOREIGN KEY (issue_id) REFERENCES issues(id),
                FOREIGN KEY (label_id) REFERENCES labels(id)
            )
            """
            )

            # Create comments table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY,
                issue_id INTEGER,
                body TEXT,
                author TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (issue_id) REFERENCES issues(id)
            )
            """
            )

            # Create API logs table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY,
                repo_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                new_issues_count INTEGER DEFAULT 0,
                updated_issues_count INTEGER DEFAULT 0,
                redundant_issues_count INTEGER DEFAULT 0,
                issues_before_count INTEGER DEFAULT 0,
                issues_after_count INTEGER DEFAULT 0,
                api_rate_limit_remaining INTEGER,
                api_rate_limit_total INTEGER,
                execution_time_seconds REAL,
                FOREIGN KEY (repo_id) REFERENCES repositories(id)
            )
            """
            )

            conn.commit()

    def execute_query(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute a custom SQL query.

        Args:
            query: SQL query string.
            params: Query parameters.

        Returns:
            List of results as dictionaries.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)

            # Convert results to dictionaries
            results = []
            for row in cursor.fetchall():
                results.append({key: row[key] for key in row.keys()})

            return results

    def parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[str]:
        """
        Parse a timestamp string to a SQLite-compatible format.
        Converts all timestamps to UTC and removes timezone information.

        Args:
            timestamp_str: ISO format timestamp string or None.

        Returns:
            SQLite-compatible timestamp string in UTC format without timezone info, or None.
        """
        if not timestamp_str:
            return None

        import datetime
        import re

        # Remove timezone information if present
        # GitHub API returns timestamps in ISO 8601 format with Z suffix (UTC)
        timestamp_str = re.sub(r"Z$", "", timestamp_str)
        timestamp_str = re.sub(r"[+-]\d{2}:\d{2}$", "", timestamp_str)

        try:
            # Parse the timestamp
            dt = datetime.datetime.fromisoformat(timestamp_str)

            # Convert to UTC if timezone info is present
            if dt.tzinfo:
                dt = dt.astimezone(datetime.timezone.utc)

            # Remove timezone info for SQLite compatibility
            dt = dt.replace(tzinfo=None)

            # Format for SQLite
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            # If we can't parse it, return as is
            return timestamp_str
