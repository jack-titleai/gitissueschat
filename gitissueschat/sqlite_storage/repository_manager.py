"""
Repository Manager

This module provides a class for managing repository data in the SQLite database.
"""

from typing import Optional
from .connection_manager import SQLiteConnectionManager


class RepositoryManager:
    """A class for managing repository data in the SQLite database."""

    def __init__(self, connection_manager: SQLiteConnectionManager):
        """
        Initialize the repository manager.

        Args:
            connection_manager: The SQLite connection manager.
        """
        self.connection_manager = connection_manager

    def create_tables(self):
        """Create the repositories table if it doesn't exist."""
        conn = self.connection_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS repositories (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()

    def get_or_create_repo(self, repo_name: str) -> int:
        """
        Get or create a repository record.

        Args:
            repo_name: Name of the repository.

        Returns:
            The repository ID.
        """
        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Check if repository exists
            cursor.execute("SELECT id FROM repositories WHERE name = ?", (repo_name,))
            result = cursor.fetchone()

            if result:
                return result[0]

            # Create new repository with current timestamp
            cursor.execute(
                "INSERT INTO repositories (name, added_at) VALUES (?, datetime('now'))",
                (repo_name,),
            )

            return cursor.lastrowid

    def get_repo_id(self, repo_name: str) -> Optional[int]:
        """
        Get the ID of a repository.

        Args:
            repo_name: Name of the repository.

        Returns:
            The repository ID, or None if the repository doesn't exist.
        """
        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT id FROM repositories WHERE name = ?", (repo_name,))
            result = cursor.fetchone()

            if result:
                return result[0]

            return None
