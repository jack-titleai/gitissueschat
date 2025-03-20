"""
Issue Manager

This module provides a class for managing issue data in the SQLite database.
"""

from typing import Any, Dict, List, Optional, Set
from .connection_manager import SQLiteConnectionManager
from .repository_manager import RepositoryManager


class IssueManager:
    """A class for managing issue data in the SQLite database."""

    def __init__(
        self, connection_manager: SQLiteConnectionManager, repository_manager: RepositoryManager
    ):
        """
        Initialize the issue manager.

        Args:
            connection_manager: The SQLite connection manager.
            repository_manager: The repository manager.
        """
        self.connection_manager = connection_manager
        self.repository_manager = repository_manager

    def create_tables(self):
        """Create the issues and comments tables if they don't exist."""
        conn = self.connection_manager.get_connection()
        cursor = conn.cursor()
        
        # Create issues table
        cursor.execute("""
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
                FOREIGN KEY (repo_id) REFERENCES repositories(id),
                UNIQUE (repo_id, number)
            )
        """)
        
        # Create comments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY,
                issue_id INTEGER,
                body TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                author TEXT,
                html_url TEXT,
                FOREIGN KEY (issue_id) REFERENCES issues(id)
            )
        """)
        
        conn.commit()

    def store_issues(self, issues: List[Dict[str, Any]], repo_name: str) -> Dict[str, int]:
        """
        Store GitHub issues in the SQLite database.

        Args:
            issues: List of GitHub issues to store.
            repo_name: Name of the repository the issues belong to.

        Returns:
            Dictionary with counts of new, updated, redundant, and total issues before and after the update.
        """
        if not issues:
            return {}

        repo_id = self.repository_manager.get_or_create_repo(repo_name)

        # Count issues before the update
        issues_before_count = len(self.get_issue_numbers(repo_name))

        # Track new and updated issues
        new_issues_count = 0
        updated_issues_count = 0
        redundant_issues_count = 0

        # Get existing issue numbers
        existing_issue_numbers = set(self.get_issue_numbers(repo_name))

        # Get the latest API call timestamp
        last_api_call = self.get_latest_api_call_timestamp(repo_name)

        # Use a single connection for the entire operation
        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Pre-fetch or create all labels to avoid nested transactions
            label_ids = {}
            for issue in issues:
                for label in issue.get("labels", []):
                    label_name = label.get("name") if isinstance(label, dict) else label
                    if label_name and label_name not in label_ids:
                        # Check if label exists
                        cursor.execute("SELECT id FROM labels WHERE name = ?", (label_name,))
                        result = cursor.fetchone()

                        if result:
                            label_ids[label_name] = result[0]
                        else:
                            # Create new label
                            cursor.execute("INSERT INTO labels (name) VALUES (?)", (label_name,))
                            label_ids[label_name] = cursor.lastrowid

            for issue in issues:
                # Determine if this is a new, updated, or redundant issue
                is_new = issue["number"] not in existing_issue_numbers
                is_redundant = False

                if is_new:
                    new_issues_count += 1
                elif last_api_call:
                    # Check if the issue was updated after the last API call
                    issue_updated_at = issue.get("updated_at")
                    if issue_updated_at and issue_updated_at > last_api_call:
                        updated_issues_count += 1
                    else:
                        redundant_issues_count += 1
                        is_redundant = True
                else:
                    updated_issues_count += 1

                # Skip redundant issues - don't write them to the database
                if is_redundant:
                    continue

                # Insert issue
                issue_id = issue["id"]
                created_at = self.connection_manager.parse_timestamp(issue.get("created_at"))
                updated_at = self.connection_manager.parse_timestamp(issue.get("updated_at"))
                closed_at = self.connection_manager.parse_timestamp(issue.get("closed_at"))

                cursor.execute(
                    """
                    INSERT OR REPLACE INTO issues (
                        id, repo_id, number, title, body, state, 
                        created_at, updated_at, closed_at, author, html_url, added_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        issue_id,
                        repo_id,
                        issue["number"],
                        issue.get("title", ""),
                        issue.get("body", ""),
                        issue.get("state", ""),
                        created_at,
                        updated_at,
                        closed_at,
                        (
                            issue.get("user", {}).get("login", "")
                            if isinstance(issue.get("user"), dict)
                            else ""
                        ),
                        issue.get("html_url", ""),
                    ),
                )

                # Clear existing labels for this issue
                cursor.execute("DELETE FROM issue_labels WHERE issue_id = ?", (issue_id,))

                # Add labels
                for label in issue.get("labels", []):
                    label_name = label.get("name") if isinstance(label, dict) else label
                    if label_name and label_name in label_ids:
                        cursor.execute(
                            "INSERT OR IGNORE INTO issue_labels (issue_id, label_id) VALUES (?, ?)",
                            (issue_id, label_ids[label_name]),
                        )

                # Add comments
                if "comments" in issue and issue["comments"]:
                    # Clear existing comments for this issue
                    cursor.execute("DELETE FROM comments WHERE issue_id = ?", (issue_id,))

                    for comment in issue["comments"]:
                        comment_id = comment.get("id")
                        created_at = self.connection_manager.parse_timestamp(
                            comment.get("created_at")
                        )
                        updated_at = self.connection_manager.parse_timestamp(
                            comment.get("updated_at")
                        )

                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO comments (
                                id, issue_id, body, author, created_at, updated_at, added_at
                            ) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                            """,
                            (
                                comment_id,
                                issue_id,
                                comment.get("body", ""),
                                (
                                    comment.get("user", {}).get("login", "")
                                    if isinstance(comment.get("user"), dict)
                                    else ""
                                ),
                                created_at,
                                updated_at,
                            ),
                        )

            conn.commit()

        # Count issues after the update
        issues_after_count = len(self.get_issue_numbers(repo_name))

        return {
            "new_count": new_issues_count,
            "updated_count": updated_issues_count,
            "redundant_count": redundant_issues_count,
            "issues_before_count": issues_before_count,
            "issues_after_count": issues_after_count,
        }

    def get_issue_numbers(self, repo_name: str) -> List[int]:
        """
        Get the issue numbers for a repository.

        Args:
            repo_name: Name of the repository.

        Returns:
            List of issue numbers.
        """
        repo_id = self.repository_manager.get_repo_id(repo_name)
        if not repo_id:
            return []

        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT number FROM issues WHERE repo_id = ?", (repo_id,))

            return [row[0] for row in cursor.fetchall()]
    
    def get_issue_codes(self, repo_name: str) -> List[str]:
        """
        Get the issue numbers for a repository.

        Args:
            repo_name: Name of the repository.

        Returns:
            List of strings where each string is a combination of issue number and updated_at date
        """
        repo_id = self.repository_manager.get_repo_id(repo_name)
        if not repo_id:
            return []

        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT number, updated_at FROM issues WHERE repo_id = ?", (repo_id,))

            return [str(row[0])+ '_' + str(row[1]) for row in cursor.fetchall()]

    def get_latest_api_call_timestamp(self, repo_name: str) -> Optional[str]:
        """
        Get the timestamp of the latest API call for a repository.

        Args:
            repo_name: Name of the repository.

        Returns:
            Timestamp of the latest API call, or None if there are no API calls.
        """
        repo_id = self.repository_manager.get_repo_id(repo_name)
        if not repo_id:
            return None

        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT timestamp FROM api_logs WHERE repo_id = ? ORDER BY timestamp DESC LIMIT 1",
                (repo_id,),
            )

            result = cursor.fetchone()
            if result:
                return result[0]

            return None

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
        repo_id = self.repository_manager.get_repo_id(repo_name)
        if not repo_id:
            return []

        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM issues WHERE repo_id = ?"
            params = [repo_id]

            if state:
                query += " AND state = ?"
                params.append(state)

            query += " ORDER BY updated_at DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)

            issues = []
            for row in cursor.fetchall():
                issue = dict(row)

                # Get labels
                cursor.execute(
                    """
                    SELECT l.name
                    FROM issue_labels il
                    JOIN labels l ON il.label_id = l.id
                    WHERE il.issue_id = ?
                    """,
                    (issue["id"],),
                )

                issue["labels"] = [row[0] for row in cursor.fetchall()]

                # Get comments
                cursor.execute(
                    "SELECT * FROM comments WHERE issue_id = ? ORDER BY created_at", (issue["id"],)
                )

                issue["comments"] = [dict(row) for row in cursor.fetchall()]

                issues.append(issue)

            return issues

    def get_issue_count(self, repo_name: str, state: Optional[str] = None) -> int:
        """
        Get the number of issues for a repository.

        Args:
            repo_name: Name of the repository.
            state: State of issues to count ('open', 'closed', or None for all).

        Returns:
            Number of issues.
        """
        repo_id = self.repository_manager.get_repo_id(repo_name)
        if not repo_id:
            return 0

        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT COUNT(*) FROM issues WHERE repo_id = ?"
            params = [repo_id]

            if state:
                query += " AND state = ?"
                params.append(state)

            cursor.execute(query, params)

            return cursor.fetchone()[0]

    def get_comment_count(self, repo_name: str) -> int:
        """
        Get the number of comments for a repository.

        Args:
            repo_name: Name of the repository.

        Returns:
            Number of comments.
        """
        repo_id = self.repository_manager.get_repo_id(repo_name)
        if not repo_id:
            return 0

        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM comments c
                JOIN issues i ON c.issue_id = i.id
                WHERE i.repo_id = ?
                """,
                (repo_id,),
            )

            return cursor.fetchone()[0]

    def get_issues_with_most_comments(self, repo_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get issues with the most comments.

        Args:
            repo_name: Name of the repository.
            limit: Maximum number of issues to get.

        Returns:
            List of issues with comment counts.
        """
        repo_id = self.repository_manager.get_repo_id(repo_name)
        if not repo_id:
            return []

        with self.connection_manager.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT i.id, i.number, i.title, i.state, COUNT(c.id) as comment_count
                FROM issues i
                LEFT JOIN comments c ON i.id = c.issue_id
                WHERE i.repo_id = ?
                GROUP BY i.id
                ORDER BY comment_count DESC
                LIMIT ?
                """,
                (repo_id, limit),
            )

            return [dict(row) for row in cursor.fetchall()]
