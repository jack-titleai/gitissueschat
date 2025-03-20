#!/usr/bin/env python3
"""
Test Update Repository Script

This script tests the update_repository.py script by mocking the necessary components.
"""

import os
import unittest
from unittest.mock import patch, MagicMock, call
import tempfile
import sqlite3
from datetime import datetime, timedelta

from gitissueschat.utils.update_repository import get_most_recent_api_call, delete_issue_chunks_from_chromadb, update_repository


class TestUpdateRepository(unittest.TestCase):
    """Test cases for the update_repository.py script."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary SQLite database
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        
        # Create the necessary tables
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        
        # Create repositories table
        cursor.execute('''
            CREATE TABLE repositories (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create issues table
        cursor.execute('''
            CREATE TABLE issues (
                id INTEGER PRIMARY KEY,
                repo_id INTEGER,
                number INTEGER,
                title TEXT,
                body TEXT,
                state TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                FOREIGN KEY (repo_id) REFERENCES repositories (id)
            )
        ''')
        
        # Create comments table
        cursor.execute('''
            CREATE TABLE comments (
                id INTEGER PRIMARY KEY,
                issue_id INTEGER,
                body TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                FOREIGN KEY (issue_id) REFERENCES issues (id)
            )
        ''')
        
        # Create api_logs table
        cursor.execute('''
            CREATE TABLE api_logs (
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
                FOREIGN KEY (repo_id) REFERENCES repositories (id)
            )
        ''')
        
        # Insert test data
        cursor.execute("INSERT INTO repositories (id, name) VALUES (1, 'test/repo')")
        
        # Insert an API log
        timestamp = datetime.now() - timedelta(days=1)
        cursor.execute('''
            INSERT INTO api_logs (
                repo_id, timestamp, new_issues_count, updated_issues_count, 
                redundant_issues_count, issues_before_count, issues_after_count,
                api_rate_limit_remaining, api_rate_limit_total, execution_time_seconds
            ) VALUES (?, ?, 10, 0, 0, 0, 10, 4990, 5000, 1.5)
        ''', (1, timestamp))
        
        # Insert some issues
        cursor.execute('''
            INSERT INTO issues (
                id, repo_id, number, title, body, state, created_at, updated_at
            ) VALUES (101, 1, 1, 'Test Issue 1', 'This is a test issue', 'open', ?, ?)
        ''', (timestamp, timestamp))
        
        cursor.execute('''
            INSERT INTO issues (
                id, repo_id, number, title, body, state, created_at, updated_at
            ) VALUES (102, 1, 2, 'Test Issue 2', 'This is another test issue', 'closed', ?, ?)
        ''', (timestamp, timestamp))
        
        # Insert some comments
        cursor.execute('''
            INSERT INTO comments (
                id, issue_id, body, created_at, updated_at
            ) VALUES (201, 101, 'This is a test comment', ?, ?)
        ''', (timestamp, timestamp))
        
        conn.commit()
        conn.close()

    def tearDown(self):
        """Tear down test fixtures."""
        os.close(self.temp_db_fd)
        os.unlink(self.temp_db_path)

    @patch('gitissueschat.utils.update_repository.SQLiteIssueStorage')
    def test_get_most_recent_api_call(self, mock_storage):
        """Test get_most_recent_api_call function."""
        # Mock the get_api_logs method
        mock_storage.return_value.get_api_logs.return_value = [{'timestamp': '2023-01-01 12:00:00'}]
        
        # Call the function
        result = get_most_recent_api_call(mock_storage.return_value, 'test/repo')
        
        # Check the result
        self.assertEqual(result, {'timestamp': '2023-01-01 12:00:00'})
        mock_storage.return_value.get_api_logs.assert_called_once_with('test/repo', limit=1)
        
        # Test with no logs
        mock_storage.return_value.get_api_logs.return_value = []
        result = get_most_recent_api_call(mock_storage.return_value, 'test/repo')
        self.assertIsNone(result)

    @patch('gitissueschat.utils.update_repository.ChunksDatabase')
    def test_delete_issue_chunks_from_chromadb(self, mock_chroma_db):
        """Test delete_issue_chunks_from_chromadb function."""
        # Mock the collection.get method
        mock_collection = MagicMock()
        mock_collection.get.return_value = {'ids': ['id1', 'id2', 'id3']}
        mock_chroma_db.collection = mock_collection
        
        # Call the function
        delete_issue_chunks_from_chromadb(mock_chroma_db, 101)
        
        # Check that the correct methods were called
        mock_collection.get.assert_called_once_with(where={"issue_id": 101})
        mock_collection.delete.assert_called_once_with(ids=['id1', 'id2', 'id3'])
        
        # Test with no chunks
        mock_collection.get.return_value = {'ids': []}
        delete_issue_chunks_from_chromadb(mock_chroma_db, 102)
        mock_collection.get.assert_called_with(where={"issue_id": 102})
        # delete should not be called again
        mock_collection.delete.assert_called_once()

    @patch('gitissueschat.utils.update_repository.SQLiteIssueStorage')
    @patch('gitissueschat.utils.update_repository.GitHubIssuesFetcher')
    @patch('gitissueschat.utils.update_repository.ChunksDatabase')
    @patch('gitissueschat.utils.update_repository.LlamaIndexChunker')
    @patch('gitissueschat.utils.update_repository.process_issue_with_comments')
    @patch('gitissueschat.utils.update_repository.sqlite3.connect')
    def test_update_repository(self, mock_connect, mock_process_issue, mock_chunker, 
                               mock_chroma_db, mock_fetcher, mock_storage):
        """Test update_repository function."""
        # Mock the get_most_recent_api_call
        mock_storage.return_value.get_api_logs.return_value = [{'timestamp': '2023-01-01 12:00:00'}]
        mock_storage.return_value.get_issue_numbers.return_value = [1, 2]
        mock_storage.return_value.get_issue_count.return_value = 2
        
        # Mock the fetcher
        mock_fetcher.return_value.get_rate_limit_info.return_value = {
            'remaining': 4990, 'limit': 5000, 'reset_time': '2023-01-01 13:00:00'
        }
        
        # Mock the cursor and connection
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        
        # Mock the cursor.fetchone to return an issue
        mock_cursor.fetchone.return_value = {
            'id': 101, 'number': 1, 'title': 'Test Issue 1', 
            'body': 'This is a test issue', 'state': 'open'
        }
        
        # Mock process_issue_with_comments to return chunks
        mock_process_issue.return_value = [{'id': 'chunk1', 'text': 'Test chunk'}]
        
        # Call the function
        result = update_repository(
            repo_name='test/repo',
            github_token='token',
            sqlite_db_path=self.temp_db_path,
            chroma_db_path='/tmp/chroma',
            collection_name='github_issues',
            chunk_size=250,
            chunk_overlap=50
        )
        
        # Check the result
        self.assertIn('total_issues', result)
        self.assertIn('execution_time', result)
        
        # Verify that the correct methods were called
        mock_storage.return_value.get_api_logs.assert_called_once_with('test/repo', limit=1)
        mock_storage.return_value.get_issue_numbers.assert_called_once_with('test/repo')
        mock_fetcher.return_value.fetch_issues.assert_called_once()
        mock_storage.return_value.log_api_call.assert_called_once()


if __name__ == '__main__':
    unittest.main()
