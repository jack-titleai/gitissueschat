"""
Database Path Manager

This module provides utility functions for managing database paths.
"""

import os
from typing import Optional


def get_sqlite_db_path(repo_name: str, base_dir: Optional[str] = None) -> str:
    """
    Get the path for a SQLite database based on the repository name.
    
    Args:
        repo_name: Name of the repository.
        base_dir: Optional base directory. If None, defaults to './data/sqlite_dbs'.
        
    Returns:
        Path to the SQLite database.
    """
    # Use default base directory if not provided
    if base_dir is None:
        base_dir = "./data/sqlite_dbs"
    
    # Create the directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Normalize repository name by replacing '/' with '_'
    normalized_repo_name = repo_name.replace("/", "_")
    
    # Construct the database path
    db_path = os.path.join(base_dir, f"{normalized_repo_name}.db")
    
    return db_path


def get_chroma_db_path(repo_name: str, base_dir: Optional[str] = None) -> str:
    """
    Get the path for a ChromaDB database based on the repository name.
    
    Args:
        repo_name: Name of the repository.
        base_dir: Optional base directory. If None, defaults to './data/chroma_dbs'.
        
    Returns:
        Path to the ChromaDB database.
    """
    # Use default base directory if not provided
    if base_dir is None:
        base_dir = "./data/chroma_dbs"
    
    # Create the directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Normalize repository name by replacing '/' with '_'
    normalized_repo_name = repo_name.replace("/", "_")
    
    # Construct the database path
    db_path = os.path.join(base_dir, normalized_repo_name)
    
    # Create the repository-specific directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)
    
    return db_path
