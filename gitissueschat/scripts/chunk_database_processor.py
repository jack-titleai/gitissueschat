"""
Process chunks to ChromaDB

This module provides functionality for processing chunks from a JSONL file and storing them in ChromaDB.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from tqdm import tqdm

from gitissueschat.embed.chroma_database import ChunksDatabase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_chunks_to_db(
    chunks_file: str,
    db_path: str,
    collection_name: str = "github_issues",
    project_id: Optional[str] = None,
    api_key: Optional[str] = None,
    credentials: Optional[str] = None,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Process chunks from a JSONL file and store them in ChromaDB.
    
    Args:
        chunks_file: Path to the JSONL file with chunks.
        db_path: Path to the ChromaDB database.
        collection_name: Name of the collection to use.
        project_id: Google Cloud project ID.
        api_key: Google API key.
        credentials: Path to the Google Cloud service account key file.
        batch_size: Number of chunks to process at once.
        
    Returns:
        Statistics about the database.
    """
    # Load chunks from JSONL file
    chunks = []
    with open(chunks_file, 'r') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
    
    # Initialize the database
    db = ChunksDatabase(
        db_path=db_path,
        collection_name=collection_name,
        project_id=project_id,
        api_key=api_key,
        credentials=credentials
    )
    
    # Process chunks in batches
    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing chunks"):
        batch = chunks[i:i + batch_size]
        
        # Add chunks to the database
        db.add_chunks(chunks=batch)
    
    # Return database statistics
    return db.get_stats()
