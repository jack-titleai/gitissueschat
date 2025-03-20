"""
LlamaIndex-based GitHub Issues Chunker

This module provides functionality to chunk GitHub issues and comments for embedding
using LlamaIndex's SentenceSplitter.
"""

from typing import Dict, List, Any
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document


class LlamaIndexChunker:
    """
    A class to chunk GitHub issues and comments using LlamaIndex's SentenceSplitter.
    """
    
    def __init__(
            self,
            chunk_size: int = 500,
            chunk_overlap: int = 100,
            issue_context_chars: int = 100
        ):
        """
        Initialize the LlamaIndex issue chunker.
        
        Args:
            chunk_size: Target size of each chunk in tokens.
            chunk_overlap: Number of tokens to overlap between chunks.
            issue_context_chars: Number of characters from the issue to include in comment chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.issue_context_chars = issue_context_chars
        
        # Initialize the LlamaIndex sentence splitter
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",  # Respect paragraph breaks
        )
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks using LlamaIndex's SentenceSplitter.
        
        Args:
            text: The text to split.
            
        Returns:
            A list of text chunks.
        """
        if not text.strip():
            return []
            
        # Create a document from the text
        document = Document(text=text)
        
        # Split the document into nodes
        nodes = self.splitter.get_nodes_from_documents([document])
        
        # Extract text from nodes
        return [node.text for node in nodes]
    
    def chunk_issue(self, issue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk an issue into smaller pieces suitable for embedding.

        Args:
            issue: A dictionary containing issue data.

        Returns:
            A list of dictionaries, each representing a chunk of the issue.
        """
        # Extract issue data
        issue_id = issue.get("id")
        issue_number = issue.get("number")
        issue_title = issue.get("title", "")
        issue_body = issue.get("body", "")
        repo_name = issue.get("repository", "")
        
        # Create issue text with metadata
        issue_text = f"Issue #{issue_number}: {issue_title}\n\n{issue_body}"
        
        # Chunk the issue text
        text_chunks = self._split_text_into_chunks(issue_text)
        
        # Create chunks with metadata
        issue_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "id": f"{issue_id}_issue_{i}",
                "issue_id": issue_id,
                "issue_number": issue_number,
                "repository": repo_name,
                "type": "issue",
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "text": chunk_text,
                "metadata": {
                    "title": issue_title,
                    "url": issue.get("html_url", ""),
                    "state": issue.get("state", ""),
                    "created_at": issue.get("created_at", ""),
                    "updated_at": issue.get("updated_at", ""),
                    "author": issue.get("user", {}).get("login", ""),
                }
            }
            issue_chunks.append(chunk)
        
        return issue_chunks

    def chunk_comments(self, issue: Dict[str, Any], comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk comments into smaller pieces suitable for embedding.
        
        Args:
            issue: The parent issue dictionary.
            comments: A list of comment dictionaries.
            
        Returns:
            A list of dictionaries, each representing a chunk of a comment.
        """
        if not comments:
            return []
            
        # Extract issue context
        issue_id = issue.get("id")
        issue_number = issue.get("number")
        issue_title = issue.get("title", "")
        issue_body = issue.get("body", "")
        repo_name = issue.get("repository", "")
        
        # Create issue context prefix (first N characters)
        issue_context = f"Issue #{issue_number}: {issue_title}\n\n"
        if issue_body:
            # Add the first N characters of the issue body
            context_body = issue_body[:self.issue_context_chars]
            # Add ellipsis if the body was truncated
            if len(issue_body) > self.issue_context_chars:
                context_body += "..."
            issue_context += context_body + "\n\n"
        
        comment_chunks = []
        
        for comment in comments:
            comment_id = comment.get("id")
            comment_body = comment.get("body", "")
            comment_author = comment.get("user", {}).get("login", "")
            
            # Prepend issue context to comment
            comment_text = f"{issue_context}Comment by {comment_author}:\n{comment_body}"
            
            # Chunk the comment text
            text_chunks = self._split_text_into_chunks(comment_text)
            
            # Create chunks with metadata
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    "id": f"{comment_id}_comment_{i}",
                    "comment_id": comment_id,
                    "issue_id": issue_id,
                    "issue_number": issue_number,
                    "repository": repo_name,
                    "type": "comment",
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "text": chunk_text,
                    "metadata": {
                        "issue_title": issue_title,
                        "issue_url": issue.get("html_url", ""),
                        "created_at": comment.get("created_at", ""),
                        "updated_at": comment.get("updated_at", ""),
                        "author": comment_author,
                    }
                }
                comment_chunks.append(chunk)
        
        return comment_chunks
    
    def process_issue_with_comments(self, issue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process an issue and its comments, chunking both appropriately.
        
        Args:
            issue: A dictionary containing issue data with comments.
            
        Returns:
            A list of all chunks from the issue and its comments.
        """
        # Extract comments from the issue
        comments = issue.pop("comments", [])
        
        # Chunk the issue
        issue_chunks = self.chunk_issue(issue)
        
        # Chunk the comments
        comment_chunks = self.chunk_comments(issue, comments)
        
        # Combine all chunks
        return issue_chunks + comment_chunks
