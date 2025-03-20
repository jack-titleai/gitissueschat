"""
GitHub Issues Embedding Module

This module provides functionality for chunking and embedding GitHub issues.
"""

from gitissueschat.embed.llamaindex_chunker import LlamaIndexChunker
from gitissueschat.embed.google_vertex_embedding_function import GoogleVertexEmbeddingFunctionCustom
from gitissueschat.embed.chroma_database import ChunksDatabase

__all__ = ["LlamaIndexChunker", "GoogleVertexEmbeddingFunctionCustom", "ChunksDatabase"]
