"""
RAG System for GitHub Issues.

This package provides a RAG (Retrieval-Augmented Generation) system for GitHub issues,
using ChromaDB for retrieval and Gemini Flash 2.0 for generation.
"""

from gitissueschat.rag.chroma_retriever import ChromaRetriever
from gitissueschat.rag.gemini_generator import GeminiGenerator
from gitissueschat.rag.rag_orchestrator import RAGOrchestrator

__all__ = [
    "ChromaRetriever",
    "GeminiGenerator",
    "RAGOrchestrator"
]
