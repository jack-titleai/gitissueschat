"""
Chainlit frontend for the RAG system.

This module provides a Chainlit-based frontend for the RAG system, allowing users
to interact with the system through a chat interface.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

import chainlit as cl
from chainlit.types import AskFileResponse
from chainlit.input_widget import Switch

from gitissueschat.rag.rag_orchestrator import RAGOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp.env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logger.info(f"Loaded environment variables from {dotenv_path}")
else:
    logger.warning(f"Environment file {dotenv_path} not found")

# Get project ID from environment
project_id = os.environ.get("GOOGLE_PROJECT_ID")
if not project_id:
    logger.error("GOOGLE_PROJECT_ID environment variable not set")
    raise ValueError("GOOGLE_PROJECT_ID environment variable not set")

# Get credentials from environment
credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if not credentials_path:
    logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

# Global variables
orchestrator = None
show_chunks = True  # Default to showing chunks

@cl.on_settings_update
async def on_settings_update(settings):
    """
    Handle settings updates.
    
    Args:
        settings: Updated settings.
    """
    global show_chunks
    show_chunks = settings.get("show_chunks", True)
    logger.info(f"Updated settings: show_chunks={show_chunks}")

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chat session.
    """
    global orchestrator, show_chunks
    
    # Send a welcome message
    await cl.Message(
        content="Welcome to the GitHub Issues RAG System! Ask me anything about GitHub issues."
    ).send()
    
    # Initialize settings directly in the sidebar
    sidebar_elements = [
        cl.Text(
            name="settings_header",
            content="## Settings",
            display="inline"
        ),
        cl.Text(
            name="toggle_description",
            content="Toggle to show/hide retrieved chunks in the sidebar:",
            display="inline"
        )
    ]
    
    # Add the toggle switch to the sidebar
    await cl.ElementSidebar.set_elements(sidebar_elements)
    await cl.ElementSidebar.set_title("RAG System Controls")
    
    # Also add the setting to the chat settings
    await cl.ChatSettings(
        [
            Switch(id="show_chunks", label="Show Retrieved Chunks", initial=show_chunks),
        ]
    ).send()
    
    # Initialize the RAG orchestrator
    try:
        orchestrator = RAGOrchestrator(
            db_path="./chroma_db_sample",
            collection_name="github_issues",
            project_id=project_id,
            api_key=None,  # We're using service account credentials
            credentials_path=credentials_path,
            top_k=10,
            relevance_threshold=0.5,
            model_name="gemini-2.0-flash-001",
            temperature=0.2
        )
        logger.info("Initialized RAG orchestrator")
    except Exception as e:
        logger.error(f"Error initializing RAG orchestrator: {e}")
        await cl.Message(
            content=f"Error initializing RAG system: {str(e)}"
        ).send()
        raise

@cl.on_message
async def on_message(message: cl.Message):
    """
    Process user messages.
    
    Args:
        message: User message.
    """
    global orchestrator, show_chunks
    
    if orchestrator is None:
        await cl.Message(
            content="RAG system not initialized. Please refresh the page."
        ).send()
        return
    
    # Get the user query
    query = message.content
    
    # Create a message with thinking indicator
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Retrieve relevant chunks
        chunks = orchestrator.retriever.retrieve(query)
        
        # If show_chunks is enabled, display the chunks in the sidebar
        if show_chunks and chunks:
            sidebar_elements = [
                cl.Text(
                    name="retrieved_chunks_header",
                    content="## Retrieved Chunks",
                    display="inline"
                )
            ]
            
            # Create elements for each chunk
            for i, chunk in enumerate(chunks):
                chunk_content = f"### Chunk {i+1}\n"
                chunk_content += f"**Similarity Score:** {chunk['similarity']:.4f}\n\n"
                chunk_content += f"**Metadata:**\n```json\n{json.dumps(chunk['metadata'], indent=2)}\n```\n\n"
                chunk_content += f"**Content:**\n```\n{chunk['content'][:500]}{'...' if len(chunk['content']) > 500 else ''}\n```"
                
                sidebar_elements.append(
                    cl.Text(
                        name=f"chunk_{i+1}",
                        content=chunk_content,
                        display="inline"
                    )
                )
            
            # Add a toggle switch at the top of the sidebar
            sidebar_elements.insert(1, cl.Text(
                name="toggle_description",
                content="Toggle to show/hide retrieved chunks:",
                display="inline"
            ))
            
            # Update the sidebar with the chunks
            await cl.ElementSidebar.set_elements(sidebar_elements)
            await cl.ElementSidebar.set_title("Retrieved Chunks")
        elif not show_chunks:
            # If show_chunks is disabled, clear the sidebar
            await cl.ElementSidebar.set_elements([
                cl.Text(
                    name="chunks_hidden",
                    content="## Retrieved Chunks\n\nChunks are currently hidden. Enable 'Show Retrieved Chunks' in settings to view them.",
                    display="inline"
                )
            ])
            await cl.ElementSidebar.set_title("RAG System Controls")
        
        # Generate response
        response = orchestrator.generator.generate(query, chunks)
        
        # Update the message with the response
        msg.content = response
        await msg.update()
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        msg.content = f"Error processing your query: {str(e)}"
        await msg.update()

if __name__ == "__main__":
    # This is used when running locally only
    logger.info("Running Chainlit app locally")
