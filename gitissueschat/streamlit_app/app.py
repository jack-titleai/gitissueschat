"""
Streamlit frontend for the RAG system.

This module provides a Streamlit-based frontend for the RAG system, allowing users
to interact with the system through a chat interface with a customizable sidebar.
"""

import os
import json
import logging
import time
import subprocess
import threading
import streamlit as st
import glob
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from datetime import datetime

from gitissueschat.rag.rag_orchestrator import RAGOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logger.info(f"Loaded environment variables from {dotenv_path}")
else:
    # Try alternate path
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        logger.info(f"Loaded environment variables from {dotenv_path}")
    else:
        logger.warning(f"Environment file not found")

# Get project ID from environment
project_id = os.environ.get("GOOGLE_PROJECT_ID")
if not project_id:
    logger.error("GOOGLE_PROJECT_ID environment variable not set")
    st.error("GOOGLE_PROJECT_ID environment variable not set")

# Get credentials from environment
credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if not credentials_path:
    logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
    st.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")

# Create temp directory for process results
temp_dir = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(temp_dir, exist_ok=True)

def get_available_databases():
    """Get available databases from the ./data/chroma_dbs directory."""
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma_dbs")
    if not os.path.exists(db_path):
        logger.warning(f"Database directory {db_path} not found")
        return []
    
    # Get all subdirectories in the chroma_dbs directory
    databases = [os.path.basename(db) for db in glob.glob(os.path.join(db_path, "*")) if os.path.isdir(db)]
    logger.info(f"Found {len(databases)} databases: {databases}")
    return databases

def normalize_repo_name(repo_input: str) -> str:
    """Normalize a repository name from various formats to owner_repo format.
    
    Args:
        repo_input: The repository name in various formats (URL, owner/repo, etc.)
        
    Returns:
        The normalized repository name in owner_repo format.
    """
    # Handle GitHub URLs
    if "github.com" in repo_input:
        # Extract owner/repo from URL
        parts = repo_input.strip("/").split("/")
        if len(parts) >= 2:
            # Get the last two parts (owner and repo)
            owner = parts[-2]
            repo = parts[-1]
            return f"{owner}_{repo}"
    
    # Handle owner/repo format
    elif "/" in repo_input:
        return repo_input.replace("/", "_")
    
    # Return as is if already in the right format or unknown format
    return repo_input

def repository_exists(repo_input: str) -> bool:
    """Check if a repository already exists in the database.
    
    Args:
        repo_input: The repository name in various formats (URL, owner/repo, etc.)
        
    Returns:
        True if the repository exists, False otherwise.
    """
    # Normalize the repository name
    normalized_name = normalize_repo_name(repo_input)
    
    # Get available databases
    available_dbs = get_available_databases()
    
    # Check if the normalized name exists in available databases
    return normalized_name in available_dbs

def initialize_rag_orchestrator():
    """Initialize the RAG orchestrator."""
    try:
        # Use the selected database if available, otherwise use the default
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             "data", "chroma_dbs", st.session_state.selected_db) if st.session_state.selected_db else "./chroma_fastai"
        
        # Log the full path for debugging
        logger.info(f"Initializing RAG orchestrator with database path: {os.path.abspath(db_path)}")
        
        # Use a consistent collection name
        collection_name = "github_issues"
        
        orchestrator = RAGOrchestrator(
            db_path=db_path,
            collection_name=collection_name,
            project_id=project_id,
            api_key=None,  # We're using service account credentials
            credentials_path=credentials_path,
            top_k=10,
            relevance_threshold=0.5,
            model_name="gemini-2.0-flash-001",
            temperature=0.2
        )
        logger.info(f"Initialized RAG orchestrator with database {db_path}")
        return orchestrator
    except Exception as e:
        logger.error(f"Error initializing RAG orchestrator: {e}")
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

def format_chunk(chunk, index):
    """Format a chunk for display."""
    # Extract metadata
    metadata = chunk.get("metadata", {})
    
    # Add content
    content_info = chunk.get("content", chunk.get("text", "No content available"))
    
    return metadata, content_info

def display_chunks_sidebar():
    """Display chunks in the sidebar."""
    if st.session_state.show_chunks and st.session_state.current_chunks:
        with st.sidebar:
            st.subheader(f"Retrieved Chunks for: '{st.session_state.last_query}'")
            
            # Display each chunk in an expander
            for i, chunk in enumerate(st.session_state.current_chunks):
                # Get similarity score if available
                similarity = chunk.get("similarity", 0.0)
                
                # Create expander with similarity score in title
                with st.expander(f"Chunk {i+1} [similarity: {similarity:.4f}]"):
                    metadata, content_info = format_chunk(chunk, i)
                    
                    # Display metadata and content as separate blocks
                    st.markdown("### Metadata")
                    st.json(metadata, expanded=False)
                    
                    st.markdown("### Content")
                    st.markdown(content_info)

def update_database(db_name: str) -> subprocess.Popen:
    """Update a database by running the update_repository.py script.
    
    Args:
        db_name: The name of the database to update (e.g., 'owner_repo').
        
    Returns:
        The process object.
    """
    try:
        # Extract repository name from database name
        repo_name = db_name.replace("_", "/", 1)
        
        # Get GitHub token from environment
        github_token = os.environ.get("GITHUB_TOKEN", "")
        
        # Construct the command
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "update_repository.py")
        cmd = ["python", script_path, repo_name]
        
        if github_token:
            cmd.extend(["--token", github_token])
        
        # Log the command for debugging
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command as a separate process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )
        
        return process
    except Exception as e:
        logger.error(f"Error running update script: {str(e)}")
        st.error(f"Error running update script: {str(e)}")
        return None

def process_new_repository(repo_name: str) -> subprocess.Popen:
    """Process a new GitHub repository.
    
    Args:
        repo_name: The name of the repository to process.
        
    Returns:
        The process object.
    """
    try:
        # Get GitHub token from environment
        github_token = os.environ.get("GITHUB_TOKEN", "")
        
        # Construct the command
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "process_repository.py")
        cmd = ["python", script_path, repo_name]
        
        if github_token:
            cmd.extend(["--token", github_token])
        
        # Log the command for debugging
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the command as a separate process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )
        
        return process
    except Exception as e:
        logger.error(f"Error running process script: {str(e)}")
        st.error(f"Error running process script: {str(e)}")
        return None

def monitor_process_thread(process, process_type):
    """Monitor a process in a separate thread.
    
    Args:
        process: The process to monitor.
        process_type: The type of process (e.g., "update", "add_repo").
    """
    # Create temp directory for process results
    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize result file path
    result_file = os.path.join(temp_dir, f"{process_type}_result.json")
    
    # Initialize output list
    output_lines = []
    
    try:
        # Read output line by line
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            
            # Add line to output
            output_lines.append(line.strip())
            
            # Update session state if this is running in the main thread
            try:
                if process_type == "update":
                    if not hasattr(st.session_state, "update_output"):
                        st.session_state.update_output = []
                    st.session_state.update_output.append(line.strip())
                elif process_type == "add_repo":
                    if not hasattr(st.session_state, "add_repo_output"):
                        st.session_state.add_repo_output = []
                    st.session_state.add_repo_output.append(line.strip())
            except Exception as e:
                logger.error(f"Error updating session state: {str(e)}")
        
        # Read any remaining output
        remaining_output, _ = process.communicate()
        if remaining_output:
            for line in remaining_output.splitlines():
                line_str = line.strip() if isinstance(line, str) else line.decode().strip()
                output_lines.append(line_str)
                
                # Update session state if this is running in the main thread
                try:
                    if process_type == "update":
                        if not hasattr(st.session_state, "update_output"):
                            st.session_state.update_output = []
                        st.session_state.update_output.append(line_str)
                    elif process_type == "add_repo":
                        if not hasattr(st.session_state, "add_repo_output"):
                            st.session_state.add_repo_output = []
                        st.session_state.add_repo_output.append(line_str)
                except Exception as e:
                    logger.error(f"Error updating session state: {str(e)}")
        
        # Get return code
        returncode = process.poll()
        
        # Write result to file
        result = {
            "completed": True,
            "returncode": returncode,
            "output": output_lines,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(result_file, "w") as f:
            json.dump(result, f)
        
    except Exception as e:
        logger.error(f"Error monitoring process: {str(e)}")
        
        # Check if process was terminated
        if process.poll() is not None and process.returncode in [-15, -9]:  # SIGTERM or SIGKILL
            # Process was terminated or killed
            result = {
                "completed": True,
                "returncode": process.returncode,
                "output": output_lines + [f"Process was cancelled: {str(e)}"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            # Other error
            result = {
                "completed": True,
                "returncode": -1,
                "output": output_lines + [f"Error monitoring process: {str(e)}"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        with open(result_file, "w") as f:
            json.dump(result, f)

def check_background_processes():
    """Check the status of background processes and update the UI accordingly."""
    process_completed = False
    
    # Check add repository process
    if st.session_state.add_repo_process is not None:
        # Check if process is still running
        if st.session_state.add_repo_process.poll() is None:
            # Process is still running - just collect output without blocking
            try:
                # Read stdout without blocking
                while True:
                    line = st.session_state.add_repo_process.stdout.readline()
                    if not line:
                        break
                    st.session_state.add_repo_output.append(line.strip())
                    if len(st.session_state.add_repo_output) > 100:
                        st.session_state.add_repo_output.pop(0)
                
                # Read stderr without blocking
                while True:
                    line = st.session_state.add_repo_process.stderr.readline()
                    if not line:
                        break
                    st.session_state.add_repo_output.append(f"ERROR: {line.strip()}")
                    if len(st.session_state.add_repo_output) > 100:
                        st.session_state.add_repo_output.pop(0)
            except:
                pass
        else:
            # Process completed - collect final output
            try:
                stdout, stderr = st.session_state.add_repo_process.communicate(timeout=0.1)
                
                # Add final output
                if stdout:
                    for line in stdout.split('\n'):
                        if line.strip():
                            st.session_state.add_repo_output.append(line.strip())
                
                if stderr:
                    for line in stderr.split('\n'):
                        if line.strip():
                            st.session_state.add_repo_output.append(f"ERROR: {line.strip()}")
                
                # Set timestamp
                st.session_state.add_repo_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                if st.session_state.add_repo_process.returncode == 0:
                    st.session_state.add_repo_status = "success"
                    process_completed = True
                    
                    # Extract the repository name for the database name
                    if "/" in st.session_state.add_repo_input:
                        repo_input = st.session_state.add_repo_input
                        # Extract the repo name from the URL or owner/repo format
                        if "github.com" in repo_input:
                            # It's a URL, extract owner/repo
                            parts = repo_input.split("/")
                            if len(parts) >= 2:
                                repo_name = f"{parts[-2]}_{parts[-1]}"
                        else:
                            # It's already in owner/repo format
                            repo_name = repo_input.replace("/", "_")
                        
                        # Refresh the database list
                        available_dbs = get_available_databases()
                        
                        # If the newly added database is the currently selected one, reinitialize
                        if repo_name == st.session_state.selected_db:
                            force_refresh_orchestrator()
                else:
                    st.session_state.add_repo_status = "error"
                    process_completed = True
                
                # Clear the process
                st.session_state.add_repo_process = None
            except subprocess.TimeoutExpired:
                # This shouldn't happen since we're checking poll() first
                pass
    
    # Similar logic for update process (if needed)
    
    # Return whether any process completed this check
    return process_completed

def check_process_completion():
    """Check if any background processes have completed by looking for result files."""
    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    if not os.path.exists(temp_dir):
        return False
    
    process_completed = False
    
    # Check for add_repo result file
    add_repo_result_file = os.path.join(temp_dir, "add_repo_result.json")
    if os.path.exists(add_repo_result_file):
        try:
            with open(add_repo_result_file, 'r') as f:
                result = json.load(f)
            
            if result.get("completed", False):
                # Update session state
                if result["returncode"] in [-15, -9]:  # SIGTERM or SIGKILL
                    st.session_state.add_repo_status = "cancelled"
                else:
                    st.session_state.add_repo_status = "success" if result["returncode"] == 0 else "error"
                
                st.session_state.add_repo_output = result["output"]
                st.session_state.add_repo_timestamp = result["timestamp"]
                st.session_state.add_repo_process = None
                
                # If successful, check if we need to refresh the orchestrator
                if st.session_state.add_repo_status == "success" and "/" in st.session_state.add_repo_input:
                    repo_input = st.session_state.add_repo_input
                    # Extract the repo name from the URL or owner/repo format
                    if "github.com" in repo_input:
                        # It's a URL, extract owner/repo
                        parts = repo_input.split("/")
                        if len(parts) >= 2:
                            repo_name = f"{parts[-2]}_{parts[-1]}"
                    else:
                        # It's already in owner/repo format
                        repo_name = repo_input.replace("/", "_")
                    
                    # If the newly added database is the currently selected one, reinitialize
                    if repo_name == st.session_state.selected_db:
                        force_refresh_orchestrator()
                
                # Remove the result file
                os.remove(add_repo_result_file)
                process_completed = True
        except Exception as e:
            logger.error(f"Error reading result file: {str(e)}")
            # Remove the result file if it's corrupted
            try:
                os.remove(add_repo_result_file)
            except:
                pass
    
    # Check for update result file
    update_result_file = os.path.join(temp_dir, "update_result.json")
    if os.path.exists(update_result_file):
        try:
            with open(update_result_file, 'r') as f:
                result = json.load(f)
            
            if result.get("completed", False):
                # Update session state
                if result["returncode"] in [-15, -9]:  # SIGTERM or SIGKILL
                    st.session_state.update_status = "cancelled"
                else:
                    st.session_state.update_status = "success" if result["returncode"] == 0 else "error"
                
                st.session_state.update_output = result["output"]
                st.session_state.update_timestamp = result["timestamp"]
                st.session_state.update_process = None
                
                # If successful, refresh the orchestrator if this is the currently selected database
                if st.session_state.update_status == "success":
                    # If the updated database is the currently selected one, reinitialize
                    if st.session_state.update_db == st.session_state.selected_db:
                        force_refresh_orchestrator()
                
                # Remove the result file
                os.remove(update_result_file)
                process_completed = True
        except Exception as e:
            logger.error(f"Error reading result file: {str(e)}")
            # Remove the result file if it's corrupted
            try:
                os.remove(update_result_file)
            except:
                pass
    
    return process_completed

def cancel_process(process):
    """Cancel a running process.
    
    Args:
        process: The process to cancel.
        
    Returns:
        True if the process was successfully cancelled, False otherwise.
    """
    if process is None:
        return False
    
    try:
        # Try to terminate the process
        process.terminate()
        
        # Wait a short time for it to terminate
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # If it doesn't terminate, kill it
            process.kill()
            process.wait()
        
        return True
    except Exception as e:
        logger.error(f"Error cancelling process: {str(e)}")
        return False

def force_refresh_orchestrator():
    """Force refresh the RAG orchestrator."""
    st.session_state.orchestrator = initialize_rag_orchestrator()
    logger.info("RAG orchestrator forcefully reinitialized.")

def main():
    """Main function to run the Streamlit app."""
    # Set page config
    st.set_page_config(
        page_title="GitHub Issues Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "current_chunks" not in st.session_state:
        st.session_state.current_chunks = []
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "timing_info" not in st.session_state:
        st.session_state.timing_info = {
            "retrieval_time": 0.0,
            "generation_time": 0.0,
            "total_time": 0.0
        }
    if "show_db_settings" not in st.session_state:
        st.session_state.show_db_settings = True
    if "selected_db" not in st.session_state:
        st.session_state.selected_db = None
    if "show_chunks" not in st.session_state:
        st.session_state.show_chunks = True
    if "show_timing" not in st.session_state:
        st.session_state.show_timing = False
    
    # Initialize add repository session state variables
    if "add_repo_process" not in st.session_state:
        st.session_state.add_repo_process = None
    if "add_repo_status" not in st.session_state:
        st.session_state.add_repo_status = None
    if "add_repo_output" not in st.session_state:
        st.session_state.add_repo_output = []
    if "add_repo_input" not in st.session_state:
        st.session_state.add_repo_input = ""
    if "add_repo_timestamp" not in st.session_state:
        st.session_state.add_repo_timestamp = None
    
    # Initialize update database session state variables
    if "update_process" not in st.session_state:
        st.session_state.update_process = None
    if "update_status" not in st.session_state:
        st.session_state.update_status = None
    if "update_output" not in st.session_state:
        st.session_state.update_output = []
    if "update_db" not in st.session_state:
        st.session_state.update_db = None
    if "update_timestamp" not in st.session_state:
        st.session_state.update_timestamp = None
    
    # Check if any background processes have completed
    process_completed = check_process_completion()
    if process_completed:
        st.rerun()
    
    # Initialize RAG orchestrator if not already initialized
    if st.session_state.orchestrator is None:
        # Get available databases
        databases = get_available_databases()
        if databases and not st.session_state.selected_db:
            st.session_state.selected_db = databases[0]
            
        st.session_state.orchestrator = initialize_rag_orchestrator()
    
    # Sidebar controls
    with st.sidebar:
        # Add a clear button at the top of the sidebar
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_chunks = []
            st.session_state.last_query = ""
            st.session_state.timing_info = {
                "retrieval_time": 0.0,
                "generation_time": 0.0,
                "total_time": 0.0
            }
            st.rerun()
        
        # Add a separator
        st.divider()
        
        # Always display settings at the top of the sidebar
        col1, col2 = st.columns([5, 1])
        with col1:
            st.title("Database Settings")
        with col2:
            # Align the toggle with the title using vertical spacing
            st.write("")  # Add some space
            show_db_settings = st.toggle(
                "",
                value=st.session_state.show_db_settings,
                help="Toggle to show or hide the database settings",
                key="db_settings_toggle",
                label_visibility="collapsed"
            )
            
            # Check if toggle state changed
            if show_db_settings != st.session_state.show_db_settings:
                st.session_state.show_db_settings = show_db_settings
                st.rerun()
        
        # Show database settings if enabled
        if st.session_state.show_db_settings:
            # Database selection
            st.subheader("Database Settings")
            
            # Get available databases
            available_dbs = get_available_databases()
            
            # Database selector
            selected_db = st.selectbox(
                "Select Database",
                options=available_dbs,
                index=available_dbs.index(st.session_state.selected_db) if st.session_state.selected_db in available_dbs else 0,
                key="db_selector"
            )
            
            # Update selected database if changed
            if selected_db != st.session_state.selected_db:
                st.session_state.selected_db = selected_db
                st.session_state.orchestrator = initialize_rag_orchestrator()
                st.session_state.messages = []
                st.rerun()
            
            # Display collection count
            if st.session_state.orchestrator:
                try:
                    collection = st.session_state.orchestrator.retriever.db.get_collection()
                    count = collection.count()
                    st.info(f"Database contains {count} documents.")
                except Exception as e:
                    st.error(f"Error getting collection count: {str(e)}")
            
            # Force refresh button
            if st.button("Force Refresh Database Connection"):
                force_refresh_orchestrator()
                st.success("Database connection refreshed!")
                st.rerun()
            
            # Update Database section
            with st.expander("Update Database"):
                # Get available databases for updating
                update_databases = get_available_databases()
                
                # Database selector for update
                update_db = st.selectbox(
                    "Select Database to Update",
                    options=update_databases,
                    key="update_db_selector"
                )
                
                # Update button
                update_clicked = st.button("Update", key="update_button")
                
                # Handle update button click
                if update_clicked:
                    # Reset previous status if not running
                    if st.session_state.update_status in ["cancelled", "error", "success"]:
                        st.session_state.update_status = None
                        st.session_state.update_output = []
                        st.session_state.update_process = None
                        st.rerun()
                    
                    if st.session_state.update_process is None or st.session_state.update_status != "running":
                        # Start the update process
                        process = update_database(update_db)
                        
                        if process:
                            st.session_state.update_process = process
                            st.session_state.update_status = "running"
                            st.session_state.update_output = []
                            st.session_state.update_db = update_db
                            st.session_state.update_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Start a thread to monitor the process
                            threading.Thread(target=monitor_process_thread, args=(process, "update")).start()
                            
                            # Force a rerun to show the cancel button immediately
                            st.rerun()
                
                # Handle process status
                if st.session_state.update_process is not None and st.session_state.update_status == "running":
                    # Display status (don't duplicate the info message if we just clicked the button)
                    if not update_clicked:
                        st.info(f"Updating database {st.session_state.update_db}... This may take a while. You can continue using the chat while this runs in the background. [{st.session_state.update_timestamp}]")
                    
                    # Show current output (last 10 lines)
                    if st.session_state.update_output:
                        st.text_area("Update Output (Last 10 Lines)", "\n".join(st.session_state.update_output[-10:]), height=150)
                    
                    # Cancel button - moved below the status message
                    cancel_clicked = st.button("Cancel", key="cancel_update_button")
                    
                    if cancel_clicked:
                        if cancel_process(st.session_state.update_process):
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.warning(f"Database update cancelled. [{timestamp}]")
                            
                            # Show logs
                            st.text_area("Update Output", "\n".join(st.session_state.update_output), height=200)
                            
                            # Reset the status after showing the cancellation message
                            st.session_state.update_status = None
                            st.session_state.update_process = None
                        else:
                            st.error(f"Failed to cancel database update.")
                elif st.session_state.update_status == "error":
                    # Show error message for previous errors
                    timestamp_info = f" [{st.session_state.update_timestamp}]" if st.session_state.update_timestamp else ""
                    st.error(f"Error updating database {st.session_state.update_db}.{timestamp_info}")
                    
                    # Always show logs
                    st.text_area("Update Output", "\n".join(st.session_state.update_output), height=200)
                elif st.session_state.update_status == "cancelled":
                    # Show cancelled message
                    timestamp_info = f" [{st.session_state.update_timestamp}]" if st.session_state.update_timestamp else ""
                    st.warning(f"Database update cancelled.{timestamp_info}")
                    
                    # Always show logs
                    st.text_area("Update Output", "\n".join(st.session_state.update_output), height=200)
                elif st.session_state.update_status == "success":
                    # Show success message
                    timestamp_info = f" [{st.session_state.update_timestamp}]" if st.session_state.update_timestamp else ""
                    st.success(f"Database {st.session_state.update_db} updated successfully.{timestamp_info}")
                    
                    # Show logs if available
                    if st.session_state.update_output:
                        st.text_area("Update Output", "\n".join(st.session_state.update_output), height=200)
            
            # Add Database section
            with st.expander("Add Database"):
                # Input for repository name or URL
                repo_input = st.text_input("GitHub Repository (URL or owner/repo format)", 
                                          key="repo_input", 
                                          placeholder="e.g., github.com/username/repo or username/repo")
                
                # Add button
                add_clicked = st.button("Add", key="add_repo_button")
                
                # Reset status if add button is clicked (even without input)
                if add_clicked:
                    # Reset previous status
                    if st.session_state.add_repo_status in ["cancelled", "error", "success"]:
                        st.session_state.add_repo_status = None
                        st.session_state.add_repo_output = []
                        st.session_state.add_repo_process = None
                        st.rerun()
                
                # Process status and output
                if add_clicked and repo_input:
                    # Check if the repository already exists
                    if repository_exists(repo_input):
                        # Show warning message
                        normalized_name = normalize_repo_name(repo_input)
                        st.warning(f"Database '{normalized_name}' already exists. Please use the 'Update Database' section to update it instead.")
                    else:
                        # Start the process
                        process = process_new_repository(repo_input)
                        if process:
                            st.session_state.add_repo_process = process
                            st.session_state.add_repo_status = "running"
                            st.session_state.add_repo_input = repo_input
                            st.session_state.add_repo_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Start a thread to monitor the process
                            threading.Thread(target=monitor_process_thread, args=(process, "add_repo")).start()
                            
                            # Force a rerun to show the cancel button immediately
                            st.rerun()
                
                # Check if process is running
                if st.session_state.add_repo_process is not None and st.session_state.add_repo_status == "running":
                    # Display status (don't duplicate the info message if we just clicked the button)
                    if not add_clicked:
                        timestamp_info = f" [{st.session_state.add_repo_timestamp}]" if st.session_state.add_repo_timestamp else ""
                        st.info(f"Adding repository {st.session_state.add_repo_input}... This may take a while. You can continue using the chat while this runs in the background.{timestamp_info}")
                    
                    # Show current output (last 10 lines)
                    if st.session_state.add_repo_output:
                        st.text_area("Current Output (Updated Automatically)", 
                                    "\n".join(st.session_state.add_repo_output[-10:]), 
                                    height=100)
                    
                    # Cancel button
                    cancel_clicked = st.button("Cancel", key="cancel_add_repo_button")
                    
                    if cancel_clicked:
                        if cancel_process(st.session_state.add_repo_process):
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.warning(f"Repository addition cancelled. [{timestamp}]")
                            
                            # Show logs
                            st.text_area("Add Repository Output", "\n".join(st.session_state.add_repo_output), height=200)
                            
                            # Reset the status after showing the cancellation message
                            st.session_state.add_repo_status = None
                            st.session_state.add_repo_process = None
                        else:
                            st.error(f"Failed to cancel repository addition.")
                elif st.session_state.add_repo_status == "error":
                    # Show error message for previous errors
                    timestamp_info = f" [{st.session_state.add_repo_timestamp}]" if st.session_state.add_repo_timestamp else ""
                    st.error(f"Error adding repository {st.session_state.add_repo_input}.{timestamp_info}")
                    
                    # Always show logs
                    st.text_area("Add Repository Output", "\n".join(st.session_state.add_repo_output), height=200)
                elif st.session_state.add_repo_status == "cancelled":
                    # Show cancelled message
                    timestamp_info = f" [{st.session_state.add_repo_timestamp}]" if st.session_state.add_repo_timestamp else ""
                    st.warning(f"Repository addition cancelled.{timestamp_info}")
                    
                    # Always show logs
                    st.text_area("Add Repository Output", "\n".join(st.session_state.add_repo_output), height=200)
                elif st.session_state.add_repo_status == "success":
                    # Show success message for previous successful adds
                    timestamp_info = f" [{st.session_state.add_repo_timestamp}]" if st.session_state.add_repo_timestamp else ""
                    st.success(f"Repository {st.session_state.add_repo_input} added successfully!{timestamp_info}")
                    
                    # Always show logs
                    st.text_area("Add Repository Output", "\n".join(st.session_state.add_repo_output), height=200)
        
        # Display timing information in its own section
        col1, col2 = st.columns([5, 1])
        with col1:
            st.title("Timing Information")
        with col2:
            # Align the toggle with the title using vertical spacing
            st.write("")  # Add some space
            show_timing = st.toggle(
                "",
                value=st.session_state.show_timing,
                help="Toggle to show or hide the timing information",
                key="timing_toggle",
                label_visibility="collapsed"
            )
            
            # Check if toggle state changed
            if show_timing != st.session_state.show_timing:
                st.session_state.show_timing = show_timing
                st.rerun()
        
        # Display timing information if enabled
        if st.session_state.show_timing and st.session_state.timing_info["total_time"] > 0:
            # Create a table for timing information
            timing_data = {
                "Operation": ["Retrieval", "Generation", "Total"],
                "Time (seconds)": [
                    f"{st.session_state.timing_info['retrieval_time']:.2f}",
                    f"{st.session_state.timing_info['generation_time']:.2f}",
                    f"{st.session_state.timing_info['total_time']:.2f}"
                ]
            }
            
            st.table(timing_data)
        
        # Add another separator
        st.divider()
        
        # Retrieved Chunks section
        col1, col2 = st.columns([5, 1])
        with col1:
            st.title("Show Chunks")
        with col2:
            # Align the toggle with the title using vertical spacing
            st.write("")  # Add some space
            show_chunks = st.toggle(
                "",
                value=st.session_state.show_chunks,
                help="Toggle to show or hide the retrieved chunks",
                key="chunks_toggle",
                label_visibility="collapsed"
            )
            
            # Check if toggle state changed
            if show_chunks != st.session_state.show_chunks:
                st.session_state.show_chunks = show_chunks
                st.rerun()
    
    # Display chunks in sidebar
    display_chunks_sidebar()
    
    # Main content area
    st.title("GitHub Issues RAG System")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Create a placeholder for the thinking message
    thinking_placeholder = st.empty()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about GitHub issues..."):
        # Save the current query
        st.session_state.last_query = prompt
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Show thinking message
        with thinking_placeholder.container():
            with st.chat_message("assistant"):
                # Use a single column with the spinner wrapped around the text
                with st.spinner():
                    st.markdown("⏳ Thinking...")
        
        # Process the query
        try:
            # Start timing
            start_time = time.time()
            
            # Retrieve chunks with timing
            retrieval_start = time.time()
            chunks = st.session_state.orchestrator.retriever.retrieve(prompt)
            retrieval_end = time.time()
            retrieval_time = retrieval_end - retrieval_start
            
            # Update current chunks in session state
            st.session_state.current_chunks = chunks
            
            # Generate response with timing
            generation_start = time.time()
            result = st.session_state.orchestrator.generator.generate(prompt, chunks)
            generation_end = time.time()
            generation_time = generation_end - generation_start
            
            # Extract response from result
            if isinstance(result, dict) and "response" in result:
                response = result["response"]
            else:
                # For backward compatibility
                response = result
            
            # Calculate total time
            end_time = time.time()
            total_time = end_time - start_time
            
            # Update timing information
            st.session_state.timing_info = {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time
            }
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })
            
            # Clear the thinking message
            thinking_placeholder.empty()
            
            # Display the assistant response
            with st.chat_message("assistant"):
                st.markdown(response)
                
            # Force a rerun to update the sidebar with chunks and timing
            st.rerun()
            
        except Exception as e:
            error_message = f"Error processing your query: {str(e)}"
            
            # Add error message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_message
            })
            
            # Clear the thinking message
            thinking_placeholder.empty()
            
            # Display the error message
            with st.chat_message("assistant"):
                st.error(error_message)
            
            # Clear chunks for this query
            st.session_state.current_chunks = []

if __name__ == "__main__":
    main()
