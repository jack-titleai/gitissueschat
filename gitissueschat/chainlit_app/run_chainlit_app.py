#!/usr/bin/env python3
"""
Script to run the Chainlit app for the RAG system.
"""

import os
import sys
import logging
import subprocess
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the Chainlit app.
    """
    # Load environment variables
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp.env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        logger.info(f"Loaded environment variables from {dotenv_path}")
    else:
        logger.warning(f"Environment file {dotenv_path} not found")
    
    # Get the app directory
    app_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app")
    app_file = os.path.join(app_dir, "chainlit_app.py")
    
    # Check if the app file exists
    if not os.path.exists(app_file):
        logger.error(f"App file not found: {app_file}")
        sys.exit(1)
    
    # Run the Chainlit app
    logger.info(f"Running Chainlit app from {app_file}")
    try:
        subprocess.run(["chainlit", "run", app_file], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running Chainlit app: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Chainlit app stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
