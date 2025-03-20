"""
Simple script to test the Gemini API.
"""

import os
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gemini_api():
    """Test the Gemini API with a simple question."""
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Configure the Gemini API
    if api_key:
        logger.info("Configuring Gemini API with API key")
        genai.configure(api_key=api_key)
    else:
        # Try to use service account credentials
        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path:
            logger.info(f"Configuring Gemini API with service account credentials from {credentials_path}")
            import google.auth
            
            # Load credentials from the service account file
            try:
                credentials, project_id = google.auth.load_credentials_from_file(credentials_path)
                genai.configure(credentials=credentials)
                logger.info(f"Successfully configured Gemini API with service account for project {project_id}")
            except Exception as e:
                logger.error(f"Error loading service account credentials: {e}")
                return
        else:
            logger.error("Neither API key nor service account credentials provided")
            return
    
    # Initialize the model
    model_name = "gemini-2.0-flash-001"
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 1024,
            "top_p": 0.95,
            "top_k": 40
        },
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    # Test question
    question = "What is the capital of France?"
    
    logger.info(f"Sending question to Gemini API: {question}")
    try:
        response = model.generate_content(question)
        logger.info(f"Response received: {response.text}")
        print("\nQuestion:", question)
        print("\nResponse:", response.text)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_gemini_api()