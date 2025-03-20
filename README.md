# gitissueschat
RAG based chatbot to interact with git issues for any given repo

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for GitHub issues. It allows you to:
1. Fetch issues from any GitHub repository
2. Store them in a SQLite database with structured metadata
3. Efficiently update only new or modified issues
4. Query the issues using SQL or simple search functions

## Environment Setup

Before running the app, you need to set up the following environment variables in a `.env` file at the root of the project:

```
GITHUB_TOKEN="your_github_personal_access_token"
GOOGLE_PROJECT_ID="your_google_project_id"
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google_credentials.json"
```

### How to Get the Required Credentials

#### GitHub Personal Access Token
1. Go to your GitHub account settings: https://github.com/settings/tokens
2. Click "Generate new token" (classic)
3. Give it a name and select the following scopes:
   - `repo` (Full control of private repositories)
   - `read:org` (Read-only access to organization membership)
4. Click "Generate token"
5. Copy the token and add it to your `.env` file

#### Finding Your Google Project ID
1. Go to the Google Cloud Console: https://console.cloud.google.com/
2. At the top of the page, click on the dropdown next to the Google Cloud logo
3. In the popup window, you'll see a list of your projects
4. The Project ID is displayed in the "ID" column (not the project name)
5. If you're creating a new project, you can specify a custom Project ID during creation
6. You can also find your Project ID in the "Dashboard" section of your project
7. Copy this Project ID and add it to the `GOOGLE_PROJECT_ID` field in your `.env` file

#### Creating Google Service Account Credentials
1. Go to the Google Cloud Console: https://console.cloud.google.com/
2. Select your project
3. Enable the Vertex AI API for your project:
   - Go to "APIs & Services" > "Library"
   - Search for "Vertex AI API"
   - Click on it and then click "Enable"
4. Create a service account:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Give it a name and description
   - Grant it the "Vertex AI User" role
5. Create a key for the service account:
   - Click on the service account you just created
   - Go to the "Keys" tab
   - Click "Add Key" > "Create new key"
   - Choose JSON format and click "Create"
   - The key file will be downloaded to your computer
6. Move the key file to a secure location and update the `GOOGLE_APPLICATION_CREDENTIALS` in your `.env` file with the absolute path to this file

## Test Repositories
The system has been tested with repositories of various sizes:

- **Extra Small** - FlyFish
    - https://github.com/CloudWise-OpenSource/FlyFish
    - 81 closed issues
    - 21 open issues

- **Small** - fastai 
    - https://github.com/fastai/fastai
    - 1595 closed issues 
    - 215 open issues

- **Medium** - mlflow
    - https://github.com/mlflow/mlflow
    - 2742 closed issues
    - 1380 open issues

- **Large** - scikit-learn
    - https://github.com/scikit-learn/scikit-learn
    - 9845 closed issues
    - 1586 open issues

## Setup

### Prerequisites
- Conda (recommended for environment management)
Or
- Python 3.10 or higher
- pip (Python package installer)

### Installation

#### Using Conda (Recommended)
1. Clone this repository
2. Create and activate the conda environment:
   ```bash
   # Create the conda environment
   conda create -n gitissueschat python=3.10 -y
   
   # Activate the environment
   conda activate gitissueschat
   ```
3. Install the package in development mode:
   ```bash
   # Navigate to the project root directory
   cd gitissueschat

   # Activate conda environment
   conda activate gitissueschat
   
   # Install the package in development mode
   pip install -e .
   ```

## Usage

There are 2 options for usage.  The first is the streamlit app.  This is the more user-friendly way, but gives less insight into the process.  The second is the command line interface.  This is the more technical way, but gives more insight into what is happening.

### Streamlit App

The project includes a user-friendly Streamlit app in the `gitissueschat/streamlit_app/` directory:

#### Features
- **Chat Interface**: Ask questions about GitHub issues and get AI-generated responses
- **Database Management**: Add new repositories, update existing ones, and switch between different repositories
- **Background Processing**: Long-running tasks run in the background, allowing continued use of the chat interface
- **Real-time Feedback**: View progress and logs of background processes with timestamps

#### Running the App
```bash
streamlit run gitissueschat/streamlit_app/app.py
```

For more details about the Streamlit app, see the [Streamlit App README](gitissueschat/streamlit_app/README.md).


### Command Line Interface

#### Initial Processing

To process a GitHub repository for the first time, use the `process_repository.py` script:

```bash
# Basic usage
python -m gitissueschat.utils.process_repository owner/repo

# With custom chunk size and overlap
python -m gitissueschat.utils.process_repository owner/repo --chunk-size 300 --chunk-overlap 50

# Limit the number of issues to process
python -m gitissueschat.utils.process_repository owner/repo --limit 100
```

Note: The download portion can be stopped and restarted - it will pick up where it left off. However, the embedding process cannot be interrupted; if it is, you will need to delete the partial database and restart with the `--skip-download` flag.

#### Updating Repositories

To update an existing repository with new or modified issues:

```bash
# Basic usage
python -m gitissueschat.utils.update_repository owner/repo

# With custom chunk size and overlap
python -m gitissueschat.utils.update_repository owner/repo --chunk-size 300 --chunk-overlap 50

# Disable the buffer period for the last update time
python -m gitissueschat.utils.update_repository owner/repo --disable-buffer
```

#### Using the RAG System

The project includes a command-line interface for the RAG system:

```bash
# Basic usage with a single query
python -m gitissueschat.rag.cli --repository owner/repo --query "Your question about the repository issues?"

# Interactive mode (no query specified)
python -m gitissueschat.rag.cli --repository owner/repo

# Customize retrieval parameters
python -m gitissueschat.rag.cli --repository owner/repo --top-k 15 --relevance-threshold 0.6

# Use a specific collection name
python -m gitissueschat.rag.cli --repository owner/repo --collection-name custom_collection

# Filter by issue number
python -m gitissueschat.rag.cli --repository owner/repo --issue-number 123
```

Additional CLI options:
- `--model`: Specify the Gemini model to use (default: "gemini-2.0-flash-001")
- `--temperature`: Set the temperature for generation (default: 0.2)
- `--db-path`: Manually specify the ChromaDB path (overrides default)

## Database Paths

The system automatically organizes databases in the following locations:

- **SQLite databases**: `./data/sqlite_dbs/<owner>_<repo>.db`
- **ChromaDB databases**: `./data/chroma_dbs/<owner>_<repo>/`


## Next Steps
- Improve file organization and naming conventions
- Add context into the RAG - currently it only supports one shot prompting, should be able to support multi-shot with user context without much additional work
- Make RAG cite sources
- Build a better app - maybe hosted on a website so users don't have to host locally
- Add support for additional LLM models