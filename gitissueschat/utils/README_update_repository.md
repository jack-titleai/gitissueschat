# Update Repository Script

This script updates an existing repository database with new or modified issues and comments. It checks the most recent API call from the SQLite database and fetches any issues/comments that have been updated since then. It updates both the SQLite database and the ChromaDB instances accordingly.

## Features

- Retrieves the timestamp of the most recent API call from the SQLite database
- Fetches only issues and comments that have been modified since the last update
- Updates the SQLite database with new or modified issues and comments
- For updated issues/comments, removes the old chunks from ChromaDB and adds the new ones
- Provides detailed statistics about the update process

## Usage

```bash
python -m gitissueschat.utils.update_repository owner/repo [options]
```

or

```bash
python -m gitissueschat.utils.update_repository https://github.com/owner/repo [options]
```

## Options

- `--token TOKEN`: GitHub API token (if not provided, will use the `GITHUB_TOKEN` environment variable)
- `--collection-name NAME`: Name of the ChromaDB collection (default: "github_issues")
- `--chunk-size SIZE`: Size of chunks in tokens (default: 500)
- `--chunk-overlap OVERLAP`: Overlap between chunks in tokens (default: 100)
- `--disable-buffer`: Disable the buffer period for the last update time (by default, a buffer period is used to ensure no issues are missed)

## Environment Variables

The script requires the following environment variables to be set:

- `GITHUB_TOKEN`: GitHub API token (required if not provided via command line)
- `GOOGLE_PROJECT_ID`: Google Cloud project ID (required)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to the Google Cloud service account key file (required)

## Example

```bash
# Update a repository using the GITHUB_TOKEN environment variable
python -m gitissueschat.utils.update_repository fastai/fastai

# Update a repository with a custom token and chunk size
python -m gitissueschat.utils.update_repository fastai/fastai --token your_token --chunk-size 300

# Update a repository and disable the buffer period
python -m gitissueschat.utils.update_repository fastai/fastai --disable-buffer
```

## Output

The script provides detailed statistics about the update process:

- Number of new issues added
- Number of issues updated
- Number of redundant issues (no changes)
- Number of issues before and after the update
- Number of chunks added and removed
- API rate limit information
- Execution time

## Database Paths

The script automatically determines the paths for the SQLite and ChromaDB databases:

- SQLite database: `./data/sqlite_dbs/<owner>_<repo>.db`
- ChromaDB database: `./data/chroma_dbs/<owner>_<repo>/`

## Update Process

The update process follows these steps:

1. Get the timestamp of the most recent API call from the SQLite database
2. Apply a buffer period to ensure no issues are missed (can be disabled with `--disable-buffer`)
3. Fetch all issues modified since that timestamp
4. For each issue:
   - If it's a new issue, add it to the SQLite database
   - If it's an existing issue, check if it has been modified
   - If modified, update it in the SQLite database and update its chunks in ChromaDB
5. Log the API call with statistics about the update

## Initial Setup

Before using this script, you must first process the repository using the `process_repository.py` script to create the initial database.
