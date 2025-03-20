# Process Repository Script

This script automates the entire workflow for processing a GitHub repository's issues:
1. Downloads all issues and comments from a GitHub repository to an SQLite database
2. Chunks the issues and comments with specified parameters
3. Embeds the chunks into a ChromaDB database for later retrieval
4. Analyzes the repository data to provide statistics about issues, comments, and chunks

note that the download portion can be stopped and restarted - it will pickup where it left off.  the embedding piece cannot, it must embed all in one go- if it is interrupted, you will need to delete the partial database and restart.  this can be done by including the --skip-download flag.

## Prerequisites

- Python 3.7+
- GitHub API token (for accessing GitHub issues)
- Google Cloud credentials (for embedding)

## Environment Variables

The script requires the following environment variables:

```
GITHUB_TOKEN=your_github_token
GOOGLE_PROJECT_ID=your_google_project_id
GOOGLE_APPLICATION_CREDENTIALS=path_to_service_account_key_file
```

You can set these in a `.env` file in the project root directory.

## Usage

The script accepts either a GitHub repository URL or a repository name in the format 'owner/repo':

```bash
# Using repository name format
python -m gitissueschat.utils.process_repository owner/repo [options]

# Using GitHub URL format
python -m gitissueschat.utils.process_repository https://github.com/owner/repo [options]

# Analyze existing data without downloading or embedding
python -m gitissueschat.utils.process_repository owner/repo --skip-download --skip-embed
```

## Options

- `--token TOKEN`: GitHub API token (if not provided, will use the `GITHUB_TOKEN` environment variable)
- `--chunk-size SIZE`: Size of chunks in tokens (default: 500)
- `--chunk-overlap OVERLAP`: Overlap between chunks in tokens (default: 100)
- `--limit LIMIT`: Limit the number of issues to process
- `--skip-download`: Skip downloading issues (useful if you already have the SQLite database)
- `--skip-embed`: Skip embedding issues (useful if you only want to download issues)
- `--collection-name NAME`: Name of the ChromaDB collection (default: "github_issues")

## Example

```bash
# Process a repository with default settings
python -m gitissueschat.utils.process_repository fastai/fastai

# Process a repository with custom chunk size and overlap
python -m gitissueschat.utils.process_repository fastai/fastai --chunk-size 300 --chunk-overlap 50

# Process a repository with a limit on the number of issues
python -m gitissueschat.utils.process_repository fastai/fastai --limit 100

# Only analyze an existing repository database
python -m gitissueschat.utils.process_repository fastai/fastai --skip-download --skip-embed
```

## Output

The script provides detailed statistics about the repository, including:

- Number of issues and comments
- Average comments per issue
- Most commented issue
- Distribution of issue states
- Issues by month
- Top issue authors
- Chunk statistics (count, character length, tokens)
- Chunks per issue statistics

## Database Paths

The script automatically determines the paths for the SQLite and ChromaDB databases:

- SQLite database: `./data/sqlite_dbs/<owner>_<repo>.db`
- ChromaDB database: `./data/chroma_dbs/<owner>_<repo>/`

## Embedding Process

The script uses the following process for embedding:

1. Issues and comments are processed separately (not combined into one text)
2. Each issue and comment is individually chunked if they're too long
3. Chunks are batched (100 at a time) for efficient embedding
4. Each chunk is stored separately in ChromaDB with appropriate metadata
5. Metadata includes issue number, title, state, author, creation date, and update date

## Incremental Updates

After initial processing, you can use the `update_repository.py` script to efficiently update the database with new or modified issues and comments.
