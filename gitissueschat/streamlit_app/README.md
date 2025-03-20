# GitHub Issues Chat - Streamlit App

This Streamlit application provides a user-friendly interface to interact with GitHub issues from various repositories using a Retrieval-Augmented Generation (RAG) system.

## Features

- **Chat Interface**: Ask questions about GitHub issues and get AI-generated responses based on the content of the issues.
- **Database Management**: Add new repositories, update existing ones, and switch between different repositories.
- **Background Processing**: Long-running tasks like adding repositories or updating databases run in the background, allowing you to continue using the chat interface.
- **Real-time Feedback**: View progress and logs of background processes with timestamps.
- **Customizable Settings**: Toggle various display options and select different databases.

## Known Issues
- There is a known issue with the process cancelling being weird.  It will successfully cancel a process, but the next action you try to take won't work - the app will work normally after that (kinda just waste an action like a button click or chat submission and things will go back to normal)
- There is a known issue with the background processes not updating until an action is taken.  this means you can use the chat as normal, but after the process (either adding a database or updating a database) finishes, you will no know until you try to submit another chat (or click a button), that action will not succeed but the success/failure message for the background process will update.  you can then continue to use the app as normal

## Environment Setup

Before running the app, you need to set up the following environment variables in a `.env` file at the root of the project:

```
GITHUB_TOKEN="your_github_personal_access_token"
GOOGLE_PROJECT_ID="your_google_project_id"
GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/google_credentials.json"
```

See main project readme for further instructions on setting up environment variables

## Running the App

To run the app, navigate to the project root directory and execute:

```bash
streamlit run gitissueschat/streamlit_app/app.py
```

## App Sections and Settings

### Main Chat Interface
- **Chat Input**: Type your questions about GitHub issues here.
- **Chat History**: View the conversation history with the AI assistant.

### Database Settings (Sidebar)
- **Database Selection**: Choose which GitHub repository database to query.
- **Force Refresh Database Connection**: Manually refresh the connection to the database.
- **Update Database**: Update an existing repository with new issues.
  - Select the database to update
  - Click "Update" to start the process
  - View real-time logs and progress
  - Cancel the update if needed
- **Add Database**: Add a new GitHub repository to the system.
  - Enter the repository URL or owner/repo format (e.g., `github.com/username/repo` or `username/repo`)
  - Click "Add" to start the process
  - View real-time logs and progress
  - Cancel the addition if needed

### Display Options (Sidebar)
- **Show Chunks**: Toggle to show or hide the retrieved chunks that were used to generate the response.
- **Show Timing Information**: Toggle to show or hide timing information about retrieval and generation.

## Background Processing

The app uses threading to handle long-running tasks in the background, allowing you to:
- Continue chatting while a repository is being added or updated
- Monitor the progress of background tasks
- Cancel background tasks if needed
- Receive notifications when tasks are completed

## Troubleshooting

If you encounter any issues:
1. Check the console logs for error messages
2. Verify that your environment variables are set correctly
3. Ensure you have the necessary permissions for the GitHub repositories you're trying to access
4. Check that your Google Cloud service account has the required permissions

## Data Storage

The app stores data in the following locations:
- **Database Files**: `./data/chroma_dbs/<owner>_<repo>/`
- **Temporary Process Logs**: `./gitissueschat/streamlit_app/temp/`

