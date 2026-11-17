# Streamlit Community Cloud deployment

## Deploy settings

1. Select the GitHub repository and the branch containing these changes.
2. Set **Main file path** to `app.py` (the repository-root entrypoint).
3. In **Advanced settings**, select **Python 3.13**. The project was checked
   with Python 3.13.2; Python 3.14 is not the selected deployment runtime.
   Community Cloud selects this in Advanced settings; do not rely on a
   repository runtime file to override the dropdown.
4. Leave **Secrets** empty. This app does not require API keys or paid services.
5. Deploy with the root `requirements.txt`.

The app selects Cloud mode by default on Streamlit Community Cloud's Linux
runtime. Do not set `PKA_MODE=local` there. Local Windows execution defaults to
local mode and can use the optional model dependencies in
`requirements-local.txt`.

## Cloud behavior and storage

Cloud mode uses CPU-only hashing embeddings, FAISS, and BM25. It does not load
Qwen at startup or for summaries and quizzes: summaries select exact source
sentences, quizzes use source sentences as choices, and chat returns a relevant
document sentence with its source or clearly reports that the answer was not
found. Whisper's `tiny` model is loaded only after a recording is submitted.
These fallbacks are extractive, not generative.

Uploads are limited to 20 MB per file and PDFs to 500 pages to reduce memory
and processing risks. Voice recording depends on the visitor's browser
supporting microphone capture and granting permission; text chat remains
available without it.

Uploads and indexes use temporary session storage. Streamlit Community Cloud
may restart or replace the instance, so this storage is not durable and files
may need to be uploaded and processed again.
