# Study-Mosaic: one source, many learning shapes

Grounded study modules from your PDFs. Upload documents, ingest them locally (chunk + embed with Gemini), and let agents draft flashcards with citations while a coverage loop checks how much of the retrieved context has been used.

## Prerequisites
- Python 3.10+ and pip
- Google Gemini API key (`GEMINI_API_KEY`)
- Optional: `pytest` for the small ingest unit test

## Quickstart
1) Clone and enter the repo  
`git clone <repo-url> && cd study-mosaic`
2) Create a virtual env  
`python -m venv .venv && .\.venv\Scripts\activate`
3) Install dependencies  
`pip install -r requirements.txt`
4) Add your Gemini key to `.env` at the repo root  
`echo GEMINI_API_KEY=your-key-here > .env`

## Running the app
- From the repo root:  
`streamlit run ui/app.py`
- In the UI: set a session ID, upload 1-5 PDFs, ingest new or existing uploads, then ask for flashcards. Outputs include coverage stats and a JSON download.

## Data and storage
- Uploads: `data/uploads/<session_id>/`
- Local Chroma vector store: `data/vectors/`
- Session state (ADK): `data/<session_id>.json`

## Tests
- (Optional) Run the ingest chunking test:  
`pytest tests/test_ingest.py`
