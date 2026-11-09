# DUM-E

DUM-E is a CPU-friendly, subject-scoped RAG chatbot with FastAPI and React
frontends.

## Setup on Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
Copy-Item .env.example .env
```

Edit `.env` with the LLM provider settings before using the application.

## Run and test

Start the authenticated API from the repository root:

```powershell
$env:PYTHONPATH="src"
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Start the React development server in a second terminal:

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

Run the backend test suite:

```powershell
python -m pytest -q
```

The API uses `data/users.sqlite` for authentication, stores user documents
under `data/raw/{user_id}/{subject}/`, and requires `LLM_API_KEY` in `.env`.
`LLM_BACKUP_API_KEY` is optional and is used for eligible provider failures.
