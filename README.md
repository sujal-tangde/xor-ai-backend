# XOR AI Backend

FastAPI service powering the XOR chat experience: a tool-using "deep agent" (LiteLLM → AWS Bedrock) with RAG over uploaded images/documents (pgvector), background upload processing (Redis + RQ), and should-cost report generation.

## Architecture at a glance

- **API** — FastAPI app (`src/main.py`) exposing chat, auth, projects, files, and reports routers.
- **Worker** — a separate RQ process (`worker.py`) that runs image/document analysis, embedding, insight, and knowledge-base jobs enqueued by the upload API.
- **Redis** — message history cache + the `xor-processing` job queue.
- **Supabase** — Postgres (with the `pgvector` extension) for metadata, chunks, insights, and reports; Storage buckets for original/compressed uploads. The schema is created automatically on startup (`src/core/db.py`).
- **External services** — AWS Bedrock (LLM + embeddings via LiteLLM), Tavily (web search), DigiKey (component pricing).

## Prerequisites

- **Python 3.10+** (the code uses `X | Y` type unions).
- **Redis** running and reachable (local install, Docker, or a managed instance).
- A **Supabase** project (Postgres connection string + service/anon keys + storage buckets).
- **AWS Bedrock** access (API key/base for LiteLLM) for the LLM and embedding models.
- Optional: **Tavily** and **DigiKey** API credentials for web search and component pricing.

## 1. Install

From the `xor-ai-backend` directory:

```powershell
# Create and activate a virtual environment (Windows / PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

<details>
<summary>macOS / Linux (bash)</summary>

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
</details>

## 2. Configure environment

Create a `.env` file in `xor-ai-backend/` (next to `requirements.txt`). All settings are read from the environment via `src/core/config.py`. Values shown are the defaults — only the ones without sensible defaults are strictly required.

```dotenv
# --- Redis (job queue + chat history) ---
REDIS_URL=redis://localhost:6379
MAX_CHAT_MESSAGES=20

# --- LLM (AWS Bedrock via LiteLLM; must support tool calling) ---
LLM_MODEL=bedrock/qwen.qwen3-vl-235b-a22b
LLM_API_KEY=
LLM_API_BASE=
LLM_TOOLS_ENABLED=true
AWS_REGION=us-east-1

# --- Embeddings (Bedrock via LiteLLM) for pgvector RAG ---
EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
EMBEDDING_API_BASE=
EMBEDDING_API_KEY=
EMBEDDING_DIM=1024
EMBEDDING_SEND_DIMENSIONS=true
KB_RELATEDNESS_THRESHOLD=0.15

# --- Supabase (Postgres metadata + Storage for uploads) ---
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
SUPABASE_ANON_KEY=
DIRECT_URL=                       # Postgres direct connection string (used for schema DDL)
STORAGE_BUCKET_1=compressed_uploads
STORAGE_BUCKET_2=original_uploads
AUTH_REDIRECT_URL=http://localhost:5173/auth/callback

# --- Tavily web search (optional) ---
TAVILY_API_KEY=

# --- DigiKey pricing (optional) ---
DIGIKEY_CLIENT_ID=
DIGIKEY_CLIENT_SECRET=
DIGIKEY_API_BASE=https://api.digikey.com
DIGIKEY_LOCALE_SITE=US
DIGIKEY_LOCALE_LANGUAGE=en
DIGIKEY_LOCALE_CURRENCY=USD

# --- Report generation defaults ---
REPORT_DEFAULT_VOLUME=1000
REPORT_MAX_QUESTIONS=4
REPORT_USD_INR_FALLBACK=85.0
```

> **Note:** `DIRECT_URL` must point at a Postgres database with the `pgvector` extension available — on startup the app runs the schema DDL (`CREATE EXTENSION IF NOT EXISTS vector`, tables, and indexes). If `DIRECT_URL` is empty, schema creation is skipped.

## 3. Run

You need **two processes**: the API server and the background worker. Make sure Redis is running first.

### API server

```powershell
# Development (auto-reload)
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

- API root health check: http://localhost:8000/
- Interactive API docs (Swagger): http://localhost:8000/docs

### Background worker

In a second terminal (with the virtualenv activated):

```powershell
python worker.py
```

This consumes the `xor-processing` queue and handles upload analysis, embedding, insight extraction, and knowledge-base recompute jobs. The worker uses `SimpleWorker` (in-process, no `os.fork`) so it runs on Windows. Without it, uploads are stored but never processed.

## Quick reference

| Component | Command |
| --- | --- |
| Install deps | `pip install -r requirements.txt` |
| Run API | `uvicorn src.main:app --reload --port 8000` |
| Run worker | `python worker.py` |
| Redis (Docker) | `docker run -p 6379:6379 redis` |

## Troubleshooting

- **Uploads stuck "processing"** — the RQ worker isn't running, or `REDIS_URL` is wrong. Enqueue failures are logged but don't fail the upload request.
- **Schema not created / DB errors** — verify `DIRECT_URL` is set and the database has the `pgvector` extension.
- **LLM/embedding errors** — confirm `LLM_API_KEY` / `LLM_API_BASE` and `AWS_REGION` are set for Bedrock, and that the configured `LLM_MODEL` supports tool calling.
