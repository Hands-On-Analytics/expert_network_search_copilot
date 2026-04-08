# Expert Network Copilot

An AI-powered copilot that ingests candidate profiles from PostgreSQL into a Qdrant vector store and exposes a RAG (Retrieval-Augmented Generation) pipeline for semantic search and natural-language querying over an expert network.

---

## Tech Stack

- **Python 3.13+** — with [uv](https://github.com/astral-sh/uv) for package management
- **FastAPI** — Async REST API framework
- **Qdrant** — Vector database for semantic search
- **PostgreSQL** — Relational data store via SQLAlchemy (async + asyncpg)
- **LangChain + LangGraph** — RAG orchestration, embeddings, and chat
- **OpenRouter** — LLM gateway (embeddings via `openai/text-embedding-3-small`, chat via `openai/gpt-4o-mini`)
- **Docker Compose** — Local Qdrant instance

---

## Project Structure

```
expert-network-copilot/
├── app/
│   ├── main.py                   # FastAPI app entrypoint
│   ├── api/
│   │   ├── deps.py               # Dependency injection (Qdrant, checkpoint services)
│   │   └── routes/
│   │       ├── health.py          # GET  /health
│   │       ├── ingest.py          # POST /ingest/candidates
│   │       ├── qdrant.py          # GET  /qdrant/collections
│   │       └── query.py           # POST /query
│   ├── core/
│   │   ├── config.py              # Pydantic settings (reads .env)
│   │   ├── lifespan.py            # App startup/shutdown hooks
│   │   └── logging.py             # Structured logging (structlog)
│   ├── db/
│   │   ├── postgres.py            # Async SQLAlchemy session factory
│   │   └── queries.py             # SQL queries for candidate data
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response models
│   ├── services/
│   │   ├── candidate_assembler.py # Builds candidate profile objects
│   │   ├── checkpoint_service.py  # Tracks ingestion cursors
│   │   ├── document_builder.py    # Converts profiles to LangChain Documents
│   │   ├── document_grader.py     # Grades retrieved docs for relevance
│   │   ├── embedding_service.py   # OpenRouter embedding wrapper
│   │   ├── filter_extractor.py    # Extracts Qdrant filters from queries
│   │   ├── ingest_service.py      # Orchestrates batch ingestion
│   │   ├── qdrant_service.py      # Qdrant upsert, search, collections
│   │   └── rag_graph.py           # LangGraph RAG pipeline
│   └── utils/
│       └── hashing.py             # Content hashing helpers
├── scripts/
│   └── full_ingest.py             # One-off bulk ingestion script
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Getting Started — Step by Step

### Step 1: Install Prerequisites

Make sure the following are installed on your machine:

1. **Python 3.13+** — [Download](https://www.python.org/downloads/)
2. **uv** (Python package manager) — [Install guide](https://github.com/astral-sh/uv#installation)
3. **Docker Desktop** — [Download](https://www.docker.com/products/docker-desktop/) (needed to run Qdrant locally)
4. **PostgreSQL** — A running instance with your candidate data
5. **OpenRouter API key** — Sign up at [openrouter.ai](https://openrouter.ai) and generate a key

---

### Step 2: Clone the Repository

```bash
git clone https://github.com/<your-username>/test_copilot.git
cd test_copilot
```

---

### Step 3: Install Python Dependencies

```bash
uv sync
```

This creates a virtual environment (`.venv/`) and installs all packages defined in `pyproject.toml`.

---

### Step 4: Configure Environment Variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and update the following:

```env
# ── PostgreSQL ──────────────────────────────────────────────
POSTGRES_DSN=postgresql+asyncpg://user:password@localhost:5432/expert_network

# ── Qdrant ──────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=                          # leave blank for local Qdrant
QDRANT_COLLECTION=candidate_profiles

# ── OpenRouter / LLM ────────────────────────────────────────
OPENROUTER_API_KEY=<your-openrouter-api-key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENROUTER_CHAT_MODEL=openai/gpt-4o-mini

# ── Ingestion ───────────────────────────────────────────────
INGEST_BATCH_SIZE=200
```

---

### Step 5: Start Qdrant (via Docker)

```bash
docker compose up -d
```

This starts a local Qdrant instance on **port 6333** (REST) and **6334** (gRPC).  
Verify it's running: open [http://localhost:6333/dashboard](http://localhost:6333/dashboard) in your browser.

---

### Step 6: Run the API Server

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at **http://localhost:8000**.  
Interactive Swagger docs: **http://localhost:8000/docs**

---

### Step 7: Ingest Candidate Profiles

You have two options to ingest candidates from PostgreSQL into Qdrant:

**Option A — Via API** (incremental or full reindex):

```bash
curl -X POST http://localhost:8000/ingest/candidates \
  -H "Content-Type: application/json" \
  -d '{"full_reindex": true, "dry_run": false, "limit": 500}'
```

**Option B — Via script** (bulk one-off ingestion):

```bash
uv run python -m scripts.full_ingest
```

This script loops through all candidates in batches of 500 until the database is fully ingested.

---

### Step 8: Query the Expert Network

Send a natural-language question to the RAG pipeline:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find candidates with experience in machine learning and NLP", "top_k": 5}'
```

The response includes a generated answer and the source document chunks used to produce it.

---

## API Endpoints

**Health**
- `GET /health` — Returns `{"status": "ok"}` when the server is running.

**Ingestion**
- `POST /ingest/candidates` — Ingest candidate profiles from PostgreSQL into Qdrant.  
  Body: `{"full_reindex": bool, "dry_run": bool, "limit": int}`

**Qdrant**
- `GET /qdrant/collections` — List all Qdrant collections and their count.

**Query**
- `POST /query` — Run the RAG pipeline: retrieves relevant candidates, grades documents, and generates an answer.  
  Body: `{"query": "your question", "top_k": 5}`

---

## How It Works

1. **Ingest** — Candidate profiles are fetched from PostgreSQL, assembled into rich text documents, embedded via OpenRouter, and upserted into Qdrant with a content hash to avoid duplicates.
2. **Query** — A user's natural-language question flows through a LangGraph RAG pipeline that:
   - Extracts metadata filters from the query
   - Performs semantic search in Qdrant
   - Grades retrieved documents for relevance
   - Generates a final answer using the chat model

---

## Stopping Services

To stop the Qdrant container:

```bash
docker compose down
```

To stop and remove stored data:

```bash
docker compose down -v
```

---

## License

MIT
