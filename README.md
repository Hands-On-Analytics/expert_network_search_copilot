# Candidate Vector APIs
FastAPI project implementing two APIs:
- API 1: vector database construction (`POST /ingest`)
- API 2: conversational expert search (`POST /search`)

## Features
- Extracts candidate/expert profile data from PostgreSQL.
- Chunks profile data by semantic sections (summary, skills, languages, education, work history).
- Generates embeddings through OpenRouter.
- Stores vectors + metadata in ChromaDB.
- Supports session-aware conversational follow-up search.

## Tech Stack
- Python
- FastAPI
- Pydantic
- PostgreSQL (`psycopg`)
- OpenRouter embeddings
- ChromaDB (persistent local store)
- `uv` for dependency and environment management

## Project Structure
```text path=null start=null
candidate_vector_api/
  src/candidate_vector_api/
    api.py            # FastAPI routes
    schemas.py        # Pydantic request/response models
    ingestion.py      # API 1 pipeline
    search.py         # API 2 conversational retrieval
    repository.py     # PostgreSQL extraction
    chunking.py       # profile chunk generation
    embeddings.py     # OpenRouter embedding client
    vector_store.py   # ChromaDB adapter
    settings.py       # environment configuration
  .env.example
  README.md
```

## Setup
1. Install dependencies:
```powershell path=null start=null
uv sync --project .\candidate_vector_api
```

2. Create environment file:
```powershell path=null start=null
Copy-Item .\candidate_vector_api\.env.example .\candidate_vector_api\.env
```

3. Update `candidate_vector_api/.env` with valid values:
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `OPENROUTER_API_KEY`
- optional tuning values for Chroma/search

## Run the Server
From repository root:
```powershell path=null start=null
uv run --project .\candidate_vector_api candidate-vector-api serve --host 0.0.0.0 --port 8000 --reload
```

API docs:
- Swagger: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## API Usage Examples
### Health
```bash path=null start=null
curl -s http://localhost:8000/health
```

### API 1: Ingest vectors
#### curl
```bash path=null start=null
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"full_refresh": true}'
```

#### PowerShell
```powershell path=null start=null
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ingest -ContentType "application/json" -Body '{"full_refresh":true}'
```

### API 2: Conversational search
#### Initial query
```bash path=null start=null
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Find me regulatory affairs experts with experience in the pharmaceutical industry in the Middle East.","top_k":5}'
```

#### Follow-up query with session context
Use the `session_id` returned by the initial response:
```bash path=null start=null
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Filter those to only people based in Saudi Arabia","session_id":"<SESSION_ID>","top_k":5}'
```

## Pydantic Request/Response Schemas
All endpoint payloads are defined with Pydantic models in `src/candidate_vector_api/schemas.py`.

- `GET /health`
  - Response: `HealthResponse`
- `POST /ingest`
  - Request: `IngestRequest`
  - Response: `IngestResponse`
- `POST /search`
  - Request: `SearchRequest`
  - Response: `SearchResponse` (contains `ExpertResult[]`)

## Design Notes
### 1) Embedding Strategy
- Candidate profiles are chunked by semantic sections instead of one large document:
  - profile summary
  - skills
  - languages
  - education entries
  - work experience entries
- This improves retrieval precision by allowing specific user intent (skills/location/history) to match the most relevant chunk type.

### 2) Vector Database Choice
- ChromaDB is used as the vector database because:
  - easy local persistence and setup
  - simple Python integration
  - supports metadata storage and filtering workflows in application logic

### 3) Query Handling Approach
- For `POST /search`, natural language query is transformed into multiple variants:
  - standalone rewritten query (for follow-ups)
  - extracted keyword-focused variant
  - location-aware expansion (for geographic intents)
- Each variant is embedded and searched; chunk-level hits are aggregated to candidate-level ranking.
- Result includes:
  - expert identity/context
  - `why_match` reasons
  - `key_highlights` snippets from matched chunks

### 4) Conversational Context
- In-memory session store maps `session_id` to prior query/results.
- Follow-up requests can reference prior results (`"Filter those..."`) and apply additional narrowing.

### 5) Trade-offs
- In-memory session storage is simple and fast but not durable across process restarts.
- Retrieval uses application-level aggregation/filtering; this is flexible but adds CPU work versus native DB filtering for all constraints.
- Query rewriting is heuristic-based (deterministic) rather than LLM-agentic rewriting, which keeps latency/cost lower but may miss some nuanced intents.

## Useful Commands
### Ingest full dataset from CLI
```powershell path=null start=null
uv run --project .\candidate_vector_api candidate-vector-api ingest --full-refresh
```

### Run a limited ingestion smoke test
```powershell path=null start=null
uv run --project .\candidate_vector_api candidate-vector-api ingest --full-refresh --limit 25
```
