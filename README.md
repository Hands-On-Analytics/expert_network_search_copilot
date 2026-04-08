# Expert Network Copilot

A FastAPI-based AI copilot that ingests candidate profiles into a vector store (Qdrant) backed by a PostgreSQL database, enabling semantic search over an expert network.

## Tech Stack

- **Python 3.13+** with [uv](https://github.com/astral-sh/uv) for package management
- **FastAPI** — REST API framework
- **Qdrant** — Vector database for semantic search
- **PostgreSQL** — Relational store via SQLAlchemy (async)
- **LangChain + OpenRouter** — Embeddings via `openai/text-embedding-3-small`
- **Docker Compose** — Local Qdrant instance

## Project Structure

```
expert_network_copilot/
├── app/
│   ├── api/
│   │   └── routes/       # health, ingest endpoints
│   ├── core/             # config, lifespan, logging
│   ├── db/               # postgres connection & queries
│   ├── models/           # pydantic schemas
│   ├── services/         # ingest, embedding, qdrant, candidate assembler
│   └── utils/            # hashing helpers
├── docker-compose.yml
├── pyproject.toml
└── main.py
```

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (for Qdrant)
- A PostgreSQL instance
- An [OpenRouter](https://openrouter.ai) API key

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/test_copilot.git
cd test_copilot

# Create virtual environment and install dependencies
uv sync
```

### Environment Variables

Copy `.env.example` to `.env` (or create `.env`) and fill in the values:

```env
POSTGRES_DSN=postgresql+asyncpg://user:password@localhost:5432/expert_network

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=          # optional for local Qdrant
QDRANT_COLLECTION=candidate_profiles

OPENROUTER_API_KEY=<your-openrouter-api-key>
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small

INGEST_BATCH_SIZE=200
```

### Start Qdrant

```bash
docker compose up -d
```

### Run the API

```bash
uv run uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest` | Ingest candidate profiles into Qdrant |

Interactive docs available at `http://localhost:8000/docs`.

## License

MIT