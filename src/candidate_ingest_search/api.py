from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from candidate_ingest_search.ingestion import CandidateIngestionPipeline
from candidate_ingest_search.schemas import (
    ExpertResult,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from candidate_ingest_search.search import CandidateSearchService
from candidate_ingest_search.settings import get_settings


@lru_cache
def get_pipeline() -> CandidateIngestionPipeline:
    settings = get_settings()
    return CandidateIngestionPipeline.from_settings(settings)


@lru_cache
def get_search_service() -> CandidateSearchService:
    settings = get_settings()
    return CandidateSearchService.from_settings(settings)


app = FastAPI(
    title="Candidate Vector APIs",
    version="1.0.0",
    description=(
        "API 1: ingest candidate data from PostgreSQL into ChromaDB. "
        "API 2: conversational natural-language expert search over ChromaDB vectors."
    ),
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    try:
        result = get_pipeline().ingest(
            full_refresh=request.full_refresh,
            limit=request.limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(status="success", **result.to_dict())


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        result = get_search_service().search(
            query=request.query,
            session_id=request.session_id,
            top_k=request.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc

    return SearchResponse(
        status="success",
        session_id=result.session_id,
        original_query=result.original_query,
        transformed_query=result.transformed_query,
        query_variants=result.query_variants,
        follow_up_detected=result.follow_up_detected,
        applied_filters=result.applied_filters,
        experts=[ExpertResult(**asdict(expert)) for expert in result.experts],
        total_experts=len(result.experts),
    )
