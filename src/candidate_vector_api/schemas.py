from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class IngestRequest(BaseModel):
    full_refresh: bool = Field(
        default=True,
        description="If true, reset vector collection before inserting new embeddings.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional candidate limit for development or smoke runs.",
    )


class IngestResponse(BaseModel):
    status: str
    candidates_processed: int
    chunks_generated: int
    chunks_upserted: int
    vector_collection: str
    full_refresh: bool
    duration_seconds: float


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query.")
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier for conversational follow-up queries.",
    )
    top_k: int = Field(default=10, ge=1, le=30, description="Maximum experts to return.")


class ExpertResult(BaseModel):
    candidate_id: str
    full_name: str
    headline: str
    city: str
    country: str
    nationality: str
    years_of_experience: int
    match_score: float
    why_match: list[str]
    key_highlights: list[str]
    matched_chunk_types: list[str]


class SearchResponse(BaseModel):
    status: str
    session_id: str
    original_query: str
    transformed_query: str
    query_variants: list[str]
    follow_up_detected: bool
    applied_filters: dict[str, str | bool]
    experts: list[ExpertResult]
    total_experts: int
