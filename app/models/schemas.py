from uuid import UUID

from pydantic import BaseModel, Field
from typing import Any


class IngestRequest(BaseModel):
    full_reindex: bool = False
    limit: int | None = Field(default=None, ge=1)
    dry_run: bool = False


class IngestResponse(BaseModel):
    status: str
    candidates_processed: int
    points_upserted: int
    cursor_updated_at: str | None = None


class CandidateProfileRow(BaseModel):
    candidate_id: UUID
    first_name: str | None = None
    last_name: str | None = None
    gender: str | None = None
    date_of_birth: Any | None = None
    email: str | None = None
    candidate_headline: str | None = None
    phone: str | None = None
    years_of_experience: int | None = None
    city_name: str | None = None
    country_name: str | None = None
    country_code: str | None = None
    created_at: Any | None = None


class CandidateLanguageRow(BaseModel):
    candidate_id: UUID
    language: str | None = None
    language_proficiency_level: str | None = None


class CandidateEducationRow(BaseModel):
    candidate_id: UUID
    institution_name: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start_year: int | None = None
    graduation_year: int | None = None


class CandidateSkillRow(BaseModel):
    candidate_id: UUID
    skill_name: str | None = None
    skill_category: str | None = None
    skill_years_of_experience: int | None = None
    skill_proficiency_level: str | None = None


class CandidateWorkRow(BaseModel):
    candidate_id: UUID
    industry: str | None = None
    country_name: str | None = None
    job_title: str | None = None
    start_date: Any | None = None
    end_date: Any | None = None
    is_current_flag: int | None = None
    workexperience_description: str | None = None


class ExtractedFilters(BaseModel):
    """Structured filters extracted from a natural-language query."""
    country_name: str | None = Field(default=None, description="Full country name (e.g. 'United States', 'United Kingdom', 'India')")
    country_code: str | None = Field(default=None, description="ISO 3166-1 alpha-2 country code (e.g. 'US', 'GB', 'IN')")
    city_name: str | None = Field(default=None, description="City to filter candidates by")
    gender: str | None = Field(default=None, description="Gender filter (e.g. Male, Female)")
    min_years_of_experience: int | None = Field(default=None, ge=0, description="Minimum years of experience")
    max_years_of_experience: int | None = Field(default=None, ge=0, description="Maximum years of experience")
    keywords: str | None = Field(default=None, description="Remaining semantic search query after extracting filters")


class GradeResult(BaseModel):
    """Binary relevance grade for a retrieved document."""
    relevant: bool = Field(description="Whether the document is relevant to the query")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language question")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Override number of retrieved chunks")


class SourceChunk(BaseModel):
    content: str
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = []


class AssembledCandidateRecord(BaseModel):
    candidate_id: UUID
    profile: CandidateProfileRow
    languages: list[CandidateLanguageRow] = []
    education: list[CandidateEducationRow] = []
    skills: list[CandidateSkillRow] = []
    work_experiences: list[CandidateWorkRow] = []