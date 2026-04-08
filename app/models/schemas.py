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


class AssembledCandidateRecord(BaseModel):
    candidate_id: UUID
    profile: CandidateProfileRow
    languages: list[CandidateLanguageRow] = []
    education: list[CandidateEducationRow] = []
    skills: list[CandidateSkillRow] = []
    work_experiences: list[CandidateWorkRow] = []