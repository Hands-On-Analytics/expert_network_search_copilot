from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from uuid import UUID


@dataclass(slots=True)
class SkillRecord:
    name: str
    category: str | None = None
    years_of_experience: int | None = None
    proficiency_level: str | None = None


@dataclass(slots=True)
class LanguageRecord:
    name: str
    proficiency_level: str | None = None


@dataclass(slots=True)
class EducationRecord:
    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    start_year: int | None = None
    graduation_year: int | None = None
    grade: str | None = None


@dataclass(slots=True)
class WorkExperienceRecord:
    company: str | None = None
    job_title: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    is_current: bool | None = None
    description: str | None = None


@dataclass(slots=True)
class CandidateProfile:
    candidate_id: UUID
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    phone: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None
    nationality: str | None = None
    city: str | None = None
    country: str | None = None
    headline: str | None = None
    years_of_experience: int | None = None
    skills: list[SkillRecord] = field(default_factory=list)
    languages: list[LanguageRecord] = field(default_factory=list)
    education: list[EducationRecord] = field(default_factory=list)
    work_experiences: list[WorkExperienceRecord] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        first = (self.first_name or "").strip()
        last = (self.last_name or "").strip()
        name = f"{first} {last}".strip()
        return name or "Unknown Candidate"


@dataclass(slots=True)
class ProfileChunk:
    chunk_id: str
    candidate_id: str
    chunk_type: str
    text: str
    metadata: dict[str, str | int | float | bool]
