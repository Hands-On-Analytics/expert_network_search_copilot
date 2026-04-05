from __future__ import annotations

from datetime import date
from typing import Iterable

from candidate_ingest_search.types import CandidateProfile, ProfileChunk


def _coalesce(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _isoformat(value: date | None) -> str | None:
    return value.isoformat() if value else None


class CandidateChunkBuilder:
    def __init__(self, max_chunk_chars: int = 1800) -> None:
        self._max_chunk_chars = max_chunk_chars

    def build_chunks(self, profile: CandidateProfile) -> list[ProfileChunk]:
        base_metadata = self._build_base_metadata(profile)
        chunks: list[ProfileChunk] = []

        summary_text = self._build_summary_text(profile)
        chunks.extend(
            self._build_chunk_records(
                profile=profile,
                chunk_key="profile_summary",
                chunk_type="profile_summary",
                text=summary_text,
                metadata=base_metadata,
            )
        )

        if profile.skills:
            skills_text = self._build_skills_text(profile)
            chunks.extend(
                self._build_chunk_records(
                    profile=profile,
                    chunk_key="skills",
                    chunk_type="skills",
                    text=skills_text,
                    metadata={
                        **base_metadata,
                        "skill_count": len(profile.skills),
                        "skill_names": ", ".join(sorted({s.name for s in profile.skills})),
                    },
                )
            )

        if profile.languages:
            languages_text = self._build_languages_text(profile)
            chunks.extend(
                self._build_chunk_records(
                    profile=profile,
                    chunk_key="languages",
                    chunk_type="languages",
                    text=languages_text,
                    metadata={
                        **base_metadata,
                        "language_count": len(profile.languages),
                        "language_names": ", ".join(sorted({l.name for l in profile.languages})),
                    },
                )
            )

        for index, education in enumerate(profile.education, start=1):
            education_lines = [
                f"Education record for {profile.full_name}",
                f"Institution: {education.institution or 'Not specified'}",
                f"Degree: {education.degree or 'Not specified'}",
                f"Field of study: {education.field_of_study or 'Not specified'}",
                f"Start year: {education.start_year if education.start_year is not None else 'Not specified'}",
                f"Graduation year: {education.graduation_year if education.graduation_year is not None else 'Not specified'}",
                f"Grade: {education.grade or 'Not specified'}",
            ]
            chunks.extend(
                self._build_chunk_records(
                    profile=profile,
                    chunk_key=f"education_{index}",
                    chunk_type="education",
                    text="\n".join(education_lines),
                    metadata={
                        **base_metadata,
                        "education_index": index,
                        "institution": education.institution or "",
                        "degree": education.degree or "",
                        "field_of_study": education.field_of_study or "",
                        "start_year": education.start_year or -1,
                        "graduation_year": education.graduation_year or -1,
                    },
                )
            )

        for index, work in enumerate(profile.work_experiences, start=1):
            work_lines = [
                f"Work experience entry for {profile.full_name}",
                f"Company: {work.company or 'Not specified'}",
                f"Role title: {work.job_title or 'Not specified'}",
                f"Start date: {_isoformat(work.start_date) or 'Not specified'}",
                f"End date: {_isoformat(work.end_date) or ('Present' if work.is_current else 'Not specified')}",
                f"Current role: {'Yes' if work.is_current else 'No'}",
            ]
            if work.description:
                work_lines.append(f"Role description: {work.description}")

            chunks.extend(
                self._build_chunk_records(
                    profile=profile,
                    chunk_key=f"work_experience_{index}",
                    chunk_type="work_experience",
                    text="\n".join(work_lines),
                    metadata={
                        **base_metadata,
                        "work_experience_index": index,
                        "company": work.company or "",
                        "job_title": work.job_title or "",
                        "is_current_role": bool(work.is_current),
                        "start_date": _isoformat(work.start_date) or "",
                        "end_date": _isoformat(work.end_date) or "",
                    },
                )
            )

        return chunks

    def _build_summary_text(self, profile: CandidateProfile) -> str:
        lines = [
            f"Candidate profile summary: {profile.full_name}",
            f"Headline: {profile.headline or 'Not specified'}",
            f"Years of experience: {profile.years_of_experience if profile.years_of_experience is not None else 'Not specified'}",
            f"Location: {self._location(profile)}",
            f"Nationality: {profile.nationality or 'Not specified'}",
            f"Gender: {profile.gender or 'Not specified'}",
            f"Date of birth: {_isoformat(profile.date_of_birth) or 'Not specified'}",
            f"Email: {profile.email or 'Not specified'}",
            f"Phone: {profile.phone or 'Not specified'}",
        ]
        return "\n".join(lines)

    def _build_skills_text(self, profile: CandidateProfile) -> str:
        lines = [f"Skills profile for {profile.full_name}"]
        for skill in profile.skills:
            lines.append(
                "- "
                + f"Skill: {skill.name}; "
                + f"Category: {skill.category or 'Unknown'}; "
                + f"Years: {skill.years_of_experience if skill.years_of_experience is not None else 'Unknown'}; "
                + f"Proficiency: {skill.proficiency_level or 'Unknown'}"
            )
        return "\n".join(lines)

    def _build_languages_text(self, profile: CandidateProfile) -> str:
        lines = [f"Languages profile for {profile.full_name}"]
        for language in profile.languages:
            lines.append(
                "- "
                + f"Language: {language.name}; "
                + f"Proficiency: {language.proficiency_level or 'Unknown'}"
            )
        return "\n".join(lines)

    def _build_base_metadata(self, profile: CandidateProfile) -> dict[str, str | int | float | bool]:
        return {
            "candidate_id": str(profile.candidate_id),
            "full_name": profile.full_name,
            "headline": profile.headline or "",
            "city": profile.city or "",
            "country": profile.country or "",
            "nationality": profile.nationality or "",
            "years_of_experience": profile.years_of_experience or 0,
            "skill_count": len(profile.skills),
            "language_count": len(profile.languages),
            "education_count": len(profile.education),
            "work_experience_count": len(profile.work_experiences),
        }

    def _location(self, profile: CandidateProfile) -> str:
        parts = [_coalesce(profile.city), _coalesce(profile.country)]
        normalized = [part for part in parts if part]
        if not normalized:
            return "Not specified"
        return ", ".join(normalized)

    def _build_chunk_records(
        self,
        profile: CandidateProfile,
        chunk_key: str,
        chunk_type: str,
        text: str,
        metadata: dict[str, str | int | float | bool],
    ) -> list[ProfileChunk]:
        records: list[ProfileChunk] = []
        for part_index, part in enumerate(self._split_text(text), start=1):
            records.append(
                ProfileChunk(
                    chunk_id=f"{profile.candidate_id}:{chunk_key}:{part_index}",
                    candidate_id=str(profile.candidate_id),
                    chunk_type=chunk_type,
                    text=part,
                    metadata={
                        **metadata,
                        "chunk_type": chunk_type,
                        "chunk_part_index": part_index,
                    },
                )
            )
        return records

    def _split_text(self, text: str) -> Iterable[str]:
        if len(text) <= self._max_chunk_chars:
            return [text]

        lines = text.splitlines()
        chunks: list[str] = []
        current_lines: list[str] = []
        current_size = 0

        for line in lines:
            line_size = len(line) + (1 if current_lines else 0)
            if current_lines and current_size + line_size > self._max_chunk_chars:
                chunks.append("\n".join(current_lines))
                current_lines = [line]
                current_size = len(line)
            else:
                current_lines.append(line)
                current_size += line_size

        if current_lines:
            chunks.append("\n".join(current_lines))

        return chunks
