from uuid import UUID, uuid5, NAMESPACE_DNS

from langchain_core.documents import Document
from app.models.schemas import AssembledCandidateRecord
from app.utils.hashing import stable_hash

_NAMESPACE = NAMESPACE_DNS


def _point_id(candidate_id: UUID) -> str:
    """Generate a deterministic UUID suitable as a Qdrant point ID."""
    return str(uuid5(_NAMESPACE, f"candidate:{candidate_id}"))


class CandidateDocumentBuilder:
    @staticmethod
    def build_documents(candidate: AssembledCandidateRecord) -> list[Document]:
        """Build a single Document per candidate with a rich page_content."""
        profile = candidate.profile
        base_payload = candidate.model_dump(mode="json")
        content_hash = stable_hash(base_payload)

        full_name = " ".join(
            part for part in [profile.first_name, profile.last_name] if part
        ).strip()

        # ── headline + location + experience ──────────────────────────
        sections: list[str] = []

        headline = profile.candidate_headline or full_name
        location_parts = [p for p in [profile.city_name, profile.country_name] if p]
        location = ", ".join(location_parts) if location_parts else "Unknown location"
        yoe = profile.years_of_experience

        sections.append(
            f"{full_name} \u2014 {headline}.\n"
            f"Located in {location}."
            + (f" {yoe} years of professional experience." if yoe else "")
        )

        # ── work experience ───────────────────────────────────────────
        if candidate.work_experiences:
            work_lines: list[str] = []
            for w in candidate.work_experiences:
                current = "Current" if w.is_current_flag else "Past"
                title = w.job_title or "Unknown role"
                industry = f" in {w.industry}" if w.industry else ""
                period = f" (since {w.start_date})" if w.start_date else ""
                desc = f"\n{w.workexperience_description}" if w.workexperience_description else ""
                work_lines.append(f"{current} Role: {title}{industry}{period}.{desc}")
            sections.append("\n".join(work_lines))

        # ── skills ────────────────────────────────────────────────────
        if candidate.skills:
            skill_parts = []
            for s in candidate.skills:
                name = s.skill_name or "Unknown"
                cat = f"{s.skill_category}, " if s.skill_category else ""
                yrs = f"{s.skill_years_of_experience} yrs, " if s.skill_years_of_experience else ""
                prof = s.skill_proficiency_level or ""
                skill_parts.append(f"{name} ({cat}{yrs}{prof})")
            sections.append("Skills: " + ", ".join(skill_parts) + ".")

        # ── education ─────────────────────────────────────────────────
        if candidate.education:
            edu_parts = []
            for e in candidate.education:
                degree = e.degree or ""
                field = f" in {e.field_of_study}" if e.field_of_study else ""
                inst = f" from {e.institution_name}" if e.institution_name else ""
                years = ""
                if e.start_year and e.graduation_year:
                    years = f" ({e.start_year}\u2013{e.graduation_year})"
                elif e.graduation_year:
                    years = f" ({e.graduation_year})"
                edu_parts.append(f"{degree}{field}{inst}{years}")
            sections.append("Education: " + "; ".join(edu_parts) + ".")

        # ── languages ─────────────────────────────────────────────────
        if candidate.languages:
            lang_parts = [
                f"{l.language} ({l.language_proficiency_level})"
                if l.language_proficiency_level
                else (l.language or "")
                for l in candidate.languages
            ]
            sections.append("Languages: " + ", ".join(lang_parts) + ".")

        page_content = "\n\n".join(sections)

        # ── lean metadata (filter + identification fields only) ───────
        metadata = {
            "candidate_id": str(candidate.candidate_id),
            "full_name": full_name,
            "candidate_headline": profile.candidate_headline,
            "years_of_experience": profile.years_of_experience,
            "country_name": profile.country_name,
            "country_code": profile.country_code,
            "city_name": profile.city_name,
            "gender": profile.gender,
            "content_hash": content_hash,
            "point_id": _point_id(candidate.candidate_id),
        }

        return [
            Document(page_content=page_content, metadata=metadata)
        ]
