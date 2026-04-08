from uuid import UUID, uuid5, NAMESPACE_DNS

from langchain_core.documents import Document
from app.models.schemas import AssembledCandidateRecord
from app.utils.hashing import stable_hash

_NAMESPACE = NAMESPACE_DNS


def _point_id(candidate_id: UUID, doc_type: str) -> str:
    """Generate a deterministic UUID suitable as a Qdrant point ID."""
    return str(uuid5(_NAMESPACE, f"candidate:{candidate_id}:{doc_type}"))


class CandidateDocumentBuilder:
    @staticmethod
    def build_documents(candidate: AssembledCandidateRecord) -> list[Document]:
        profile = candidate.profile
        base_payload = candidate.model_dump(mode="json")
        content_hash = stable_hash(base_payload)

        full_name = " ".join(
            part for part in [profile.first_name, profile.last_name] if part
        ).strip()

        profile_text = f"""
Candidate ID: {candidate.candidate_id}
Full name: {full_name}
Headline: {profile.candidate_headline or ""}
Gender: {profile.gender or ""}
Years of experience: {profile.years_of_experience or ""}
City: {profile.city_name or ""}
Country: {profile.country_name or ""}
Country code: {profile.country_code or ""}
Email: {profile.email or ""}
Phone: {profile.phone or ""}
Created at: {profile.created_at or ""}
""".strip()

        languages_text = "\n".join(
            f"Language: {item.language or ''}, Proficiency: {item.language_proficiency_level or ''}"
            for item in candidate.languages
        ).strip()

        education_text = "\n".join(
            f"Institution: {item.institution_name or ''}, Degree: {item.degree or ''}, Field: {item.field_of_study or ''}, Start year: {item.start_year or ''}, Graduation year: {item.graduation_year or ''}"
            for item in candidate.education
        ).strip()

        skills_text = "\n".join(
            f"Skill: {item.skill_name or ''}, Category: {item.skill_category or ''}, Years: {item.skill_years_of_experience or ''}, Proficiency: {item.skill_proficiency_level or ''}"
            for item in candidate.skills
        ).strip()

        work_text = "\n".join(
            f"Industry: {item.industry or ''}, Country: {item.country_name or ''}, Job title: {item.job_title or ''}, Start date: {item.start_date or ''}, End date: {item.end_date or ''}, Current flag: {item.is_current_flag or 0}, Description: {item.workexperience_description or ''}"
            for item in candidate.work_experiences
        ).strip()

        common_metadata = {
            "candidate_id": str(candidate.candidate_id),
            "full_name": full_name,
            "country_name": profile.country_name,
            "country_code": profile.country_code,
            "city_name": profile.city_name,
            "years_of_experience": profile.years_of_experience,
            "content_hash": content_hash,
            "full_candidate_record": base_payload,
        }

        docs = [
            Document(
                page_content=profile_text,
                metadata={
                    **common_metadata,
                    "doc_type": "profile",
                    "point_id": _point_id(candidate.candidate_id, "profile"),
                },
            )
        ]

        if languages_text:
            docs.append(
                Document(
                    page_content=languages_text,
                    metadata={
                        **common_metadata,
                        "doc_type": "languages",
                        "point_id": _point_id(candidate.candidate_id, "languages"),
                    },
                )
            )

        if education_text:
            docs.append(
                Document(
                    page_content=education_text,
                    metadata={
                        **common_metadata,
                        "doc_type": "education",
                        "point_id": _point_id(candidate.candidate_id, "education"),
                    },
                )
            )

        if skills_text:
            docs.append(
                Document(
                    page_content=skills_text,
                    metadata={
                        **common_metadata,
                        "doc_type": "skills",
                        "point_id": _point_id(candidate.candidate_id, "skills"),
                    },
                )
            )

        if work_text:
            docs.append(
                Document(
                    page_content=work_text,
                    metadata={
                        **common_metadata,
                        "doc_type": "work_experience",
                        "point_id": _point_id(candidate.candidate_id, "work_experience"),
                    },
                )
            )

        return docs