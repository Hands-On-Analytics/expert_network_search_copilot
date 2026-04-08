from collections import defaultdict

from app.models.schemas import (
    AssembledCandidateRecord,
    CandidateEducationRow,
    CandidateLanguageRow,
    CandidateProfileRow,
    CandidateSkillRow,
    CandidateWorkRow,
)


class CandidateAssembler:
    @staticmethod
    def assemble(
        profile_rows: list[dict],
        language_rows: list[dict],
        education_rows: list[dict],
        skill_rows: list[dict],
        work_rows: list[dict],
    ) -> list[AssembledCandidateRecord]:
        languages_by_candidate = defaultdict(list)
        education_by_candidate = defaultdict(list)
        skills_by_candidate = defaultdict(list)
        work_by_candidate = defaultdict(list)

        for row in language_rows:
            item = CandidateLanguageRow(**row)
            languages_by_candidate[item.candidate_id].append(item)

        for row in education_rows:
            item = CandidateEducationRow(**row)
            education_by_candidate[item.candidate_id].append(item)

        for row in skill_rows:
            item = CandidateSkillRow(**row)
            skills_by_candidate[item.candidate_id].append(item)

        for row in work_rows:
            item = CandidateWorkRow(**row)
            work_by_candidate[item.candidate_id].append(item)

        assembled = []
        for row in profile_rows:
            profile = CandidateProfileRow(**row)
            assembled.append(
                AssembledCandidateRecord(
                    candidate_id=profile.candidate_id,
                    profile=profile,
                    languages=languages_by_candidate.get(profile.candidate_id, []),
                    education=education_by_candidate.get(profile.candidate_id, []),
                    skills=skills_by_candidate.get(profile.candidate_id, []),
                    work_experiences=work_by_candidate.get(profile.candidate_id, []),
                )
            )

        return assembled