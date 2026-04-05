from __future__ import annotations

from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from candidate_ingest_search.types import (
    CandidateProfile,
    EducationRecord,
    LanguageRecord,
    SkillRecord,
    WorkExperienceRecord,
)


class CandidateRepository:
    def __init__(self, postgres_url: str) -> None:
        self._postgres_url = postgres_url

    def fetch_candidate_profiles(self, limit: int | None = None) -> list[CandidateProfile]:
        with psycopg.connect(self._postgres_url, row_factory=dict_row) as conn:
            candidates = self._fetch_candidates(conn, limit)
            if not candidates:
                return []

            profiles = {
                row["id"]: CandidateProfile(
                    candidate_id=row["id"],
                    first_name=row["first_name"],
                    last_name=row["last_name"],
                    email=row["email"],
                    phone=row["phone"],
                    date_of_birth=row["date_of_birth"],
                    gender=row["gender"],
                    nationality=row["nationality"],
                    city=row["city"],
                    country=row["country"],
                    headline=row["headline"],
                    years_of_experience=row["years_of_experience"],
                )
                for row in candidates
            }

            candidate_ids = list(profiles.keys())
            self._load_skills(conn, candidate_ids, profiles)
            self._load_languages(conn, candidate_ids, profiles)
            self._load_education(conn, candidate_ids, profiles)
            self._load_work_experience(conn, candidate_ids, profiles)

        return list(profiles.values())

    def _fetch_candidates(self, conn: psycopg.Connection, limit: int | None) -> list[dict]:
        query = """
            SELECT
                c.id,
                c.first_name,
                c.last_name,
                c.email,
                c.phone,
                c.date_of_birth,
                c.gender,
                c.headline,
                c.years_of_experience,
                nat.name AS nationality,
                city.name AS city,
                country.name AS country
            FROM candidates c
            LEFT JOIN countries nat ON nat.id = c.nationality_id
            LEFT JOIN cities city ON city.id = c.city_id
            LEFT JOIN countries country ON country.id = city.country_id
            ORDER BY c.created_at DESC
        """

        with conn.cursor() as cur:
            if limit is None:
                cur.execute(query)
            else:
                cur.execute(f"{query} LIMIT %s", (limit,))
            return list(cur.fetchall())

    def _load_skills(
        self,
        conn: psycopg.Connection,
        candidate_ids: list[UUID],
        profiles: dict[UUID, CandidateProfile],
    ) -> None:
        query = """
            SELECT
                cs.candidate_id,
                s.name AS skill_name,
                sc.name AS category_name,
                cs.years_of_experience,
                cs.proficiency_level
            FROM candidate_skills cs
            INNER JOIN skills s ON s.id = cs.skill_id
            LEFT JOIN skill_categories sc ON sc.id = s.category_id
            WHERE cs.candidate_id = ANY(%s)
            ORDER BY cs.candidate_id, cs.years_of_experience DESC NULLS LAST, s.name
        """
        with conn.cursor() as cur:
            cur.execute(query, (candidate_ids,))
            for row in cur.fetchall():
                profile = profiles[row["candidate_id"]]
                profile.skills.append(
                    SkillRecord(
                        name=row["skill_name"],
                        category=row["category_name"],
                        years_of_experience=row["years_of_experience"],
                        proficiency_level=row["proficiency_level"],
                    )
                )

    def _load_languages(
        self,
        conn: psycopg.Connection,
        candidate_ids: list[UUID],
        profiles: dict[UUID, CandidateProfile],
    ) -> None:
        query = """
            SELECT
                cl.candidate_id,
                l.name AS language_name,
                pl.name AS proficiency_name
            FROM candidate_languages cl
            INNER JOIN languages l ON l.id = cl.language_id
            LEFT JOIN proficiency_levels pl ON pl.id = cl.proficiency_level_id
            WHERE cl.candidate_id = ANY(%s)
            ORDER BY cl.candidate_id, l.name
        """
        with conn.cursor() as cur:
            cur.execute(query, (candidate_ids,))
            for row in cur.fetchall():
                profile = profiles[row["candidate_id"]]
                profile.languages.append(
                    LanguageRecord(
                        name=row["language_name"],
                        proficiency_level=row["proficiency_name"],
                    )
                )

    def _load_education(
        self,
        conn: psycopg.Connection,
        candidate_ids: list[UUID],
        profiles: dict[UUID, CandidateProfile],
    ) -> None:
        query = """
            SELECT
                e.candidate_id,
                i.name AS institution_name,
                d.name AS degree_name,
                fos.name AS field_of_study_name,
                e.start_year,
                e.graduation_year,
                e.grade
            FROM education e
            LEFT JOIN institutions i ON i.id = e.institution_id
            LEFT JOIN degrees d ON d.id = e.degree_id
            LEFT JOIN fields_of_study fos ON fos.id = e.field_of_study_id
            WHERE e.candidate_id = ANY(%s)
            ORDER BY e.candidate_id, e.graduation_year DESC NULLS LAST
        """
        with conn.cursor() as cur:
            cur.execute(query, (candidate_ids,))
            for row in cur.fetchall():
                profile = profiles[row["candidate_id"]]
                profile.education.append(
                    EducationRecord(
                        institution=row["institution_name"],
                        degree=row["degree_name"],
                        field_of_study=row["field_of_study_name"],
                        start_year=row["start_year"],
                        graduation_year=row["graduation_year"],
                        grade=row["grade"],
                    )
                )

    def _load_work_experience(
        self,
        conn: psycopg.Connection,
        candidate_ids: list[UUID],
        profiles: dict[UUID, CandidateProfile],
    ) -> None:
        query = """
            SELECT
                w.candidate_id,
                comp.name AS company_name,
                w.job_title,
                w.start_date,
                w.end_date,
                w.is_current,
                w.description
            FROM work_experience w
            LEFT JOIN companies comp ON comp.id = w.company_id
            WHERE w.candidate_id = ANY(%s)
            ORDER BY w.candidate_id, w.start_date DESC NULLS LAST
        """
        with conn.cursor() as cur:
            cur.execute(query, (candidate_ids,))
            for row in cur.fetchall():
                profile = profiles[row["candidate_id"]]
                profile.work_experiences.append(
                    WorkExperienceRecord(
                        company=row["company_name"],
                        job_title=row["job_title"],
                        start_date=row["start_date"],
                        end_date=row["end_date"],
                        is_current=row["is_current"],
                        description=row["description"],
                    )
                )
