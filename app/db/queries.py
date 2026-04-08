from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession


_CANDIDATE_PROFILE_BASE = """
select
    c.id as candidate_id,
    c.first_name,
    c.last_name,
    c.gender,
    c.date_of_birth,
    c.email,
    c.headline as candidate_headline,
    c.phone,
    c.years_of_experience,
    c2.name as city_name,
    c3.name as country_name,
    c3.code as country_code,
    c.created_at
from candidates c
left join cities c2 on c.city_id = c2.id
left join countries c3 on c2.country_id = c3.id
{where_clause}
order by c.id
limit :limit
"""

LANGUAGES_SQL = text("""
select
    cl.candidate_id,
    l.name as language,
    pl.name as language_proficiency_level
from candidate_languages cl
left join languages l on cl.language_id = l.id
left join proficiency_levels pl on pl.id = cl.proficiency_level_id
where cl.candidate_id in :candidate_ids
""").bindparams(bindparam("candidate_ids", expanding=True))

EDUCATION_SQL = text("""
select
    e.candidate_id,
    i.name as institution_name,
    d.name as degree,
    fos.name as field_of_study,
    e.start_year,
    e.graduation_year
from education e
left join institutions i on e.institution_id = i.id
left join degrees d on e.degree_id = d.id
left join fields_of_study fos on e.field_of_study_id = fos.id
where e.candidate_id in :candidate_ids
""").bindparams(bindparam("candidate_ids", expanding=True))

SKILLS_SQL = text("""
select
    cs.candidate_id,
    s.name as skill_name,
    sc.name as skill_category,
    cs.years_of_experience as skill_years_of_experience,
    cs.proficiency_level as skill_proficiency_level
from candidate_skills cs
left join skills s on cs.skill_id = s.id
left join skill_categories sc on sc.id = s.category_id
where cs.candidate_id in :candidate_ids
""").bindparams(bindparam("candidate_ids", expanding=True))

WORK_SQL = text("""
select
    we.candidate_id,
    c.industry,
    c2.name as country_name,
    we.job_title,
    we.start_date,
    we.end_date,
    case when we.is_current is true then 1 else 0 end as is_current_flag,
    we.description as workexperience_description
from work_experience we
left join companies c on we.company_id = c.id
left join countries c2 on c.country_id = c2.id
where we.candidate_id in :candidate_ids
""").bindparams(bindparam("candidate_ids", expanding=True))


async def fetch_candidate_profiles(
    session: AsyncSession,
    cursor_candidate_id: object | None,
    limit: int,
) -> list[dict]:
    if cursor_candidate_id is not None:
        sql = text(_CANDIDATE_PROFILE_BASE.format(where_clause="where c.id > :cursor_candidate_id"))
        params = {"cursor_candidate_id": cursor_candidate_id, "limit": limit}
    else:
        sql = text(_CANDIDATE_PROFILE_BASE.format(where_clause=""))
        params = {"limit": limit}
    result = await session.execute(sql, params)
    return [dict(row._mapping) for row in result.fetchall()]


async def fetch_languages(
    session: AsyncSession,
    candidate_ids: list[int],
) -> list[dict]:
    if not candidate_ids:
        return []
    result = await session.execute(LANGUAGES_SQL, {"candidate_ids": candidate_ids})
    return [dict(row._mapping) for row in result.fetchall()]


async def fetch_education(
    session: AsyncSession,
    candidate_ids: list[int],
) -> list[dict]:
    if not candidate_ids:
        return []
    result = await session.execute(EDUCATION_SQL, {"candidate_ids": candidate_ids})
    return [dict(row._mapping) for row in result.fetchall()]


async def fetch_skills(
    session: AsyncSession,
    candidate_ids: list[int],
) -> list[dict]:
    if not candidate_ids:
        return []
    result = await session.execute(SKILLS_SQL, {"candidate_ids": candidate_ids})
    return [dict(row._mapping) for row in result.fetchall()]


async def fetch_work_experiences(
    session: AsyncSession,
    candidate_ids: list[int],
) -> list[dict]:
    if not candidate_ids:
        return []
    result = await session.execute(WORK_SQL, {"candidate_ids": candidate_ids})
    return [dict(row._mapping) for row in result.fetchall()]