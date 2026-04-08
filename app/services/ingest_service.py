from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.queries import (
    fetch_candidate_profiles,
    fetch_education,
    fetch_languages,
    fetch_skills,
    fetch_work_experiences,
)
from app.services.candidate_assembler import CandidateAssembler
from app.services.document_builder import CandidateDocumentBuilder
from app.services.checkpoint_service import CheckpointService
from app.services.qdrant_service import QdrantService


class IngestService:
    def __init__(
        self,
        qdrant_service: QdrantService,
        checkpoint_service: CheckpointService,
    ) -> None:
        self.qdrant_service = qdrant_service
        self.checkpoint_service = checkpoint_service

    async def run(
        self,
        session: AsyncSession,
        *,
        full_reindex: bool = False,
        dry_run: bool = False,
        limit: int | None = None,
    ) -> dict:
        self.qdrant_service.ensure_collection()
        vectorstore = self.qdrant_service.vectorstore()

        cursor = None if full_reindex else await self.checkpoint_service.get_cursor()
        batch_limit = limit or settings.INGEST_BATCH_SIZE

        profile_rows = await fetch_candidate_profiles(
            session=session,
            cursor_candidate_id=cursor,
            limit=batch_limit,
        )

        if not profile_rows:
            return {
                "status": "success",
                "candidates_processed": 0,
                "points_upserted": 0,
                "cursor_updated_at": None,
            }

        candidate_ids = [row["candidate_id"] for row in profile_rows]

        language_rows = await fetch_languages(session, candidate_ids)
        education_rows = await fetch_education(session, candidate_ids)
        skill_rows = await fetch_skills(session, candidate_ids)
        work_rows = await fetch_work_experiences(session, candidate_ids)

        assembled_candidates = CandidateAssembler.assemble(
            profile_rows=profile_rows,
            language_rows=language_rows,
            education_rows=education_rows,
            skill_rows=skill_rows,
            work_rows=work_rows,
        )

        documents = []
        ids = []

        for candidate in assembled_candidates:
            docs = CandidateDocumentBuilder.build_documents(candidate)
            documents.extend(docs)
            ids.extend([doc.metadata["point_id"] for doc in docs])

        if not dry_run and documents:
            vectorstore.add_documents(documents=documents, ids=ids)

        new_cursor = max(candidate_ids)
        await self.checkpoint_service.set_cursor(new_cursor)

        return {
            "status": "success",
            "candidates_processed": len(assembled_candidates),
            "points_upserted": 0 if dry_run else len(documents),
            "cursor_updated_at": str(new_cursor),
        }