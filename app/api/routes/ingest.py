from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import traceback

from app.api.deps import get_checkpoint_service, get_qdrant_service
from app.db.postgres import get_db_session
from app.models.schemas import IngestRequest, IngestResponse
from app.services.checkpoint_service import CheckpointService
from app.services.ingest_service import IngestService
from app.services.qdrant_service import QdrantService

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/candidates", response_model=IngestResponse)
async def ingest_candidates(
    payload: IngestRequest,
    session: AsyncSession = Depends(get_db_session),
    qdrant_service: QdrantService = Depends(get_qdrant_service),
    checkpoint_service: CheckpointService = Depends(get_checkpoint_service),
) -> IngestResponse:
    try:
        service = IngestService(
            qdrant_service=qdrant_service,
            checkpoint_service=checkpoint_service,
        )

        result = await service.run(
            session=session,
            full_reindex=payload.full_reindex,
            dry_run=payload.dry_run,
            limit=payload.limit,
        )
        return IngestResponse(**result)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))