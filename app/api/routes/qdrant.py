from fastapi import APIRouter, Depends

from app.api.deps import get_qdrant_service
from app.services.qdrant_service import QdrantService

router = APIRouter(prefix="/qdrant", tags=["qdrant"])


@router.get("/collections")
async def list_collections(
    qdrant_service: QdrantService = Depends(get_qdrant_service),
) -> dict:
    collections = qdrant_service.list_collections()
    return {
        "total_collections": len(collections),
        "collections": collections,
    }
