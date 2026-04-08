from app.services.qdrant_service import QdrantService
from app.services.checkpoint_service import CheckpointService

qdrant_service = QdrantService()
checkpoint_service = CheckpointService()

def get_qdrant_service() -> QdrantService:
    return qdrant_service

def get_checkpoint_service() -> CheckpointService:
    return checkpoint_service