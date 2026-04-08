from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

from app.core.config import settings
from app.services.embedding_service import build_embeddings


class QdrantService:
    def __init__(self) -> None:
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=60,
        )
        self.embeddings = build_embeddings()

    def ensure_collection(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if settings.QDRANT_COLLECTION not in collections:
            vector_size = len(self.embeddings.embed_query("candidate skill profile"))
            self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def vectorstore(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=settings.QDRANT_COLLECTION,
            embedding=self.embeddings,
        )