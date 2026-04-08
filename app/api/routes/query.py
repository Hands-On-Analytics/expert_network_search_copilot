from fastapi import APIRouter, Depends, HTTPException
import traceback

from app.api.deps import get_qdrant_service
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services.qdrant_service import QdrantService
from app.services.rag_graph import build_rag_graph
from app.core.config import settings

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_candidates(
    payload: QueryRequest,
    qdrant_service: QdrantService = Depends(get_qdrant_service),
) -> QueryResponse:
    """Run the RAG pipeline: retrieve relevant candidates → generate answer."""
    try:
        # Allow per-request override of top_k
        if payload.top_k is not None:
            original_top_k = settings.RAG_TOP_K
            settings.RAG_TOP_K = payload.top_k

        graph = build_rag_graph(qdrant_service)
        result = graph.invoke({"query": payload.query})

        # Restore default
        if payload.top_k is not None:
            settings.RAG_TOP_K = original_top_k

        sources = [
            SourceChunk(content=doc.page_content, metadata=doc.metadata)
            for doc in result.get("documents", [])
        ]
        return QueryResponse(answer=result["answer"], sources=sources)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
