from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

from candidate_ingest_search.chunking import CandidateChunkBuilder
from candidate_ingest_search.embeddings import OpenRouterEmbeddingClient
from candidate_ingest_search.repository import CandidateRepository
from candidate_ingest_search.settings import Settings
from candidate_ingest_search.types import ProfileChunk
from candidate_ingest_search.vector_store import ChromaVectorStore


def _batched(items: list[ProfileChunk], batch_size: int) -> Iterator[list[ProfileChunk]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


@dataclass(slots=True)
class IngestionResult:
    candidates_processed: int
    chunks_generated: int
    chunks_upserted: int
    vector_collection: str
    full_refresh: bool
    duration_seconds: float

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "candidates_processed": self.candidates_processed,
            "chunks_generated": self.chunks_generated,
            "chunks_upserted": self.chunks_upserted,
            "vector_collection": self.vector_collection,
            "full_refresh": self.full_refresh,
            "duration_seconds": round(self.duration_seconds, 3),
        }


class CandidateIngestionPipeline:
    def __init__(
        self,
        repository: CandidateRepository,
        chunk_builder: CandidateChunkBuilder,
        embedding_client: OpenRouterEmbeddingClient,
        vector_store: ChromaVectorStore,
        upsert_batch_size: int = 128,
    ) -> None:
        self._repository = repository
        self._chunk_builder = chunk_builder
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._upsert_batch_size = upsert_batch_size

    @classmethod
    def from_settings(cls, settings: Settings) -> CandidateIngestionPipeline:
        repository = CandidateRepository(postgres_url=settings.postgres_url)
        chunk_builder = CandidateChunkBuilder(max_chunk_chars=settings.max_chunk_chars)
        embedding_client = OpenRouterEmbeddingClient(
            api_key=settings.openrouter_api_key,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            base_url=settings.openrouter_base_url,
            site_url=settings.openrouter_site_url,
            app_name=settings.openrouter_app_name,
        )
        vector_store = ChromaVectorStore(
            persist_path=settings.resolved_chroma_path,
            collection_name=settings.chroma_collection_name,
        )
        return cls(
            repository=repository,
            chunk_builder=chunk_builder,
            embedding_client=embedding_client,
            vector_store=vector_store,
            upsert_batch_size=settings.upsert_batch_size,
        )

    def ingest(self, full_refresh: bool = True, limit: int | None = None) -> IngestionResult:
        start = perf_counter()

        if full_refresh:
            self._vector_store.reset_collection()

        profiles = self._repository.fetch_candidate_profiles(limit=limit)
        chunks: list[ProfileChunk] = []
        for profile in profiles:
            chunks.extend(self._chunk_builder.build_chunks(profile))

        upserted = 0
        for batch in _batched(chunks, self._upsert_batch_size):
            embeddings = self._embedding_client.embed_texts([chunk.text for chunk in batch])
            self._vector_store.upsert(chunks=batch, embeddings=embeddings)
            upserted += len(batch)

        duration = perf_counter() - start
        return IngestionResult(
            candidates_processed=len(profiles),
            chunks_generated=len(chunks),
            chunks_upserted=upserted,
            vector_collection=self._vector_store.collection_name,
            full_refresh=full_refresh,
            duration_seconds=duration,
        )
