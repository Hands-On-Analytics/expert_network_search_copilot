from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

from candidate_ingest_search.types import ProfileChunk


class ChromaVectorStore:
    def __init__(self, persist_path: Path, collection_name: str) -> None:
        self._persist_path = persist_path
        self._persist_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(self._persist_path))
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    def count(self) -> int:
        return self._collection.count()

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
    ) -> dict[str, Any]:
        return self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def reset_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[ProfileChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count does not match embedding count")

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._sanitize_metadata(chunk.metadata) for chunk in chunks],
            embeddings=embeddings,
        )

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        sanitized: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = json.dumps(value, ensure_ascii=False)
        return sanitized
