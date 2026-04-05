from __future__ import annotations

from collections.abc import Iterator, Sequence

from openai import OpenAI


def _batched(items: Sequence[str], batch_size: int) -> Iterator[list[str]]:
    for index in range(0, len(items), batch_size):
        yield list(items[index : index + batch_size])


class OpenRouterEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        batch_size: int = 64,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str | None = None,
        app_name: str | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=headers if headers else None,
        )
        self._model = model
        self._batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batched(texts, self._batch_size):
            response = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend([item.embedding for item in response.data])
        return vectors
