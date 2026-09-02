"""Reusable OpenAI embedding agent.

Phase 2 vectorization and any later semantic search should call `embed()` /
`embed_one()` here so the model name, batching, and API key stay in one place.
"""

from __future__ import annotations

from openai import OpenAI

from config.settings import Settings, get_settings
from src.state_manager import StateManager


class EmbeddingAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        state: StateManager | None = None,
        embed_fn=None,
        client: OpenAI | None = None,
    ):
        self.settings = settings or get_settings()
        self.state = state
        self._embed_fn = embed_fn
        self._client = client

    @property
    def model(self) -> str:
        return self.settings.embedding_model

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._embed_fn is not None:
            return self._embed_fn(texts, self.model)

        if self._client is None:
            if not self.settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for embeddings")
            self._client = OpenAI(api_key=self.settings.openai_api_key)

        out: list[list[float]] = []
        batch = self.settings.embed_batch_size
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            response = self._client.embeddings.create(model=self.model, input=chunk)
            ordered = sorted(response.data, key=lambda item: item.index)
            out.extend(item.embedding for item in ordered)
        return out

    def embed_and_store(self, chunk_id: str, text: str) -> list[float]:
        """Idempotent helper: skip the API if this chunk is already embedded."""
        if self.state:
            cached = self.state.get_embedding(chunk_id, self.model)
            if cached is not None:
                return cached
        vector = self.embed_one(text)
        if self.state:
            self.state.save_embedding(chunk_id, self.model, vector)
        return vector
