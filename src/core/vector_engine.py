"""Local vector math used by Phase 2 and reusable by later graph phases."""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN

from src.agents.embeddings import EmbeddingAgent
from src.state_manager import ChunkRecord, ChunkStatus, StateManager


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = vectors / norms
    return normalized @ normalized.T


def pairs_above_threshold(
    ids: list[str],
    vectors: np.ndarray,
    threshold: float,
) -> list[tuple[str, str, float]]:
    if len(ids) < 2:
        return []
    sims = cosine_matrix(vectors)
    pairs: list[tuple[str, str, float]] = []
    n = len(ids)
    for i, j in itertools.combinations(range(n), 2):
        score = float(sims[i, j])
        if score > threshold:
            left, right = sorted((ids[i], ids[j]))
            pairs.append((left, right, score))
    return pairs


def dbscan_duplicate_clusters(
    ids: list[str],
    vectors: np.ndarray,
    similarity: float,
) -> list[list[str]]:
    """Cluster items whose cosine similarity is >= `similarity` (eps = 1 - sim)."""
    if len(ids) < 2:
        return []
    eps = max(1.0 - similarity, 1e-6)
    clustering = DBSCAN(eps=eps, min_samples=2, metric="cosine")
    labels = clustering.fit_predict(vectors)
    buckets: dict[int, list[str]] = defaultdict(list)
    for chunk_id, label in zip(ids, labels):
        if label >= 0:
            buckets[int(label)].append(chunk_id)
    return [group for group in buckets.values() if len(group) >= 2]


def embed_text_for_chunk(chunk: ChunkRecord) -> str:
    payload = chunk.payload() or {}
    if chunk.pipeline == "hadith":
        return str(payload.get("hadith") or chunk.text)
    if chunk.pipeline == "tafsir":
        return str(payload.get("tafsir_chunk") or payload.get("summary_fa") or chunk.text)
    if chunk.pipeline == "history":
        events = payload.get("events") or []
        titles = [e.get("event_title", "") for e in events if isinstance(e, dict)]
        return "\n".join(titles) or chunk.text
    return chunk.text


def concepts_for_chunk(chunk: ChunkRecord) -> list[str]:
    payload = chunk.payload() or {}
    if chunk.pipeline == "hadith":
        return list(payload.get("tags") or [])
    if chunk.pipeline == "tafsir":
        return list(payload.get("core_concepts") or [])
    if chunk.pipeline == "history":
        tags: list[str] = []
        for event in payload.get("events") or []:
            if isinstance(event, dict):
                tags.extend(event.get("historical_concepts") or [])
        return tags
    return []


def embed_pending_chunks(
    state: StateManager,
    agent: EmbeddingAgent,
    book_id: str | None = None,
) -> int:
    chunks = state.list_chunks(
        book_id=book_id,
        statuses=[ChunkStatus.PROCESSED_PHASE1, ChunkStatus.EMBEDDED],
    )
    count = 0
    for chunk in chunks:
        if chunk.status == ChunkStatus.SKIPPED:
            continue
        agent.embed_and_store(chunk.id, embed_text_for_chunk(chunk))
        if chunk.status == ChunkStatus.PROCESSED_PHASE1:
            state.mark(chunk.id, ChunkStatus.EMBEDDED, payload=chunk.payload())
        count += 1
    return count
