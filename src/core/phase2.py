"""Phase 2 orchestration: embed, canonicalise hadiths, classify edges."""

from __future__ import annotations

import logging

import numpy as np

from config.settings import Settings, get_settings
from src.agents.embeddings import EmbeddingAgent
from src.agents.errors import AllKeysExhausted
from src.agents.gemini import GeminiAgent
from src.core.edge_classifier import apply_canonical_ids, classify_tag_groups, verify_duplicates
from src.core.vector_engine import dbscan_duplicate_clusters, embed_pending_chunks
from src.state_manager import ChunkStatus, StateManager

logger = logging.getLogger(__name__)


def run_phase2(
    book_id: str,
    state: StateManager,
    gemini: GeminiAgent,
    embeddings: EmbeddingAgent,
    settings: Settings | None = None,
) -> dict[str, int]:
    settings = settings or get_settings()
    job_id = state.record_job("phase2", book_id)
    try:
        embedded = embed_pending_chunks(state, embeddings, book_id=book_id)
        hadiths = [
            c
            for c in state.list_chunks(
                book_id=book_id,
                pipeline="hadith",
                statuses=[ChunkStatus.EMBEDDED, ChunkStatus.PROCESSED_PHASE2],
            )
        ]
        mapping: dict[str, str] = {}
        if hadiths:
            vectors = state.load_embeddings(embeddings.model, [c.id for c in hadiths])
            ids = [c.id for c in hadiths if c.id in vectors]
            matrix = np.array([vectors[i] for i in ids], dtype=float)
            clusters = dbscan_duplicate_clusters(ids, matrix, settings.dedup_similarity)
            mapping = verify_duplicates(gemini, {c.id: c for c in hadiths}, clusters)
            apply_canonical_ids(state, mapping, hadiths)

        work = state.list_chunks(
            book_id=book_id,
            statuses=[ChunkStatus.EMBEDDED, ChunkStatus.PROCESSED_PHASE2],
        )
        vecs = state.load_embeddings(embeddings.model, [c.id for c in work])
        edges = classify_tag_groups(
            gemini,
            state,
            work,
            vecs,
            threshold=settings.edge_similarity,
            group_cap=settings.edge_group_cap,
        )
    except AllKeysExhausted:
        state.finish_job(job_id, pause_reason="all_keys_exhausted")
        raise
    state.finish_job(job_id)
    logger.info("phase2 book=%s embedded=%s edges=%s", book_id, embedded, edges)
    return {"embedded": embedded, "duplicates": len(mapping), "edges": edges}
