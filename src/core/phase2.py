"""Phase 2 orchestration: embed, canonicalise hadiths, classify edges."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from config.paths import OUTPUT_DIR
from config.settings import Settings, get_settings
from src.agents.embeddings import EmbeddingAgent
from src.agents.errors import AllKeysExhausted, ProviderServerError
from src.agents.gemini import GeminiAgent
from src.core.edge_classifier import (
    apply_canonical_ids,
    classify_candidate_pairs,
    verify_duplicates,
)
from src.core.vector_engine import dbscan_duplicate_clusters, embed_pending_chunks
from src.pipelines.llm_processor import phase1_filename
from src.pipelines.ontology import remap_hadith_payload
from src.state_manager import ChunkStatus, StateManager

logger = logging.getLogger(__name__)


def remap_existing_semantic_nodes(
    state: StateManager,
    output_dir: Path | None = None,
) -> int:
    """Apply the current alias table and node policy to stored hadith payloads. No Gemini."""
    dest_root = output_dir or (OUTPUT_DIR / "phase1")
    chunks = state.list_chunks(
        pipeline="hadith",
        statuses=[
            ChunkStatus.PROCESSED_PHASE1,
            ChunkStatus.EMBEDDED,
            ChunkStatus.PROCESSED_PHASE2,
        ],
    )
    updated = 0
    for chunk in chunks:
        payload = chunk.payload()
        if not payload:
            continue
        if payload.get("nodes"):
            # Already resolved corpus-wide. Re-running the per-hadith gate over
            # it would overwrite canonical identities with the old alias table,
            # which is exactly the dual-identity problem this pass predates.
            continue
        remapped = remap_hadith_payload(payload)
        if remapped == payload:
            continue
        kwargs = {"payload": remapped}
        if chunk.canonical_id:
            kwargs["canonical_id"] = chunk.canonical_id
        state.mark(chunk.id, chunk.status, **kwargs)
        path = dest_root / chunk.book_id / phase1_filename(
            chunk.source_path, chunk.locator, chunk.id
        )
        if path.exists():
            path.write_text(
                json.dumps(remapped, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        updated += 1
    if updated:
        logger.info("remapped semantic_nodes on %s hadith chunks", updated)
    return updated


def run_phase2(
    book_id: str,
    state: StateManager,
    gemini: GeminiAgent,
    embeddings: EmbeddingAgent,
    settings: Settings | None = None,
) -> dict[str, int]:
    settings = settings or get_settings()
    job_id = state.record_job("phase2", book_id)
    remapped = 0
    mapping: dict[str, str] = {}
    try:
        remapped = remap_existing_semantic_nodes(state)
        embedded = embed_pending_chunks(
            state,
            embeddings,
            book_id=book_id,
            statuses=[ChunkStatus.PROCESSED_PHASE1],
        )
        hadiths = [
            c
            for c in state.list_chunks(
                book_id=book_id,
                pipeline="hadith",
                statuses=[ChunkStatus.EMBEDDED, ChunkStatus.PROCESSED_PHASE2],
            )
        ]
        if hadiths:
            vectors = state.load_embeddings(embeddings.model, [c.id for c in hadiths])
            ids = [c.id for c in hadiths if c.id in vectors]
            matrix = np.array([vectors[i] for i in ids], dtype=float)
            clusters = dbscan_duplicate_clusters(ids, matrix, settings.dedup_similarity)
            mapping = verify_duplicates(gemini, {c.id: c for c in hadiths}, clusters)
            apply_canonical_ids(state, mapping, hadiths)

        work = state.list_chunks(
            statuses=[ChunkStatus.EMBEDDED, ChunkStatus.PROCESSED_PHASE2],
        )
        vecs = state.load_embeddings(embeddings.model, [c.id for c in work])
        edges = classify_candidate_pairs(
            gemini,
            state,
            work,
            vecs,
            max_pairs=settings.edge_max_pairs,
        )
    except AllKeysExhausted:
        state.finish_job(job_id, pause_reason="all_keys_exhausted")
        raise
    except ProviderServerError:
        state.finish_job(job_id, pause_reason="provider_unavailable")
        raise
    state.finish_job(job_id)
    logger.info(
        "phase2 book=%s remapped=%s embedded=%s edges=%s",
        book_id,
        remapped,
        embedded,
        edges,
    )
    return {
        "remapped": remapped,
        "embedded": embedded,
        "duplicates": len(mapping),
        "edges": edges,
    }
