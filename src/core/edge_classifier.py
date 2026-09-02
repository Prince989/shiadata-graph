"""Gemini classification of SUPPORTS / CONTRADICTS / EXCEPTS between texts."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

import numpy as np

from src.agents.gemini import GeminiAgent
from src.core.vector_engine import concepts_for_chunk, pairs_above_threshold
from src.models import DuplicateVerdict, EdgeRelation
from src.state_manager import ChunkRecord, ChunkStatus, StateManager

logger = logging.getLogger(__name__)

EDGE_SYSTEM = (
    "Determine the logical relationship between these two Shia textual units. "
    "SUPPORTS: they agree or one corroborates the other. "
    "CONTRADICTS: they cannot both be acted on as stated. "
    "EXCEPTS: one restricts, qualifies, or carves an exception from the other. "
    "UNRELATED: no legal or conceptual relation. "
    "Judge only the provided text."
)

DUP_SYSTEM = (
    "Are these two Arabic hadiths the same narration (same meaning, possibly "
    "different wording or chain)? Answer duplicate=true only for semantic duplicates."
)


def pair_id(left: str, right: str) -> str:
    a, b = sorted((left, right))
    return hashlib.sha256(f"{a}|{b}".encode()).hexdigest()


def verify_duplicates(
    agent: GeminiAgent,
    chunks: dict[str, ChunkRecord],
    clusters: list[list[str]],
) -> dict[str, str]:
    """Return map child_id -> canonical_id for verified duplicate clusters."""
    canonical: dict[str, str] = {}
    for group in clusters:
        records = [chunks[cid] for cid in group if cid in chunks]
        if len(records) < 2:
            continue
        records.sort(key=lambda r: len(embed_preview(r)), reverse=True)
        root = records[0]
        for other in records[1:]:
            verdict = agent.complete_structured(
                f"A:\n{embed_preview(root)}\n\nB:\n{embed_preview(other)}",
                DuplicateVerdict,
                system=DUP_SYSTEM,
            )
            if verdict.duplicate:
                canonical[other.id] = root.id
                logger.info("duplicate %s -> %s", other.id, root.id)
    return canonical


def embed_preview(chunk: ChunkRecord) -> str:
    payload = chunk.payload() or {}
    return str(payload.get("hadith") or chunk.text)[:4000]


def classify_tag_groups(
    agent: GeminiAgent,
    state: StateManager,
    chunks: list[ChunkRecord],
    vectors: dict[str, list[float]],
    *,
    threshold: float,
    group_cap: int,
) -> int:
    by_tag: dict[str, list[str]] = defaultdict(list)
    lookup = {c.id: c for c in chunks}
    for chunk in chunks:
        for tag in concepts_for_chunk(chunk):
            by_tag[tag].append(chunk.id)

    created = 0
    for tag, ids in by_tag.items():
        unique = list(dict.fromkeys(ids))
        if len(unique) > group_cap:
            logger.warning("tag %s has %d items; truncating to %d", tag, len(unique), group_cap)
            unique = unique[:group_cap]
        present = [cid for cid in unique if cid in vectors]
        if len(present) < 2:
            continue
        matrix = np.array([vectors[cid] for cid in present], dtype=float)
        for left, right, score in pairs_above_threshold(present, matrix, threshold):
            pid = pair_id(left, right)
            if state.has_edge(pid):
                continue
            relation = agent.complete_structured(
                f"Text A:\n{embed_preview(lookup[left])}\n\nText B:\n{embed_preview(lookup[right])}",
                EdgeRelation,
                system=EDGE_SYSTEM,
            )
            if relation.relation == "UNRELATED":
                state.save_edge(pid, left, right, "UNRELATED", score)
                continue
            state.save_edge(pid, left, right, relation.relation, score)
            created += 1
    return created


def apply_canonical_ids(
    state: StateManager,
    mapping: dict[str, str],
    chunks: list[ChunkRecord],
) -> None:
    mapped = set(mapping)
    for chunk in chunks:
        parent = mapping.get(chunk.id, chunk.canonical_id or chunk.id)
        state.mark(
            chunk.id,
            ChunkStatus.PROCESSED_PHASE2,
            payload=chunk.payload(),
            canonical_id=parent,
        )
        mapped.discard(chunk.id)
    for child, parent in mapping.items():
        if child in mapped:
            continue
        chunk = state.get_chunk(child)
        if chunk:
            state.mark(
                child,
                ChunkStatus.PROCESSED_PHASE2,
                payload=chunk.payload(),
                canonical_id=parent,
            )
