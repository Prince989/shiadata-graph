"""Gemini classification of SUPPORTS / CONTRADICTS / EXCEPTS between texts."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

import numpy as np

from src.agents.gemini import GeminiAgent
from src.core.vector_engine import bucket_keys_for_chunk, pairs_above_threshold
from src.models import DuplicateVerdict, EdgeRelation
from src.state_manager import ChunkRecord, ChunkStatus, StateManager

logger = logging.getLogger(__name__)

EXTREME_DF_RATIO = 0.15
EXTREME_DF_MIN_CHUNKS = 20

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
    from src.core.vector_engine import embed_text_for_chunk

    return embed_text_for_chunk(chunk)[:4000]


def build_concept_buckets(
    chunks: list[ChunkRecord],
    *,
    max_df_ratio: float = EXTREME_DF_RATIO,
    min_chunks_for_df: int = EXTREME_DF_MIN_CHUNKS,
) -> dict[str, list[str]]:
    """Chunk ids per edge-eligible node. Singletons are kept. Extreme-df skipped only on large sets."""
    by_node: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        for node in bucket_keys_for_chunk(chunk):
            by_node[node].append(chunk.id)
    n = len(chunks)
    apply_df = n >= min_chunks_for_df
    buckets: dict[str, list[str]] = {}
    for node, ids in by_node.items():
        unique = list(dict.fromkeys(ids))
        if apply_df and unique and (len(unique) / n) > max_df_ratio:
            logger.info("skip extreme-df node %s df=%s/%s", node, len(unique), n)
            continue
        buckets[node] = unique
    return buckets


def documents_for(chunks: list[ChunkRecord]) -> list:
    """Project chunks into the linker's Document shape."""
    from src.core.candidates import Document
    from src.core.vector_engine import (
        hadith_items,
        resolved_nodes_for_chunk,
        section_nodes_for_chunk,
    )

    docs = []
    for chunk in chunks:
        payload = chunk.payload() or {}
        sections = {n["type"]: n["node"] for n in section_nodes_for_chunk(chunk)}
        ayahs: list[str] = []
        ravis: list[str] = []
        for item in hadith_items(payload) or [payload]:
            ayahs.extend(item.get("quran_refs") or [])
            ravis.extend(item.get("ravis") or [])
        nodes = [n["key"] for n in resolved_nodes_for_chunk(chunk)]
        if not nodes:
            from src.core.vector_engine import bucket_keys_for_chunk

            nodes = bucket_keys_for_chunk(chunk)
        docs.append(
            Document(
                doc_id=chunk.id,
                nodes=list(dict.fromkeys(nodes)),
                ayahs=list(dict.fromkeys(ayahs)),
                ravis=list(dict.fromkeys(ravis)),
                bab=sections.get("bab", ""),
                kitab=sections.get("kitab", ""),
            )
        )
    return docs


def classify_candidate_pairs(
    agent: GeminiAgent,
    state: StateManager,
    chunks: list[ChunkRecord],
    vectors: dict[str, list[float]],
    *,
    max_pairs: int | None = None,
    min_score: float = 0.0,
) -> int:
    """Rank every candidate pair, then spend the model from the top down.

    Replaces bucket-and-compare-everything-inside. Two differences that matter:
    a pair raised by several signals at once outranks one raised by a single
    common node, and `max_pairs` is a real budget dial -- the ranking is
    complete before any call is made, so stopping early stops at the least
    promising pairs rather than at an arbitrary bucket boundary.
    """
    from src.core.candidates import generate

    lookup = {c.id: c for c in chunks}
    docs = documents_for(chunks)
    candidates = generate(
        docs, vectors=vectors, min_score=min_score, limit=max_pairs
    )
    logger.info("phase2 considering %d ranked candidate pairs", len(candidates))

    created = 0
    for candidate in candidates:
        pid = pair_id(candidate.left, candidate.right)
        if state.has_edge(pid):
            continue
        left, right = lookup.get(candidate.left), lookup.get(candidate.right)
        if left is None or right is None:
            continue
        relation = agent.complete_structured(
            f"Text A:\n{embed_preview(left)}\n\nText B:\n{embed_preview(right)}",
            EdgeRelation,
            system=EDGE_SYSTEM,
        )
        state.save_edge(
            pid, candidate.left, candidate.right, relation.relation, candidate.score
        )
        if relation.relation != "UNRELATED":
            created += 1
    return created


def classify_concept_groups(
    agent: GeminiAgent,
    state: StateManager,
    chunks: list[ChunkRecord],
    vectors: dict[str, list[float]],
    *,
    threshold: float,
    group_cap: int,
) -> int:
    """Legacy bucket-and-compare path. Superseded by classify_candidate_pairs."""
    lookup = {c.id: c for c in chunks}
    created = 0
    for node, ids in build_concept_buckets(chunks).items():
        unique = ids
        if len(unique) > group_cap:
            logger.warning(
                "node %s has %d items; truncating to %d", node, len(unique), group_cap
            )
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
