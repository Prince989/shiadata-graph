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


def is_embeddable_hadith_payload(payload: dict) -> bool:
    """Drop legacy page-array extracts; keep flushed complete narrations."""
    if payload.get("page") and isinstance(payload.get("hadiths"), list) and not payload.get("page_start"):
        return False
    return True


def hadith_items(payload: dict) -> list[dict]:
    items = payload.get("hadiths")
    if isinstance(items, list) and items:
        return [item for item in items if isinstance(item, dict)]
    if payload.get("hadith"):
        return [payload]
    return []


def embed_text_for_chunk(chunk: ChunkRecord) -> str:
    payload = chunk.payload() or {}
    if chunk.pipeline == "hadith":
        parts = [str(item.get("hadith") or "") for item in hadith_items(payload)]
        joined = "\n\n".join(part for part in parts if part)
        return joined or str(payload.get("hadith") or chunk.text)
    if chunk.pipeline == "tafsir":
        return str(payload.get("tafsir_chunk") or payload.get("summary_fa") or chunk.text)
    if chunk.pipeline == "history":
        events = payload.get("events") or []
        titles = [e.get("event_title", "") for e in events if isinstance(e, dict)]
        return "\n".join(titles) or chunk.text
    return chunk.text


def resolved_nodes_for_chunk(chunk: ChunkRecord) -> list[dict]:
    """Nodes written by `main.py resolve-nodes`, or [] if it has not run.

    This is the identity system. Anything else in this module is the pre-resolver
    fallback, kept only so a corpus part-extracted under the old contract still
    exports; it must not be consulted when resolved nodes exist, or the graph
    carries two incompatible sets of identities at once.
    """
    payload = chunk.payload() or {}
    found: list[dict] = []
    seen: set[str] = set()
    for item in payload.get("nodes") or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        found.append(
            {
                "node": str(item.get("label") or key),
                "key": key,
                "type": str(item.get("type") or "concept"),
                "weight": float(item.get("weight") or 0.0),
                # Role is derived, not stored: IDF and weight drive ranking now,
                # but the export edge shape still wants a role.
                "role": "primary" if float(item.get("weight") or 0.0) >= 0.6 else "secondary",
                "parent": item.get("parent"),
                "broader": list(item.get("broader") or []),
            }
        )
    return found


def graph_nodes_for_chunk(chunk: ChunkRecord) -> list[dict]:
    """Every semantic node with type and role. Feeds export and search."""
    from src.pipelines.ontology import enforce_node_policy, semantic_nodes_of

    resolved = resolved_nodes_for_chunk(chunk)
    if resolved:
        return resolved

    payload = chunk.payload() or {}
    if chunk.pipeline == "hadith":
        nodes: list[dict] = []
        ravis: list[str] = []
        for item in hadith_items(payload):
            nodes.extend(semantic_nodes_of(item))
            ravis.extend(item.get("ravis") or [])
        if not nodes:
            nodes = semantic_nodes_of(payload)
        if not ravis:
            ravis = list(payload.get("ravis") or [])
        return enforce_node_policy(nodes, ravis)
    if chunk.pipeline == "tafsir":
        # strict=False: tafsir has no proposals channel yet, and closing its
        # vocabulary would silently empty every tafsir chunk.
        return enforce_node_policy(
            [
                {"node": str(c), "type": "concept", "role": "primary"}
                for c in (payload.get("core_concepts") or [])
            ],
            strict=False,
        )
    if chunk.pipeline == "history":
        nodes = []
        for event in payload.get("events") or []:
            if isinstance(event, dict):
                for c in event.get("historical_concepts") or []:
                    nodes.append({"node": str(c), "type": "concept", "role": "primary"})
        return enforce_node_policy(nodes, strict=False)
    return []


def bucket_keys_for_chunk(chunk: ChunkRecord) -> list[str]:
    """Edge-eligible node strings, plus every ancestor of each.

    Ancestors are what let sibling concepts meet: hadith 7 (الحساب), 8 (الثواب)
    and 9 (الجزاء) share no node at all, but all three sit under
    الجزاء الأخروي, so that bucket is where they get compared. The nodes
    themselves stay distinct for search.
    """
    from src.pipelines.ontology import broader_chain, bucket_eligible

    resolved = resolved_nodes_for_chunk(chunk)
    if resolved:
        # Resolver keys are already canonical identities, so no role gate: a node
        # reached this chunk because a mention resolved onto it, which is the
        # whole membership test. Ancestors are still expanded -- a child like
        # خلق العقل must reach العقل, and الحساب / الثواب / الجزاء must meet
        # under الجزاء الأخروي, neither of which happens on the leaf key alone.
        keys: list[str] = []
        for node in resolved:
            keys.append(node["key"])
            if node.get("parent"):
                keys.append(node["parent"])
            keys.extend(node.get("broader") or [])
        keys.extend(
            n["node"] for n in section_nodes_for_chunk(chunk) if n["type"] == "bab"
        )
        return list(dict.fromkeys(keys))

    keys: list[str] = []
    for n in graph_nodes_for_chunk(chunk):
        if not bucket_eligible(n["type"], n["role"]):
            continue
        keys.append(n["node"])
        keys.extend(broader_chain(n["node"]))
    # Bab only. A kitab spans hundreds of narrations, so as a bucket key it is
    # both useless for comparison and liable to trip the extreme-df cutoff, which
    # would silently drop it anyway. The kitab still reaches the graph as an
    # IN_KITAB edge; it just is not a unit of pairwise comparison.
    keys.extend(
        n["node"] for n in section_nodes_for_chunk(chunk) if n["type"] == "bab"
    )
    return list(dict.fromkeys(keys))


def section_nodes_for_chunk(chunk: ChunkRecord) -> list[dict]:
    """The kitab/bab this chunk was printed under.

    Kept separate from semantic_nodes because it is not an inference: the
    heading is printed above the narration, so this is the one grouping that
    holds no matter how the extraction went.
    """
    from src.extractors.classification import section_nodes

    payload = chunk.payload() or {}
    if chunk.pipeline != "hadith":
        return []
    found: list[dict] = []
    for item in hadith_items(payload) or [payload]:
        found.extend(
            section_nodes(str(item.get("kitab") or ""), str(item.get("bab") or ""))
        )
    if not found:
        found = section_nodes(
            str(payload.get("kitab") or ""), str(payload.get("bab") or "")
        )
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for node in found:
        key = (node["type"], node["node"])
        if key not in seen:
            seen.add(key)
            unique.append(node)
    return unique


def embed_pending_chunks(
    state: StateManager,
    agent: EmbeddingAgent,
    book_id: str | None = None,
    statuses: list[ChunkStatus] | None = None,
) -> int:
    chunks = state.list_chunks(
        book_id=book_id,
        statuses=statuses or [ChunkStatus.PROCESSED_PHASE1, ChunkStatus.EMBEDDED],
    )
    count = 0
    for chunk in chunks:
        if chunk.status == ChunkStatus.SKIPPED:
            continue
        payload = chunk.payload() or {}
        if chunk.pipeline == "hadith" and not is_embeddable_hadith_payload(payload):
            continue
        agent.embed_and_store(chunk.id, embed_text_for_chunk(chunk))
        if chunk.status == ChunkStatus.PROCESSED_PHASE1:
            state.mark(chunk.id, ChunkStatus.EMBEDDED, payload=chunk.payload())
        count += 1
    return count
