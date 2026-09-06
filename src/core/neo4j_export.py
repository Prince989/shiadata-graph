"""Write Neo4j-ready JSONL plus a Cypher import script. No live DB required."""

from __future__ import annotations

import json
from pathlib import Path

from config.paths import OUTPUT_DIR
from src.core.vector_engine import (
    graph_nodes_for_chunk,
    hadith_items,
    section_nodes_for_chunk,
)
from src.state_manager import ChunkStatus, StateManager

_EDGE_BY_TYPE = {
    "concept": "HAS_CONCEPT",
    "person": "MENTIONS",
    "place": "MENTIONS",
    "group": "MENTIONS",
    "event": "MENTIONS",
    "work": "MENTIONS",
    "ayah": "CITES",
}
_LABEL_BY_TYPE = {
    "concept": "Concept",
    "person": "Person",
    "place": "Place",
    "group": "Group",
    "event": "Event",
    "work": "Work",
    "ayah": "Ayah",
    "kitab": "Kitab",
    "bab": "Bab",
}


def export_neo4j(state: StateManager, dest: Path | None = None) -> Path:
    dest = dest or (OUTPUT_DIR / "neo4j")
    dest.mkdir(parents=True, exist_ok=True)
    nodes_path = dest / "nodes.jsonl"
    edges_path = dest / "edges.jsonl"

    chunks = state.list_chunks(
        statuses=[
            ChunkStatus.PROCESSED_PHASE1,
            ChunkStatus.EMBEDDED,
            ChunkStatus.PROCESSED_PHASE2,
        ]
    )
    with nodes_path.open("w", encoding="utf-8") as nodes, edges_path.open(
        "w", encoding="utf-8"
    ) as edges:
        books: set[str] = set()
        nodes_seen: set[tuple[str, str]] = set()
        narrators: set[str] = set()
        for chunk in chunks:
            books.add(chunk.book_id)
            payload = chunk.payload() or {}
            canonical = chunk.canonical_id or chunk.id
            label = {
                "hadith": "CanonicalHadith",
                "tafsir": "TafsirChunk",
                "history": "HistoricalEvent",
            }.get(chunk.pipeline, "Chunk")
            nodes.write(
                json.dumps(
                    {
                        "id": canonical if chunk.pipeline == "hadith" else chunk.id,
                        "labels": [label],
                        "book_id": chunk.book_id,
                        "locator": chunk.locator,
                        "pipeline": chunk.pipeline,
                        "payload": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if chunk.pipeline == "hadith" and canonical != chunk.id:
                edges.write(
                    json.dumps(
                        {
                            "type": "APPEARS_IN",
                            "start": canonical,
                            "end": chunk.id,
                            "locator": chunk.locator,
                            "book_id": chunk.book_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            for n in graph_nodes_for_chunk(chunk):
                ntype = n["type"]
                name = n["node"]
                nodes_seen.add((ntype, name))
                edges.write(
                    json.dumps(
                        {
                            "type": _EDGE_BY_TYPE.get(ntype, "MENTIONS"),
                            "start": chunk.id,
                            "end": f"{ntype}:{name}",
                            "role": n["role"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            for node in section_nodes_for_chunk(chunk):
                nodes_seen.add((node["type"], node["node"]))
                edges.write(
                    json.dumps(
                        {
                            "type": "IN_KITAB" if node["type"] == "kitab" else "IN_BAB",
                            "start": chunk.id,
                            "end": f"{node['type']}:{node['node']}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            # quran_refs is injected by the extractor, never by the model, and
            # uses the same ayah: prefix the tafsir branch writes below, so a
            # hadith and the tafsir of the verse it cites meet on one node.
            cited: list[str] = []
            for item in hadith_items(payload):
                cited.extend(item.get("quran_refs") or [])
            cited.extend(payload.get("quran_refs") or [])
            for ref in dict.fromkeys(cited):
                nodes_seen.add(("ayah", ref))
                edges.write(
                    json.dumps(
                        {
                            "type": "CITES",
                            "start": chunk.id,
                            "end": f"ayah:{ref}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            ravis: list[str] = []
            for item in hadith_items(payload):
                ravis.extend(item.get("ravis") or [])
            if not ravis:
                ravis = list(payload.get("ravis") or [])
            for ravi in ravis:
                narrators.add(ravi)
                edges.write(
                    json.dumps(
                        {
                            "type": "NARRATED_BY",
                            "start": chunk.id,
                            "end": f"narrator:{ravi}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if chunk.pipeline == "tafsir" and payload.get("ayah_anchor"):
                edges.write(
                    json.dumps(
                        {
                            "type": "COMMENTS_ON",
                            "start": chunk.id,
                            "end": f"ayah:{payload['ayah_anchor']}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        for book_id in sorted(books):
            nodes.write(
                json.dumps({"id": f"book:{book_id}", "labels": ["Book"], "book_id": book_id})
                + "\n"
            )
        for ntype, name in sorted(nodes_seen):
            nodes.write(
                json.dumps(
                    {
                        "id": f"{ntype}:{name}",
                        "labels": [_LABEL_BY_TYPE.get(ntype, "Concept")],
                        "name": name,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        for name in sorted(narrators):
            nodes.write(
                json.dumps({"id": f"narrator:{name}", "labels": ["Narrator"], "name": name})
                + "\n"
            )
        for row in state.list_edges():
            if row["relation"] == "UNRELATED":
                continue
            edges.write(
                json.dumps(
                    {
                        "type": row["relation"],
                        "start": row["left_id"],
                        "end": row["right_id"],
                        "cosine": row["cosine"],
                    }
                )
                + "\n"
            )

    (dest / "import.cypher").write_text(
        """
// Load JSONL produced by `python main.py export-neo4j`
// CALL apoc.load.json('file:///nodes.jsonl') YIELD value
// MERGE (n {id: value.id}) SET n += value;
// CALL apoc.load.json('file:///edges.jsonl') YIELD value
// MATCH (a {id: value.start}), (b {id: value.end})
// CALL apoc.merge.relationship(a, value.type, {}, {}, b) YIELD rel
// RETURN count(rel);
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return dest
