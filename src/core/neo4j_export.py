"""Write Neo4j-ready JSONL plus a Cypher import script. No live DB required."""

from __future__ import annotations

import json
from pathlib import Path

from config.paths import OUTPUT_DIR
from src.core.vector_engine import concepts_for_chunk
from src.state_manager import ChunkStatus, StateManager


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
        concepts: set[str] = set()
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
            for tag in concepts_for_chunk(chunk):
                concepts.add(tag)
                edges.write(
                    json.dumps(
                        {
                            "type": "TAGGED",
                            "start": chunk.id,
                            "end": f"concept:{tag}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            for ravi in payload.get("ravis") or []:
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
        for tag in sorted(concepts):
            nodes.write(
                json.dumps({"id": f"concept:{tag}", "labels": ["Concept"], "name": tag})
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
