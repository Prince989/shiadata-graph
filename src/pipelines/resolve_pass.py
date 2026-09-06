"""Corpus-wide pass: mentions in, resolved nodes out.

Runs after phase 1 and before phase 2. It has to be a separate pass because
identity is a property of the whole corpus -- deciding that عقل المرء is العقل
requires having seen العقل elsewhere, which per-page extraction cannot do.

Reads every phase-1 payload, resolves all mentions together, then writes the
resolved node keys back onto each payload and saves the node table alongside.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from config.paths import OUTPUT_DIR
from src.pipelines.resolver import Mention, Node, resolve_with_assignments

logger = logging.getLogger(__name__)

NODES_FILENAME = "nodes.json"


def _doc_id(payload: dict, path: Path) -> str:
    marker = str(payload.get("marker") or "").strip()
    locator = str(payload.get("locator") or "").strip()
    return f"{path.parent.name}|{locator}|{marker}" if (marker or locator) else str(path)


def _upcast_legacy(items, salience_for=None) -> list[dict]:
    """Turn a pre-resolver node list into mentions."""
    out: list[dict] = []
    for item in items or []:
        if isinstance(item, dict):
            text = item.get("node") or item.get("text")
            salience = 0.9 if item.get("role") == "primary" else 0.4
            node_type = item.get("type") or "concept"
        else:
            text = str(item or "")
            salience = salience_for if salience_for is not None else 0.6
            node_type = "concept"
        if not str(text or "").strip():
            continue
        out.append(
            {
                "text": text,
                "type": node_type,
                "salience": salience,
                "evidence": "",
            }
        )
    return out


def _mentions_of(payload: dict) -> list[dict]:
    """Every mention in this payload, whatever pipeline wrote it.

    The three pipelines nest differently, and reading only the top level meant
    history never resolved at all: `HistoryExtraction` puts its mentions inside
    `events[]`, so a walk that stops at the root finds nothing and the whole
    pipeline writes back empty `nodes`. That silently broke the one node space
    the design exists to provide.

    Legacy shapes are upcast too, so a corpus part-extracted under the old
    contract still resolves: hadith `semantic_nodes`, tafsir `core_concepts`,
    and history `historical_concepts` / `characters_involved`.
    """
    found: list[dict] = []

    raw = payload.get("mentions")
    if isinstance(raw, list):
        found.extend(m for m in raw if isinstance(m, dict))

    # History nests one level down, per event.
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        nested = event.get("mentions")
        if isinstance(nested, list):
            found.extend(m for m in nested if isinstance(m, dict))
        if not nested:
            found.extend(_upcast_legacy(event.get("historical_concepts")))
            found.extend(
                {**m, "type": "person"}
                for m in _upcast_legacy(event.get("characters_involved"))
            )

    if found:
        return found

    found.extend(_upcast_legacy(payload.get("semantic_nodes")))
    # Tafsir's pre-mention channel.
    found.extend(_upcast_legacy(payload.get("core_concepts")))
    return found


def collect_mentions(root: Path) -> tuple[list[Mention], dict[str, Path]]:
    mentions: list[Mention] = []
    sources: dict[str, Path] = {}
    for path in sorted(root.rglob("*.json")):
        if path.name == NODES_FILENAME:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("skip unreadable payload %s: %s", path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        doc_id = _doc_id(payload, path)
        sources[doc_id] = path
        for item in _mentions_of(payload):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            try:
                salience = float(item.get("salience", 0.5))
            except (TypeError, ValueError):
                salience = 0.5
            mentions.append(
                Mention(
                    text=text,
                    type=str(item.get("type") or "concept"),
                    doc_id=doc_id,
                    salience=max(0.0, min(1.0, salience)),
                    evidence=str(item.get("evidence") or ""),
                )
            )
    return mentions, sources


def _ancestors(node: Node) -> list[str]:
    """Curated broader terms above this node, as node keys.

    Only the concept catalog has a hierarchy worth walking; a morph parent is
    already carried separately as `parent`.
    """
    from src.pipelines.ontology import broader_chain, normalize_ar

    if not node.curated or node.type != "concept":
        return []
    return [f"concept:{normalize_ar(label)}" for label in broader_chain(node.label)]


def run(root: Path | None = None, write: bool = True) -> dict:
    """Resolve the whole corpus. Returns a summary dict."""
    root = root or (OUTPUT_DIR / "phase1")
    mentions, sources = collect_mentions(root)
    if not mentions:
        return {"documents": 0, "mentions": 0, "nodes": 0}

    # Assignments come back from resolution itself. Deriving them afterwards
    # from a (type, surface) index was wrong: resolution retypes mentions, so a
    # lookup keyed on the ORIGINAL type missed every corrected one and wrote the
    # narration back with no nodes at all -- invisible in the node table, which
    # showed the right label and the right df.
    nodes, assignments = resolve_with_assignments(mentions)

    per_doc: dict[str, dict[str, float]] = {}
    for mention, key in zip(mentions, assignments):
        if not key:
            continue
        bucket = per_doc.setdefault(mention.doc_id, {})
        bucket[key] = max(bucket.get(key, 0.0), mention.salience)

    if write:
        for doc_id, path in sources.items():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            assigned = per_doc.get(doc_id, {})
            payload["nodes"] = [
                {
                    "key": key,
                    "label": nodes[key].label,
                    "type": nodes[key].type,
                    "weight": round(weight, 3),
                    # Carried onto the document so the linker can reach a
                    # parent without loading the node table: خلق العقل has to
                    # meet plain العقل, and الحساب / الثواب / الجزاء have to
                    # meet under الجزاء الأخروي. Stored only on the node table,
                    # the hierarchy was documentation rather than a signal.
                    "parent": nodes[key].parent,
                    "broader": _ancestors(nodes[key]),
                }
                for key, weight in sorted(assigned.items(), key=lambda kv: -kv[1])
                if key in nodes
            ]
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        table = {
            key: {
                **{k: v for k, v in asdict(node).items() if k not in ("surfaces", "docs")},
                "surfaces": sorted(node.surfaces),
                "df": node.df,
            }
            for key, node in nodes.items()
        }
        (root / NODES_FILENAME).write_text(
            json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    singletons = sum(1 for n in nodes.values() if n.df == 1)
    return {
        "documents": len(sources),
        "mentions": len(mentions),
        "nodes": len(nodes),
        "singletons": singletons,
        "curated": sum(1 for n in nodes.values() if n.curated),
        "children": sum(1 for n in nodes.values() if n.parent),
    }


def report(nodes: dict[str, Node], limit: int = 30) -> str:
    ranked = sorted(nodes.values(), key=lambda n: (-n.df, n.label))
    lines = [f"{len(nodes)} nodes", ""]
    for node in ranked[:limit]:
        kind = "curated" if node.curated else ("child" if node.parent else "derived")
        lines.append(f"   df={node.df:<5} {kind:<8} {node.label}")
    return "\n".join(lines)
