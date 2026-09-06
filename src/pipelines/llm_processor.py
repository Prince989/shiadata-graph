"""Phase 1: send one text unit through Gemini structured output via GeminiAgent."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel

from src.agents.gemini import GeminiAgent
from src.extractors.chunkers import split_hadith_page, strip_folklib_footnotes
from src.extractors.epub_parser import ParsedUnit
from src.models import (
    HadithPageExtraction,
    HadithUnify,
    HistoryExtraction,
    MentionsFill,
    TafsirExtraction,
)
from src.pipelines.ontology import (
    enforce_node_policy,
    remap_hadith_payload,
    semantic_nodes_of,
)
from src.pipelines.prompts import (
    HISTORY_PROMPT,
    TAFSIR_PROMPT,
    mentions_fill_prompt,
    hadith_prompt,
    unify_prompt,
)
from src.state_manager import ChunkStatus, StateManager

logger = logging.getLogger(__name__)

SKIP_MARKERS = ("رقم الصفحة", "عناوين الأبواب", "عدد الأحاديث")


def chunk_id(book_id: str, locator: str, text: str) -> str:
    payload = f"{book_id}\n{locator}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def should_skip(text: str, min_chars: int) -> bool:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    return any(marker in stripped for marker in SKIP_MARKERS)


def phase1_filename(source_path: str, locator: str, cid: str, marker: str = "") -> str:
    stem = Path(source_path).stem or "unknown"
    loc = re.sub(r"[^\w\u0600-\u06FF]+", "_", locator, flags=re.UNICODE)
    loc = re.sub(r"_+", "_", loc).strip("_")[:120]
    mark = re.sub(r"[^\w\u0600-\u06FF]+", "_", marker, flags=re.UNICODE)
    mark = re.sub(r"_+", "_", mark).strip("_")[:32]
    if mark and loc:
        return f"{stem}__{mark}__{loc}.json"
    if loc:
        return f"{stem}__{loc}.json"
    return f"{stem}__{cid[:16]}.json"
    stem = Path(source_path).stem or "unknown"
    loc = re.sub(r"[^\w\u0600-\u06FF]+", "_", locator, flags=re.UNICODE)
    loc = re.sub(r"_+", "_", loc).strip("_")[:120]
    if loc:
        return f"{stem}__{loc}.json"
    return f"{stem}__{cid[:16]}.json"


def hadith_system_extra(text: str) -> str:
    tokens = [token for token, _ in split_hadith_page(text)]
    if not tokens:
        return (
            "No numbered start was detected; this page is likely a continuation. "
            "Return hadiths with one continuation item (or more if the page still "
            "contains distinct narrations)."
        )
    listed = ", ".join(tokens)
    return (
        f"Detected numbered starts on this page: {listed}. "
        f"The hadiths array MUST include each of these {len(tokens)} numbered "
        "narrations (plus a leading continuation item if the page starts mid-hadith). "
        "Each numbered item is a SEPARATE claim. semantic_nodes for item N may use "
        "only the matn of item N. Do not copy nodes from a neighbour or from a "
        "kitab/bab heading."
    )


def schema_for(pipeline: str) -> type[BaseModel]:
    if pipeline == "hadith":
        return HadithPageExtraction
    if pipeline == "tafsir":
        return TafsirExtraction
    if pipeline == "history":
        return HistoryExtraction
    raise ValueError(f"Unknown pipeline {pipeline}")


def system_prompt(pipeline: str, extra: str = "") -> str:
    if pipeline == "hadith":
        return hadith_prompt(extra)
    if pipeline == "tafsir":
        return TAFSIR_PROMPT
    return HISTORY_PROMPT


UNIFY_AR_HEAD = 4000
UNIFY_BODY_BUDGET = 20_000



def extract_hadith_page(agent: GeminiAgent, unit: ParsedUnit) -> dict:
    text = strip_folklib_footnotes(unit.text)
    extra = hadith_system_extra(text)
    result = agent.complete_structured(
        f"Locator: {unit.locator}\n\nText:\n{text}",
        HadithPageExtraction,
        system=system_prompt("hadith", extra=extra),
    )
    return remap_hadith_payload(result.model_dump())


def needs_enrichment(payload: dict) -> bool:
    if payload.get("page_start") != payload.get("page_end"):
        return True
    if not str(payload.get("hadith_fa") or "").strip():
        return True
    if not str(payload.get("hadith_en") or "").strip():
        return True
    # Either channel counts. Keying on semantic_nodes alone made every payload
    # extracted under the mention contract look hollow, so every one of them
    # went to unify -- a second model call per hadith, for nothing.
    if len(payload.get("mentions") or []) + len(semantic_nodes_of(payload)) < 2:
        return True
    if not payload.get("ravis"):
        return True
    return False


def unify_assembled_hadith(agent: GeminiAgent, payload: dict) -> dict:
    """Fill missing FA/EN/mentions/ravis; always refresh ravis on multi-page.

    When topics are missing, a dedicated MentionsFill call runs first so the
    model cannot return FA/ravis with mentions:[]. Soft unify then fills
    translations. When topics are already present, a failed unify keeps the
    assembled payload. When topics stay missing after MentionsFill, failure is
    raised so the runner marks ERROR instead of writing hollow PROCESSED files.
    """
    from src.agents.errors import ProviderServerError, StructuredOutputError

    payload = dict(payload)
    payload["hadith"] = strip_folklib_footnotes(str(payload.get("hadith") or ""))
    if not needs_enrichment(payload):
        return remap_hadith_payload(payload)
    arabic = str(payload.get("hadith") or "")
    head = arabic[:UNIFY_AR_HEAD]
    if len(arabic) <= UNIFY_BODY_BUDGET:
        body = arabic
    else:
        body = arabic[:UNIFY_BODY_BUDGET]
    topic_count = len(payload.get("mentions") or []) + len(semantic_nodes_of(payload))
    need_topics = topic_count < 1
    label = payload.get("marker") or payload.get("locator")

    if need_topics:
        try:
            filled = agent.complete_structured(
                f"Arabic hadith:\n{body}",
                MentionsFill,
                system=mentions_fill_prompt(),
            )
        except (StructuredOutputError, ProviderServerError) as exc:
            raise StructuredOutputError(
                f"mentions fill failed for {label}: {exc}"
            ) from exc
        payload["mentions"] = _merge_mentions(
            payload.get("mentions"),
            [m.model_dump() for m in filled.mentions],
        )
        payload = remap_hadith_payload(payload)
        if not (payload.get("mentions") or []):
            raise StructuredOutputError(
                f"mentions fill returned topics that grounding dropped for {label}"
            )
        if not needs_enrichment(payload):
            return payload

    # Topics present (or MentionsFill already ran): soft unify for FA/EN/ravis.
    schema = HadithUnify
    system = unify_prompt(require_topics=False)
    try:
        result = agent.complete_structured(
            f"Isnad / opening:\n{head}\n\nMatn:\n{body}",
            schema,
            system=system,
        )
    except (StructuredOutputError, ProviderServerError) as exc:
        if need_topics and not (payload.get("mentions") or []):
            raise StructuredOutputError(
                f"unify could not extract mentions for {label}: {exc}"
            ) from exc
        logger.warning(
            "unify failed for %s; persisting assembled payload: %s",
            label,
            exc,
        )
        return remap_hadith_payload(payload)
    out = dict(payload)
    if result.ravis:
        out["ravis"] = list(result.ravis)
    if result.mentions:
        out["mentions"] = _merge_mentions(
            payload.get("mentions"), [m.model_dump() for m in result.mentions]
        )
    if result.quotes:
        out["quotes"] = _merge_quotes(
            payload.get("quotes"), [q.model_dump() for q in result.quotes]
        )
    existing = semantic_nodes_of(payload)
    if result.semantic_nodes and len(existing) < 2:
        out["semantic_nodes"] = enforce_node_policy(
            [n.model_dump() for n in result.semantic_nodes],
            out.get("ravis"),
        )
    if result.hadith_fa and not str(payload.get("hadith_fa") or "").strip():
        out["hadith_fa"] = result.hadith_fa
    if result.hadith_en and not str(payload.get("hadith_en") or "").strip():
        out["hadith_en"] = result.hadith_en
    out = remap_hadith_payload(out)
    if need_topics and not (out.get("mentions") or []):
        raise StructuredOutputError(
            f"unify returned mentions that grounding dropped for {label}"
        )
    return out


def _merge_mentions(existing, extra) -> list[dict]:
    """Union by (text, type), keeping the better salience and any real evidence."""
    merged: dict[tuple[str, str], dict] = {}
    for item in list(existing or []) + list(extra or []):
        if not isinstance(item, dict):
            continue
        key = (str(item.get("text") or "").strip(), str(item.get("type") or "concept"))
        if not key[0]:
            continue
        current = merged.get(key)
        if current is None:
            merged[key] = dict(item)
            continue
        if float(item.get("salience") or 0) > float(current.get("salience") or 0):
            current["salience"] = item.get("salience")
        if not str(current.get("evidence") or "").strip():
            current["evidence"] = item.get("evidence") or ""
    return list(merged.values())


def _merge_quotes(existing, extra) -> list[dict]:
    merged: dict[str, dict] = {}
    for item in list(existing or []) + list(extra or []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text and text not in merged:
            merged[text] = dict(item)
    return list(merged.values())


def persist_complete_hadith(
    state: StateManager,
    *,
    book_id: str,
    source_path: str,
    payload: dict,
    output_dir: Path,
    status: ChunkStatus = ChunkStatus.PROCESSED_PHASE1,
    error: str | None = None,
) -> str:
    """Write one complete narration and upsert it as an embeddable Phase 1 chunk."""
    payload["hadith"] = strip_folklib_footnotes(str(payload.get("hadith") or ""))
    locator = str(payload.get("locator") or "")
    marker = str(payload.get("marker") or "")
    arabic = str(payload.get("hadith") or "")
    cid = chunk_id(book_id, f"{marker}|{locator}", arabic)
    existing = state.get_chunk(cid)
    if existing and existing.status not in {ChunkStatus.PENDING, ChunkStatus.ERROR}:
        if status == ChunkStatus.PROCESSED_PHASE1:
            return cid
    dest = output_dir / book_id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / phase1_filename(source_path, locator, cid, marker=marker)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if not existing:
        state.upsert_chunks(
            [
                {
                    "id": cid,
                    "book_id": book_id,
                    "pipeline": "hadith",
                    "locator": locator,
                    "source_path": source_path,
                    "text": arabic,
                }
            ]
        )
    state.mark(cid, status, payload=payload, error=error)
    logger.info(
        "phase1 hadith %s %s %s",
        status.value,
        locator,
        marker,
    )
    return cid


def process_unit(
    agent: GeminiAgent,
    state: StateManager,
    *,
    book_id: str,
    pipeline: str,
    unit: ParsedUnit,
    output_dir: Path,
    min_chars: int,
) -> ChunkStatus:
    cid = chunk_id(book_id, unit.locator, unit.text)
    existing = state.get_chunk(cid)
    if existing and existing.status not in {ChunkStatus.PENDING, ChunkStatus.ERROR}:
        return existing.status

    if should_skip(unit.text, min_chars):
        state.mark(cid, ChunkStatus.SKIPPED, error="too short or table of contents")
        return ChunkStatus.SKIPPED

    extra = hadith_system_extra(unit.text) if pipeline == "hadith" else ""
    result = agent.complete_structured(
        f"Locator: {unit.locator}\n\nText:\n{unit.text}",
        schema_for(pipeline),
        system=system_prompt(pipeline, extra=extra),
    )
    payload = result.model_dump()
    if pipeline == "hadith":
        payload = remap_hadith_payload(payload)
    dest = output_dir / book_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / phase1_filename(unit.source_path, unit.locator, cid)).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    state.mark(cid, ChunkStatus.PROCESSED_PHASE1, payload=payload)
    logger.info("phase1 %s %s %s", book_id, pipeline, unit.locator)
    return ChunkStatus.PROCESSED_PHASE1
