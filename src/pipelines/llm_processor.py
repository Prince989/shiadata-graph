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
from src.models import HadithPageExtraction, HadithUnify, HistoryExtraction, TafsirExtraction
from src.pipelines.ontology import grouping_prefs, remap_hadith_payload, remap_tag_list
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
        "narrations (plus a leading continuation item if the page starts mid-hadith)."
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
        examples = "، ".join(grouping_prefs())
        return (
            "You extract EVERY hadith on this printed page, not just the first. "
            "Return JSON with page (copy the locator) and hadiths: an array with one "
            "object per distinct narration. "
            "If the page starts mid-hadith (no new number), include that fragment first "
            "with marker 'continuation'. "
            "Then include each numbered hadith (e.g. '3 -', '8-', '[ ١٥٤٩٥ ] ١ ـ'). "
            "Ignore editor footnotes like [1] [2] at the bottom of the page. "
            "For each item: original Arabic, fluent Persian, precise English, "
            "narrators in chain order, and tags. "
            "Tags are ontological concept IDs (semantic bridges), not keywords and not "
            "kitāb/bāb titles. Assign 2–6 mid-grain fuṣḥā labels that name the "
            "theological, legal, or ethical CLAIM of the matn. "
            "Never tag instruments, props, proper names-as-topics, or a word merely "
            "because it occurs. Never use a root heading (الإيمان، المعاد، الحرام، …) "
            "unless the hadith is actually defining that heading. "
            "Example: afterlife punishment for a man who killed himself with hot steel "
            "→ [الانتحار, عذاب البرزخ, الجزاء الأخروي] — not حديد and not كتاب العقل. "
            "Known grouping concepts (examples, not an exclusive menu): "
            f"{examples}. "
            "JSON must be complete and compact: copy each Arabic matn once, "
            "do not repeat sentences, do not pad translations. "
            "Never fabricate a hadith. "
            f"{extra}"
        )
    if pipeline == "tafsir":
        return (
            "You extract one Al-Mizan tafsir unit anchored to a Qur'anic ayah range. "
            "Copy ayah_anchor from the locator. Extract any quoted hadith. "
            "Write a two-line Persian summary. Keep tafsir_chunk as the main Arabic/Persian text."
        )
    return (
        "You extract historical events from a long classical Arabic narrative. "
        "Split into distinct events with titles, characters, concepts, and the "
        "paragraphs covering each event. Do not invent events absent from the text."
    )


UNIFY_AR_HEAD = 4000
UNIFY_BODY_BUDGET = 20_000

UNIFY_SYSTEM = (
    "From the assembled hadith, return only ravis (isnad chain in order, from the opening) "
    "and mid-grain fuṣḥā concept tags for the CLAIM of the whole narration. "
    "Do not retell the matn. Do not use kitāb titles or instruments as tags."
)


def extract_hadith_page(agent: GeminiAgent, unit: ParsedUnit) -> dict:
    text = strip_folklib_footnotes(unit.text)
    extra = hadith_system_extra(text)
    result = agent.complete_structured(
        f"Locator: {unit.locator}\n\nText:\n{text}",
        HadithPageExtraction,
        system=system_prompt("hadith", extra=extra),
    )
    return remap_hadith_payload(result.model_dump())


def unify_assembled_hadith(agent: GeminiAgent, payload: dict) -> dict:
    """Fill tags/ravis for multi-page hadiths. Does not re-translate."""
    if payload.get("page_start") == payload.get("page_end"):
        return remap_hadith_payload(payload)
    arabic = str(payload.get("hadith") or "")
    head = arabic[:UNIFY_AR_HEAD]
    if len(arabic) <= UNIFY_BODY_BUDGET:
        body = arabic
    else:
        body = str(payload.get("hadith_en") or payload.get("hadith_fa") or arabic[:UNIFY_BODY_BUDGET])
    result = agent.complete_structured(
        f"Isnad / opening:\n{head}\n\nMatn (for tags):\n{body}",
        HadithUnify,
        system=UNIFY_SYSTEM,
    )
    out = dict(payload)
    out["ravis"] = list(result.ravis or payload.get("ravis") or [])
    out["tags"] = remap_tag_list(list(result.tags or []))
    return remap_hadith_payload(out)


def persist_complete_hadith(
    state: StateManager,
    *,
    book_id: str,
    source_path: str,
    payload: dict,
    output_dir: Path,
) -> str:
    """Write one complete narration and upsert it as an embeddable Phase 1 chunk."""
    locator = str(payload.get("locator") or "")
    marker = str(payload.get("marker") or "")
    arabic = str(payload.get("hadith") or "")
    cid = chunk_id(book_id, f"{marker}|{locator}", arabic)
    existing = state.get_chunk(cid)
    if existing and existing.status not in {ChunkStatus.PENDING, ChunkStatus.ERROR}:
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
    state.mark(cid, ChunkStatus.PROCESSED_PHASE1, payload=payload)
    logger.info("phase1 hadith flushed %s %s", locator, marker)
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
