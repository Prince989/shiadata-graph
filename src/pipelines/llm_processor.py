"""Phase 1: send one text unit through Gemini structured output via GeminiAgent."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from pydantic import BaseModel

from src.agents.gemini import GeminiAgent
from src.extractors.epub_parser import ParsedUnit
from src.models import HadithExtraction, HistoryExtraction, TafsirExtraction
from src.pipelines.catalog import load_ontology
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


def schema_for(pipeline: str) -> type[BaseModel]:
    if pipeline == "hadith":
        return HadithExtraction
    if pipeline == "tafsir":
        return TafsirExtraction
    if pipeline == "history":
        return HistoryExtraction
    raise ValueError(f"Unknown pipeline {pipeline}")


def system_prompt(pipeline: str) -> str:
    if pipeline == "hadith":
        ontology = "، ".join(load_ontology())
        return (
            "You extract one hadith unit from classical Shia Arabic. "
            "Return the original Arabic, fluent Persian, precise English, "
            "narrators in chain order, and tags. "
            "Prefer tags from this Base Ontology and do not invent a parallel vocabulary: "
            f"{ontology}. "
            "If none apply, emit at most one extra tag. Never fabricate a hadith."
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

    result = agent.complete_structured(
        f"Locator: {unit.locator}\n\nText:\n{unit.text}",
        schema_for(pipeline),
        system=system_prompt(pipeline),
    )
    payload = result.model_dump()
    dest = output_dir / book_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{cid}.json").write_text(
        result.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    state.mark(cid, ChunkStatus.PROCESSED_PHASE1, payload=payload)
    logger.info("phase1 %s %s %s", book_id, pipeline, unit.locator)
    return ChunkStatus.PROCESSED_PHASE1
