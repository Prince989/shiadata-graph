"""Orchestrate Phase 1 ingest + Gemini extraction for one book."""

from __future__ import annotations

import logging
from pathlib import Path

from config.paths import OUTPUT_DIR, RAW_EPUBS_DIR
from config.settings import Settings, get_settings
from src.agents.errors import AllKeysExhausted
from src.agents.gemini import GeminiAgent
from src.extractors.epub_parser import parse_epub
from src.extractors.txt_parser import parse_txt
from src.pipelines import hadith as hadith_pipe
from src.pipelines import history as history_pipe
from src.pipelines import tafsir as tafsir_pipe
from src.pipelines.catalog import resolve_book
from src.pipelines.llm_processor import chunk_id, process_unit
from src.state_manager import ChunkStatus, StateManager

logger = logging.getLogger(__name__)


def parse_file(path: Path):
    if path.suffix.lower() == ".epub":
        return parse_epub(path)
    return parse_txt(path)


def prepare_units(pipeline: str, units, settings: Settings):
    if pipeline == "hadith":
        return hadith_pipe.prepare(units)
    if pipeline == "tafsir":
        return tafsir_pipe.prepare(units)
    if pipeline == "history":
        return history_pipe.prepare(
            units,
            settings.history_pages_per_call,
            settings.history_max_chars,
        )
    raise ValueError(pipeline)


def ingest_book(
    book_id: str,
    state: StateManager,
    settings: Settings | None = None,
    raw_dir: Path | None = None,
    limit: int | None = None,
) -> int:
    settings = settings or get_settings()
    spec = resolve_book(book_id, raw_dir or settings.raw_data_dir)
    rows = []
    remaining = limit
    for path in spec.files:
        units = prepare_units(spec.pipeline, parse_file(path), settings)
        for unit in units:
            if remaining is not None and remaining <= 0:
                break
            rows.append(
                {
                    "id": chunk_id(book_id, unit.locator, unit.text),
                    "book_id": book_id,
                    "pipeline": spec.pipeline,
                    "locator": unit.locator,
                    "source_path": unit.source_path,
                    "text": unit.text,
                }
            )
            if remaining is not None:
                remaining -= 1
        if remaining is not None and remaining <= 0:
            break
    return state.upsert_chunks(rows)


def run_phase1(
    book_id: str,
    state: StateManager,
    agent: GeminiAgent,
    settings: Settings | None = None,
    raw_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    settings = settings or get_settings()
    ingest_book(book_id, state, settings, raw_dir=raw_dir, limit=limit)
    pending = state.list_chunks(
        book_id=book_id,
        statuses=[ChunkStatus.PENDING, ChunkStatus.ERROR],
        limit=limit,
    )
    job_id = state.record_job("phase1", book_id)
    processed = 0
    skipped = 0
    try:
        for chunk in pending:
            unit_text = chunk.text
            from src.extractors.epub_parser import ParsedUnit

            status = process_unit(
                agent,
                state,
                book_id=book_id,
                pipeline=chunk.pipeline,
                unit=ParsedUnit(
                    locator=chunk.locator,
                    text=unit_text,
                    source_path=chunk.source_path,
                ),
                output_dir=OUTPUT_DIR / "phase1",
                min_chars=settings.skip_min_chars,
            )
            if status == ChunkStatus.SKIPPED:
                skipped += 1
            else:
                processed += 1
    except AllKeysExhausted:
        state.finish_job(job_id, pause_reason="all_keys_exhausted")
        raise
    state.finish_job(job_id)
    return {"processed": processed, "skipped": skipped, "pending_seen": len(pending)}
