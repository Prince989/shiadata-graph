"""Orchestrate Phase 1 ingest + Gemini extraction for one book."""

from __future__ import annotations

import logging
from pathlib import Path

from config.paths import OUTPUT_DIR, RAW_EPUBS_DIR
from config.settings import Settings, get_settings
from src.agents.errors import AllKeysExhausted, StructuredOutputError
from src.agents.gemini import GeminiAgent
from src.extractors.epub_parser import parse_epub
from src.extractors.txt_parser import parse_txt
from src.pipelines import hadith as hadith_pipe
from src.pipelines import history as history_pipe
from src.pipelines import tafsir as tafsir_pipe
from src.pipelines.catalog import resolve_book
from src.pipelines.hadith_accumulator import OpenHadith, consume_page
from src.pipelines.llm_processor import (
    chunk_id,
    extract_hadith_page,
    persist_complete_hadith,
    process_unit,
    should_skip,
    unify_assembled_hadith,
)
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


def run_hadith_phase1(
    book_id: str,
    state: StateManager,
    agent: GeminiAgent,
    settings: Settings,
    raw_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    spec = resolve_book(book_id, raw_dir or settings.raw_data_dir)
    output_dir = OUTPUT_DIR / "phase1"
    job_id = state.record_job("phase1", book_id)
    flushed = 0
    pages = 0
    skipped = 0
    errors = 0
    remaining = limit
    stop = False
    last_source: str | None = None
    last_buf: OpenHadith | None = None
    try:
        for path in spec.files:
            if stop or (remaining is not None and remaining <= 0):
                break
            source = str(path)
            last_source = source
            ordered = prepare_units("hadith", parse_file(path), settings)
            last = state.get_hadith_progress(book_id, source)
            start_i = 0
            if last:
                for i, unit in enumerate(ordered):
                    if unit.locator == last:
                        start_i = i + 1
                        break
            raw_buf = state.get_hadith_buffer(book_id, source)
            buf = OpenHadith.from_dict(raw_buf) if raw_buf else None
            last_buf = buf
            i = start_i
            while i < len(ordered):
                if remaining is not None and remaining <= 0:
                    break
                unit = ordered[i]
                next_text = ordered[i + 1].text if i + 1 < len(ordered) else None
                if should_skip(unit.text, settings.skip_min_chars):
                    skipped += 1
                    state.set_hadith_progress(book_id, source, unit.locator)
                    if remaining is not None:
                        remaining -= 1
                    i += 1
                    continue
                try:
                    page_payload = extract_hadith_page(agent, unit)
                except StructuredOutputError as exc:
                    errors += 1
                    logger.error("phase1 JSON failed %s: %s", unit.locator, exc)
                    state.set_hadith_buffer(
                        book_id, source, buf.to_dict() if buf else None
                    )
                    stop = True
                    break
                items = [it for it in (page_payload.get("hadiths") or []) if isinstance(it, dict)]
                complete, buf = consume_page(unit.locator, unit.text, items, buf, next_text)
                last_buf = buf
                for rec in complete:
                    rec = unify_assembled_hadith(agent, rec)
                    persist_complete_hadith(
                        state,
                        book_id=book_id,
                        source_path=source,
                        payload=rec,
                        output_dir=output_dir,
                    )
                    flushed += 1
                state.set_hadith_buffer(book_id, source, buf.to_dict() if buf else None)
                state.set_hadith_progress(book_id, source, unit.locator)
                pages += 1
                if remaining is not None:
                    remaining -= 1
                i += 1
            if stop:
                break
            if i >= len(ordered) and buf:
                rec = unify_assembled_hadith(agent, buf.assemble())
                persist_complete_hadith(
                    state,
                    book_id=book_id,
                    source_path=source,
                    payload=rec,
                    output_dir=output_dir,
                )
                flushed += 1
                buf = None
                last_buf = None
                state.set_hadith_buffer(book_id, source, None)
    except AllKeysExhausted:
        if last_source is not None:
            state.set_hadith_buffer(
                book_id, last_source, last_buf.to_dict() if last_buf else None
            )
        state.finish_job(job_id, pause_reason="all_keys_exhausted")
        raise
    state.finish_job(job_id)
    logger.info(
        "phase1 hadith book=%s pages=%s flushed=%s errors=%s",
        book_id,
        pages,
        flushed,
        errors,
    )
    return {
        "processed": flushed,
        "pages": pages,
        "skipped": skipped,
        "errors": errors,
        "pending_seen": pages,
    }


def run_phase1(
    book_id: str,
    state: StateManager,
    agent: GeminiAgent,
    settings: Settings | None = None,
    raw_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    settings = settings or get_settings()
    spec = resolve_book(book_id, raw_dir or settings.raw_data_dir)
    if spec.pipeline == "hadith":
        return run_hadith_phase1(book_id, state, agent, settings, raw_dir=raw_dir, limit=limit)

    ingest_book(book_id, state, settings, raw_dir=raw_dir, limit=limit)
    pending = state.list_chunks(
        book_id=book_id,
        statuses=[ChunkStatus.PENDING, ChunkStatus.ERROR],
        limit=limit,
    )
    job_id = state.record_job("phase1", book_id)
    processed = 0
    skipped = 0
    errors = 0
    try:
        for chunk in pending:
            from src.extractors.epub_parser import ParsedUnit

            try:
                status = process_unit(
                    agent,
                    state,
                    book_id=book_id,
                    pipeline=chunk.pipeline,
                    unit=ParsedUnit(
                        locator=chunk.locator,
                        text=chunk.text,
                        source_path=chunk.source_path,
                    ),
                    output_dir=OUTPUT_DIR / "phase1",
                    min_chars=settings.skip_min_chars,
                )
            except StructuredOutputError as exc:
                state.mark(
                    chunk.id,
                    ChunkStatus.ERROR,
                    error=str(exc)[:2000],
                    bump_attempts=True,
                )
                errors += 1
                logger.error("phase1 JSON failed %s: %s", chunk.locator, exc)
                continue
            if status == ChunkStatus.SKIPPED:
                skipped += 1
            else:
                processed += 1
    except AllKeysExhausted:
        state.finish_job(job_id, pause_reason="all_keys_exhausted")
        raise
    state.finish_job(job_id)
    return {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "pending_seen": len(pending),
    }
