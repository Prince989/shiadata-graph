"""Orchestrate Phase 1 ingest + Gemini extraction for one book."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from config.paths import OUTPUT_DIR, RAW_EPUBS_DIR
from config.settings import Settings, get_settings
from src.agents.errors import AllKeysExhausted, ProviderServerError, StructuredOutputError
from src.agents.gemini import GeminiAgent
from src.extractors.epub_parser import parse_epub
from src.extractors.txt_parser import parse_txt
from src.pipelines import hadith as hadith_pipe
from src.pipelines import history as history_pipe
from src.pipelines import tafsir as tafsir_pipe
from src.pipelines.catalog import resolve_book
from src.pipelines.hadith_accumulator import (
    OpenHadith,
    close_open_hadith_with_page,
    consume_page,
)
from src.pipelines.llm_processor import (
    chunk_id,
    extract_hadith_page,
    persist_complete_hadith,
    process_unit,
    should_skip,
    unify_assembled_hadith,
)
from src.pipelines.ontology import remap_hadith_payload
from src.state_manager import ChunkStatus, StateManager

logger = logging.getLogger(__name__)


def locator_matches_page(
    locator: str,
    page: str,
    volume: int | str | None = None,
) -> bool:
    """True when a unit locator is the requested print page (debug targeting).

    `page` may be a bare number (`30`) or a locator fragment
    (`جلد 1 - صفحه 30`). Optional `volume` further requires `جلد N`.
    """
    loc = str(locator or "").strip()
    want = str(page or "").strip()
    if not loc or not want:
        return False
    if "صفحه" in want or "جلد" in want:
        if want not in loc and loc != want:
            return False
    else:
        if not re.search(rf"صفحه\s*{re.escape(want)}(?!\d)", loc):
            return False
    if volume is None or str(volume).strip() == "":
        return True
    vol = str(volume).strip()
    return bool(re.search(rf"جلد\s*{re.escape(vol)}(?!\d)", loc))


def _flush_open_hadith(
    *,
    state: StateManager,
    agent: GeminiAgent,
    book_id: str,
    source: str,
    buf: OpenHadith | None,
    output_dir: Path,
    force: bool = False,
) -> tuple[int, int]:
    """Assemble and persist a trailing open buffer. Returns (flushed, errors)."""
    if buf is None:
        return 0, 0
    assembled = buf.assemble()
    try:
        rec = unify_assembled_hadith(agent, assembled)
    except StructuredOutputError as exc:
        logger.error(
            "phase1 unify mentions failed %s: %s",
            assembled.get("marker") or assembled.get("locator"),
            exc,
        )
        persist_complete_hadith(
            state,
            book_id=book_id,
            source_path=source,
            payload=remap_hadith_payload(dict(assembled)),
            output_dir=output_dir,
            status=ChunkStatus.ERROR,
            error=str(exc)[:2000],
            force=force,
        )
        return 0, 1
    persist_complete_hadith(
        state,
        book_id=book_id,
        source_path=source,
        payload=rec,
        output_dir=output_dir,
        force=force,
    )
    return 1, 0


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
    page: str | None = None,
    volume: int | str | None = None,
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
    # Targeted page runs are for debugging: do not resume mid-book, and do not
    # rewrite progress/buffer so a normal full run can continue afterward.
    page_filter = str(page).strip() if page else None
    last_source: str | None = None
    last_buf: OpenHadith | None = None
    try:
        for path in spec.files:
            if stop or (remaining is not None and remaining <= 0):
                break
            source = str(path)
            last_source = source
            all_units = prepare_units("hadith", parse_file(path), settings)
            if page_filter:
                target_indexes = [
                    i
                    for i, u in enumerate(all_units)
                    if locator_matches_page(u.locator, page_filter, volume)
                ]
                if not target_indexes:
                    continue
                ordered = [all_units[i] for i in target_indexes]
                next_by_pos = {
                    pos: (
                        all_units[target_indexes[pos] + 1].text
                        if target_indexes[pos] + 1 < len(all_units)
                        else None
                    )
                    for pos in range(len(target_indexes))
                }
                start_i = 0
                buf = None
            else:
                ordered = all_units
                next_by_pos = None
                target_indexes: list[int] = []
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
                if next_by_pos is not None:
                    next_text = next_by_pos[i]
                else:
                    next_text = ordered[i + 1].text if i + 1 < len(ordered) else None
                if should_skip(unit.text, settings.skip_min_chars):
                    skipped += 1
                    if not page_filter:
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
                    if not page_filter:
                        state.set_hadith_buffer(
                            book_id, source, buf.to_dict() if buf else None
                        )
                    stop = True
                    break
                except ProviderServerError as exc:
                    # Temporary Gemini overload (503). Save resume point and stop
                    # cleanly so the same CLI command can continue later.
                    errors += 1
                    logger.error(
                        "phase1 provider unavailable %s: %s", unit.locator, exc
                    )
                    if not page_filter:
                        state.set_hadith_buffer(
                            book_id, source, buf.to_dict() if buf else None
                        )
                    raise
                items = [it for it in (page_payload.get("hadiths") or []) if isinstance(it, dict)]
                complete, buf = consume_page(
                    unit.locator,
                    unit.text,
                    items,
                    buf,
                    next_text,
                    unit.quran_refs,
                    (unit.kitab, unit.bab),
                )
                last_buf = buf
                for rec in complete:
                    try:
                        rec = unify_assembled_hadith(agent, rec)
                    except StructuredOutputError as exc:
                        errors += 1
                        logger.error(
                            "phase1 unify mentions failed %s: %s",
                            rec.get("marker") or unit.locator,
                            exc,
                        )
                        persist_complete_hadith(
                            state,
                            book_id=book_id,
                            source_path=source,
                            payload=remap_hadith_payload(dict(rec)),
                            output_dir=output_dir,
                            status=ChunkStatus.ERROR,
                            error=str(exc)[:2000],
                            force=bool(page_filter),
                        )
                        continue
                    persist_complete_hadith(
                        state,
                        book_id=book_id,
                        source_path=source,
                        payload=rec,
                        output_dir=output_dir,
                        force=bool(page_filter),
                    )
                    flushed += 1
                if not page_filter:
                    state.set_hadith_buffer(book_id, source, buf.to_dict() if buf else None)
                    state.set_hadith_progress(book_id, source, unit.locator)
                pages += 1
                if remaining is not None:
                    remaining -= 1
                i += 1
            if stop:
                break
            # Targeted --page left an open buffer because the next printed page
            # continues the last hadith: fetch that page only to close the
            # spanning marker (do not flush that page's new numbered hadiths).
            if page_filter and buf and target_indexes:
                next_i = target_indexes[-1] + 1
                if next_i < len(all_units):
                    unit = all_units[next_i]
                    if not should_skip(unit.text, settings.skip_min_chars):
                        try:
                            page_payload = extract_hadith_page(agent, unit)
                        except StructuredOutputError as exc:
                            errors += 1
                            logger.error(
                                "phase1 JSON failed %s (lookahead): %s",
                                unit.locator,
                                exc,
                            )
                        except ProviderServerError:
                            raise
                        else:
                            items = [
                                it
                                for it in (page_payload.get("hadiths") or [])
                                if isinstance(it, dict)
                            ]
                            closed, buf = close_open_hadith_with_page(
                                unit.locator,
                                unit.text,
                                items,
                                buf,
                                unit.quran_refs,
                            )
                            last_buf = buf
                            if closed is not None:
                                try:
                                    rec = unify_assembled_hadith(agent, closed)
                                except StructuredOutputError as exc:
                                    errors += 1
                                    logger.error(
                                        "phase1 unify mentions failed %s: %s",
                                        closed.get("marker") or unit.locator,
                                        exc,
                                    )
                                    persist_complete_hadith(
                                        state,
                                        book_id=book_id,
                                        source_path=source,
                                        payload=remap_hadith_payload(dict(closed)),
                                        output_dir=output_dir,
                                        status=ChunkStatus.ERROR,
                                        error=str(exc)[:2000],
                                        force=True,
                                    )
                                else:
                                    persist_complete_hadith(
                                        state,
                                        book_id=book_id,
                                        source_path=source,
                                        payload=rec,
                                        output_dir=output_dir,
                                        force=True,
                                    )
                                    flushed += 1
            # End of volume, or end of a targeted page selection with an open
            # spanning buffer — flush so debug runs still write a payload.
            at_volume_end = not page_filter and i >= len(ordered)
            if buf and (at_volume_end or page_filter):
                n_flush, n_err = _flush_open_hadith(
                    state=state,
                    agent=agent,
                    book_id=book_id,
                    source=source,
                    buf=buf,
                    output_dir=output_dir,
                    force=bool(page_filter),
                )
                flushed += n_flush
                errors += n_err
                buf = None
                last_buf = None
                if not page_filter:
                    state.set_hadith_buffer(book_id, source, None)
        if page_filter and pages == 0:
            hint = f"page={page_filter}"
            if volume is not None and str(volume).strip():
                hint += f" volume={volume}"
            raise FileNotFoundError(
                f"No hadith pages matched {hint} in book '{book_id}'"
            )
    except AllKeysExhausted:
        if last_source is not None and not page_filter:
            state.set_hadith_buffer(
                book_id, last_source, last_buf.to_dict() if last_buf else None
            )
        state.finish_job(job_id, pause_reason="all_keys_exhausted")
        raise
    except ProviderServerError:
        if last_source is not None and not page_filter:
            state.set_hadith_buffer(
                book_id, last_source, last_buf.to_dict() if last_buf else None
            )
        state.finish_job(job_id, pause_reason="provider_unavailable")
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
    page: str | None = None,
    volume: int | str | None = None,
) -> dict[str, int]:
    settings = settings or get_settings()
    spec = resolve_book(book_id, raw_dir or settings.raw_data_dir)
    if spec.pipeline == "hadith":
        return run_hadith_phase1(
            book_id,
            state,
            agent,
            settings,
            raw_dir=raw_dir,
            limit=limit,
            page=page,
            volume=volume,
        )

    if page:
        raise ValueError("--page is only supported for hadith books")

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
            except ProviderServerError as exc:
                logger.error("phase1 provider unavailable %s: %s", chunk.locator, exc)
                raise
            if status == ChunkStatus.SKIPPED:
                skipped += 1
            else:
                processed += 1
    except AllKeysExhausted:
        state.finish_job(job_id, pause_reason="all_keys_exhausted")
        raise
    except ProviderServerError:
        state.finish_job(job_id, pause_reason="provider_unavailable")
        raise
    state.finish_job(job_id)
    return {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "pending_seen": len(pending),
    }
