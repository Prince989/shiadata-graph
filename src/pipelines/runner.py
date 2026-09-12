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


def _split_resume_buffer(
    raw: dict | None,
) -> tuple[OpenHadith | None, list[dict]]:
    """Load open spanning buffer + assembled hadiths still waiting on unify.

    Legacy rows were a bare OpenHadith dict. New rows wrap both channels so a
    quota death mid-unify cannot drop a closed multi-page narration while
    progress has already walked past its pages.
    """
    if not raw:
        return None, []
    if "open" in raw or "pending_unify" in raw:
        open_raw = raw.get("open")
        pending = [p for p in (raw.get("pending_unify") or []) if isinstance(p, dict)]
        buf = OpenHadith.from_dict(open_raw) if open_raw else None
        return buf, pending
    return OpenHadith.from_dict(raw), []


def _dump_resume_buffer(
    buf: OpenHadith | None, pending: list[dict] | None
) -> dict | None:
    pending = [p for p in (pending or []) if isinstance(p, dict)]
    if buf is None and not pending:
        return None
    return {
        "open": buf.to_dict() if buf else None,
        "pending_unify": pending,
    }


def _unify_and_persist_one(
    *,
    state: StateManager,
    agent: GeminiAgent,
    book_id: str,
    source: str,
    rec: dict,
    output_dir: Path,
    force: bool = False,
) -> bool:
    """Unify+persist one assembled hadith. True if flushed OK, False if ERROR row.

    Propagates AllKeysExhausted / ProviderServerError so the caller can stash
    this record and any still-waiting siblings in the resume buffer.
    """
    try:
        out = unify_assembled_hadith(agent, rec)
    except StructuredOutputError as exc:
        logger.error(
            "phase1 unify mentions failed %s: %s",
            rec.get("marker") or rec.get("locator"),
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
            force=force,
        )
        return False
    persist_complete_hadith(
        state,
        book_id=book_id,
        source_path=source,
        payload=out,
        output_dir=output_dir,
        force=force,
    )
    return True


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
    ok = _unify_and_persist_one(
        state=state,
        agent=agent,
        book_id=book_id,
        source=source,
        rec=buf.assemble(),
        output_dir=output_dir,
        force=force,
    )
    return (1, 0) if ok else (0, 1)


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
    last_pending: list[dict] = []
    try:
        for path in spec.files:
            if stop or (remaining is not None and remaining <= 0):
                break
            source = str(path)
            last_source = source
            all_units = prepare_units("hadith", parse_file(path), settings)
            pending: list[dict] = []
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
                buf, pending = _split_resume_buffer(
                    state.get_hadith_buffer(book_id, source)
                )
                # Finish hadiths that closed before quota died mid-unify, before
                # walking further pages (otherwise progress skips their span).
                while pending:
                    last_pending = list(pending)
                    last_buf = buf
                    if _unify_and_persist_one(
                        state=state,
                        agent=agent,
                        book_id=book_id,
                        source=source,
                        rec=pending[0],
                        output_dir=output_dir,
                    ):
                        flushed += 1
                    else:
                        errors += 1
                    pending.pop(0)
                    state.set_hadith_buffer(
                        book_id, source, _dump_resume_buffer(buf, pending)
                    )
                last_pending = []
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
                            book_id, source, _dump_resume_buffer(buf, pending)
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
                            book_id, source, _dump_resume_buffer(buf, pending)
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
                pending = list(complete)
                last_pending = list(pending)
                while pending:
                    last_pending = list(pending)
                    if _unify_and_persist_one(
                        state=state,
                        agent=agent,
                        book_id=book_id,
                        source=source,
                        rec=pending[0],
                        output_dir=output_dir,
                        force=bool(page_filter),
                    ):
                        flushed += 1
                    else:
                        errors += 1
                    pending.pop(0)
                last_pending = []
                if not page_filter:
                    state.set_hadith_buffer(
                        book_id, source, _dump_resume_buffer(buf, pending)
                    )
                    state.set_hadith_progress(book_id, source, unit.locator)
                pages += 1
                if remaining is not None:
                    remaining -= 1
                i += 1
            if stop:
                break
            # Targeted --page left an open buffer because the next printed page
            # continues the last hadith: follow pages until the spanning marker
            # closes (do not flush those pages' new numbered hadiths).
            if page_filter and buf and target_indexes:
                next_i = target_indexes[-1] + 1
                lookahead = 0
                max_lookahead = 12
                while buf is not None and next_i < len(all_units) and lookahead < max_lookahead:
                    unit = all_units[next_i]
                    following = (
                        all_units[next_i + 1].text
                        if next_i + 1 < len(all_units)
                        else None
                    )
                    if should_skip(unit.text, settings.skip_min_chars):
                        next_i += 1
                        lookahead += 1
                        continue
                    try:
                        page_payload = extract_hadith_page(agent, unit)
                    except StructuredOutputError as exc:
                        errors += 1
                        logger.error(
                            "phase1 JSON failed %s (lookahead): %s",
                            unit.locator,
                            exc,
                        )
                        break
                    except ProviderServerError:
                        raise
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
                        next_text=following,
                        force_close=False,
                    )
                    last_buf = buf
                    lookahead += 1
                    next_i += 1
                    logger.info(
                        "phase1 --page lookahead %s (open=%s)",
                        unit.locator,
                        buf is not None,
                    )
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
                        break
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
            # Keep assembled-but-unflushed hadiths in pending_unify. Do not rewind
            # progress when that stash exists — resume finishes unify first, then
            # continues from last_locator without re-paying for those pages.
            state.set_hadith_buffer(
                book_id,
                last_source,
                _dump_resume_buffer(last_buf, last_pending),
            )
            if last_pending:
                logger.warning(
                    "phase1 saved %s pending unify hadith(s); first=%s",
                    len(last_pending),
                    last_pending[0].get("marker") or last_pending[0].get("locator"),
                )
        state.finish_job(job_id, pause_reason="all_keys_exhausted")
        raise
    except ProviderServerError:
        if last_source is not None and not page_filter:
            state.set_hadith_buffer(
                book_id,
                last_source,
                _dump_resume_buffer(last_buf, last_pending),
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
