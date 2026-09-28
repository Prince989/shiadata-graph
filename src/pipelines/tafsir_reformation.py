"""Translate stored tafsir units into the missing languages.

This is not Phase 1. Long sections are overlapping windows. Each window is
sent and translated in full (no summary). Resume is a `.draft.json` beside
the final file; pause is safe after any window.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, Field, create_model

from config.paths import OUTPUT_DIR
from config.settings import Settings, get_settings
from src.agents.errors import ProviderServerError, RequestTooLarge, StructuredOutputError
from src.agents.gemini import GeminiAgent
from src.extractors.epub_parser import ParsedUnit
from src.models import TafsirReformation
from src.pipelines.catalog import resolve_book
from src.pipelines.llm_processor import phase1_filename, should_skip
from src.pipelines.prompts import tafsir_reformation_prompt
from src.pipelines.runner import locator_matches_page, parse_file, prepare_units

logger = logging.getLogger(__name__)

_PERSIAN_LETTERS = re.compile(r"[پچژگ]")

OUTPUT_SUBDIR = "tafsir-reformation"
# Unique source per window (overlap is extra, also fully translated).
DEFAULT_WINDOW_CHARS = 520
DEFAULT_OVERLAP_CHARS = 160
MIN_WINDOW_CHARS = 160

_TARGET_KEYS = {
    "fa": ("tafsir_ar", "tafsir_en"),
    "ar": ("tafsir_fa", "tafsir_en"),
}


def infer_source_lang(text: str, catalog_lang: str = "") -> str:
    lang = str(catalog_lang or "").strip().lower()
    if lang in {"ar", "fa"}:
        return lang
    return "fa" if _PERSIAN_LETTERS.search(text or "") else "ar"


def llm_schema(source_lang: str) -> type[BaseModel]:
    """JSON the model may write: the two missing languages only."""
    if source_lang == "ar":
        return create_model(
            "TafsirReformation",
            tafsir_fa=(str, Field(..., min_length=1)),
            tafsir_en=(str, Field(..., min_length=1)),
        )
    return create_model(
        "TafsirReformation",
        tafsir_ar=(str, Field(..., min_length=1)),
        tafsir_en=(str, Field(..., min_length=1)),
    )


def targets_complete(payload: dict) -> bool:
    lang = str(payload.get("source_lang") or "")
    keys = _TARGET_KEYS.get(lang)
    if not keys:
        return False
    return all(str(payload.get(k) or "").strip() for k in keys)


def _window_complete(row: dict, source_lang: str) -> bool:
    keys = _TARGET_KEYS.get(source_lang) or ("tafsir_en",)
    return all(str(row.get(k) or "").strip() for k in keys)


def source_sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _cut_end(text: str, start: int, limit: int) -> int:
    hard = min(len(text), start + max(limit, 1))
    if hard >= len(text):
        return len(text)
    region = text[start:hard]
    min_keep = max(80, (hard - start) // 4)
    for sep in ("\n\n", "\n", "۔", ". ", "? ", "! ", "؟", "، ", " "):
        pos = region.rfind(sep)
        if pos >= min_keep:
            return start + pos + len(sep)
    return hard


def _cut_start(text: str, pos: int) -> int:
    if pos <= 0:
        return 0
    region = text[max(0, pos - 80) : pos]
    for sep in ("\n\n", "\n", ". ", "۔", " "):
        at = region.rfind(sep)
        if at >= 0:
            return max(0, pos - 80) + at + len(sep)
    return pos


def split_source_windows(
    text: str,
    *,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[dict]:
    """Overlapping character windows. Each row is the FULL slice sent to the model."""
    text = text or ""
    if not text:
        return []
    window_chars = max(MIN_WINDOW_CHARS, int(window_chars))
    overlap_chars = max(0, int(overlap_chars))
    rows: list[dict] = []
    unique_start = 0
    index = 0
    n = len(text)
    while unique_start < n:
        unique_end = _cut_end(text, unique_start, window_chars)
        if unique_end <= unique_start:
            unique_end = min(n, unique_start + window_chars)
        send_start = 0 if index == 0 else _cut_start(text, max(0, unique_start - overlap_chars))
        if send_start > unique_start:
            send_start = unique_start
        rows.append(
            {
                "i": index,
                "send_start": send_start,
                "send_end": unique_end,
                "unique_start": unique_start,
                "unique_end": unique_end,
            }
        )
        if unique_end >= n:
            break
        unique_start = unique_end
        index += 1
        if index > 10_000:
            break
    return rows


def _paras(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", str(text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _cut_is_clean(curr: str, index: int) -> str | None:
    """Return the suffix after `index` only when the cut is not mid-word or mid-sentence."""
    if index <= 0:
        return curr.strip()
    if index >= len(curr):
        return None
    if curr[index - 1].isalnum() and curr[index].isalnum():
        return None
    rest = curr[index:].lstrip()
    if not rest:
        return ""
    if rest[0].islower() or rest[0] in ",;:،؛":
        return None
    return rest


def seam_unique_suffix(
    *,
    prev_translation: str,
    curr_translation: str,
    overlap_source: str,
    window_source: str,
) -> str:
    """Drop the overlapping head of curr when the cut is exact and on a boundary.

    A guessed cut (character fraction of the overlap) used to leave a fragment
    such as "miracle, so that". If no boundary matches, the whole window is
    kept so unique sentences are not dropped.
    """
    curr = str(curr_translation or "")
    prev = str(prev_translation or "")
    overlap_source = str(overlap_source or "")
    window_source = str(window_source or "")
    if not curr.strip():
        return curr
    if not overlap_source.strip():
        return curr

    max_k = min(len(prev), len(curr))
    for k in range(max_k, 39, -1):
        if prev[-k:] == curr[:k]:
            rest = _cut_is_clean(curr, k)
            if rest is not None:
                return rest

    ov_n = len(_paras(overlap_source))
    curr_paras = _paras(curr)
    win_n = len(_paras(window_source))
    if ov_n and len(curr_paras) > ov_n and abs(len(curr_paras) - win_n) <= 3:
        rest = "\n\n".join(curr_paras[ov_n:]).strip()
        if rest and not (rest[0].islower() or rest[0] in ",;:،؛"):
            return rest

    matcher = SequenceMatcher(None, prev, curr, autojunk=False)
    best = 0
    for _a, b, size in matcher.get_matching_blocks():
        if b == 0 and size >= 40:
            best = max(best, size)
    if best:
        rest = _cut_is_clean(curr, best)
        if rest is not None:
            return rest

    return curr


def stitch_window_translations(text: str, windows: list[dict], field: str) -> str:
    pieces: list[str] = []
    prev = ""
    for i, row in enumerate(windows):
        curr = str(row.get(field) or "").strip()
        if not curr:
            continue
        if i == 0:
            pieces.append(curr)
            prev = curr
            continue
        send_start = int(row.get("send_start") or 0)
        unique_start = int(row.get("unique_start") or 0)
        send_end = int(row.get("send_end") or 0)
        overlap_source = text[send_start:unique_start]
        window_source = text[send_start:send_end]
        unique = seam_unique_suffix(
            prev_translation=prev,
            curr_translation=curr,
            overlap_source=overlap_source,
            window_source=window_source,
        )
        if unique.strip():
            pieces.append(unique.strip())
            prev = unique.strip()
        else:
            pieces.append(curr)
            prev = curr
    return "\n\n".join(p for p in pieces if p.strip())


def assemble_payload(unit: ParsedUnit, source_lang: str, generated: dict) -> dict:
    text = unit.text
    fa = text if source_lang == "fa" else str(generated.get("tafsir_fa") or "")
    ar = text if source_lang == "ar" else str(generated.get("tafsir_ar") or "")
    en = str(generated.get("tafsir_en") or "")
    return TafsirReformation(
        locator=unit.locator,
        source_lang=source_lang,
        tafsir_fa=fa,
        tafsir_ar=ar,
        tafsir_en=en,
    ).model_dump()


def output_path(book_id: str, unit: ParsedUnit, root: Path | None = None) -> Path:
    dest = (root or OUTPUT_DIR) / OUTPUT_SUBDIR / book_id
    name = phase1_filename(unit.source_path, unit.locator, "reformation")
    return dest / name


def draft_path(dest: Path) -> Path:
    return dest.with_suffix(".draft.json")


def _empty_draft(
    unit: ParsedUnit,
    source_lang: str,
    windows: list[dict],
    *,
    window_chars: int,
    overlap_chars: int,
) -> dict:
    return {
        "locator": unit.locator,
        "source_lang": source_lang,
        "source_sha": source_sha(unit.text),
        "source_chars": len(unit.text or ""),
        "window_chars": window_chars,
        "overlap_chars": overlap_chars,
        "windows": [dict(row) for row in windows],
    }


def _load_draft(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _split_row(row: dict, text: str) -> list[dict]:
    unique_start = int(row["unique_start"])
    unique_end = int(row["unique_end"])
    mid = _cut_end(text, unique_start, max(MIN_WINDOW_CHARS, (unique_end - unique_start) // 2))
    if mid <= unique_start or mid >= unique_end:
        mid = unique_start + max(1, (unique_end - unique_start) // 2)
    overlap = max(0, unique_start - int(row["send_start"]))
    first_send = int(row["send_start"])
    second_send = _cut_start(text, max(0, mid - overlap))
    return [
        {
            "i": 0,
            "send_start": first_send,
            "send_end": mid,
            "unique_start": unique_start,
            "unique_end": mid,
        },
        {
            "i": 0,
            "send_start": second_send,
            "send_end": unique_end,
            "unique_start": mid,
            "unique_end": unique_end,
        },
    ]


def _reindex(windows: list[dict]) -> None:
    for i, row in enumerate(windows):
        row["i"] = i


def _translate_window(
    agent: GeminiAgent,
    unit: ParsedUnit,
    source_lang: str,
    row: dict,
) -> dict:
    text = unit.text or ""
    slice_text = text[int(row["send_start"]) : int(row["send_end"])]
    total = int(row.get("total") or 0)
    prompt = (
        f"Locator: {unit.locator}\n"
        f"Window {int(row['i']) + 1}/{total or '?'} "
        f"(source chars {row['send_start']}-{row['send_end']} of {len(text)}).\n"
        "Translate this entire window. Do not summarize. Do not invent ayah numbers.\n\n"
        f"Text:\n{slice_text}"
    )
    result = agent.complete_structured(
        prompt,
        llm_schema(source_lang),
        system=tafsir_reformation_prompt(source_lang),
    )
    return result.model_dump()


def process_unit(
    agent: GeminiAgent,
    unit: ParsedUnit,
    *,
    source_lang: str,
    dest: Path,
    force: bool = False,
    budget: int | None = None,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> tuple[str, int]:
    """Translate one unit in overlapping windows.

    Returns (processed|skipped|partial|error, llm_calls_this_invocation).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    draft_file = draft_path(dest)
    if dest.exists() and not force:
        try:
            existing = json.loads(dest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if isinstance(existing, dict) and targets_complete(existing):
            return "skipped", 0

    planned = split_source_windows(
        unit.text, window_chars=window_chars, overlap_chars=overlap_chars
    )
    draft = None if force else _load_draft(draft_file)
    if (
        draft
        and draft.get("source_sha") == source_sha(unit.text)
        and isinstance(draft.get("windows"), list)
        and draft.get("windows")
    ):
        windows = list(draft["windows"])
    else:
        windows = [dict(row) for row in planned]
        draft = _empty_draft(
            unit,
            source_lang,
            windows,
            window_chars=window_chars,
            overlap_chars=overlap_chars,
        )
        _write_json(draft_file, draft)

    calls = 0
    i = 0
    while i < len(windows):
        if budget is not None and calls >= budget:
            _write_json(draft_file, draft)
            return "partial", calls
        row = windows[i]
        row["total"] = len(windows)
        if _window_complete(row, source_lang) and not force:
            i += 1
            continue
        try:
            generated = _translate_window(agent, unit, source_lang, row)
        except (RequestTooLarge, StructuredOutputError) as exc:
            split_output = isinstance(exc, RequestTooLarge) or (
                "truncated" in str(exc).lower()
            )
            if not split_output:
                logger.error(
                    "tafsir-reformation JSON failed %s w%s: %s",
                    unit.locator,
                    i,
                    exc,
                )
                _write_json(draft_file, draft)
                return "error", calls
            unique_len = int(row["unique_end"]) - int(row["unique_start"])
            if unique_len <= MIN_WINDOW_CHARS:
                logger.error(
                    "tafsir-reformation window still too large %s w%s (%s chars)",
                    unit.locator,
                    i,
                    unique_len,
                )
                _write_json(draft_file, draft)
                return "error", calls
            logger.warning(
                "tafsir-reformation split oversized window %s w%s (%s unique chars): %s",
                unit.locator,
                i,
                unique_len,
                exc,
            )
            halves = _split_row(row, unit.text or "")
            windows[i : i + 1] = halves
            _reindex(windows)
            draft["windows"] = windows
            _write_json(draft_file, draft)
            continue
        keys = _TARGET_KEYS[source_lang]
        for key in keys:
            row[key] = str(generated.get(key) or "")
        calls += 1
        draft["windows"] = windows
        _write_json(draft_file, draft)
        i += 1

    if not all(_window_complete(row, source_lang) for row in windows):
        _write_json(draft_file, draft)
        return "partial", calls

    text = unit.text or ""
    generated = {}
    if source_lang == "fa":
        generated["tafsir_ar"] = stitch_window_translations(text, windows, "tafsir_ar")
        generated["tafsir_en"] = stitch_window_translations(text, windows, "tafsir_en")
    else:
        generated["tafsir_fa"] = stitch_window_translations(text, windows, "tafsir_fa")
        generated["tafsir_en"] = stitch_window_translations(text, windows, "tafsir_en")
    payload = assemble_payload(unit, source_lang, generated)
    _write_json(dest, payload)
    rec = getattr(getattr(agent, "day_report", None), "record_item", None)
    if callable(rec):
        rec(unit.locator)
    try:
        draft_file.unlink()
    except OSError:
        pass
    return "processed", calls


def run_tafsir_reformation(
    book_id: str,
    agent: GeminiAgent,
    *,
    settings: Settings | None = None,
    limit: int | None = None,
    page: str | None = None,
    volume: int | str | None = None,
    output_dir: Path | None = None,
) -> dict[str, int]:
    settings = settings or get_settings()
    spec = resolve_book(book_id, settings.raw_data_dir)
    if spec.pipeline != "tafsir":
        raise ValueError(f"Book '{book_id}' is pipeline={spec.pipeline}, not tafsir")
    root = output_dir or OUTPUT_DIR
    processed = 0
    skipped = 0
    errors = 0
    partial = 0
    remaining = limit
    page_filter = bool(str(page or "").strip())
    for path in spec.files:
        units = prepare_units(spec.pipeline, parse_file(path), settings)
        for unit in units:
            if page_filter and not locator_matches_page(unit.locator, page, volume):
                continue
            if should_skip(unit.text, settings.skip_min_chars):
                skipped += 1
                continue
            if remaining is not None and remaining <= 0:
                return {
                    "processed": processed,
                    "skipped": skipped,
                    "partial": partial,
                    "errors": errors,
                }
            source_lang = infer_source_lang(unit.text, spec.source_lang)
            dest = output_path(book_id, unit, root)
            try:
                status, used = process_unit(
                    agent,
                    unit,
                    source_lang=source_lang,
                    dest=dest,
                    force=page_filter,
                    budget=remaining,
                )
            except ProviderServerError:
                raise
            if remaining is not None:
                remaining -= used
            if status == "processed":
                processed += 1
            elif status == "skipped":
                skipped += 1
            elif status == "partial":
                partial += 1
            else:
                errors += 1
    return {
        "processed": processed,
        "skipped": skipped,
        "partial": partial,
        "errors": errors,
    }
