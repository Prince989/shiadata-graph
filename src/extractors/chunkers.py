"""Turn parsed units into pipeline-specific Gemini inputs."""

from __future__ import annotations

import re

from src.extractors.epub_parser import ParsedUnit
from src.extractors.txt_parser import is_ayah_locator

# ASCII N-  (al-Kafi, al-Khisal). Eastern digits are reserved for Wasa'il headers
# so that "١ ـ باب" / "١ ـ الكافي ٢ : ٤٦٤" are not treated as new hadiths.
_DIGIT = r"[0-9\u0660-\u0669\u06F0-\u06F9]"
_DASH = r"[-–—ـ]"

HADITH_START_RE = re.compile(
    rf"^(?:(?P<kafi>[0-9]+\s*[-–—])|(?P<wasail>\[\s*{_DIGIT}+\s*\]\s*{_DIGIT}+\s*{_DASH}))",
    re.MULTILINE,
)

_MIN_HADITH_CHARS = 20

# Folklib editor notes: ASCII [1] at line start. Do not touch Wasa'il "[ ١٥٤٩٥ ]".
_FOOTNOTE_LINE = re.compile(r"^[ \t]*\[\d{1,3}\]")
_INLINE_FOOTNOTE_REF = re.compile(r"\[\d{1,3}\]")
_HARKAT = re.compile(r"[\u064B-\u0652]")
_NOTE_CONTINUATION = re.compile(
    r"^(أي|اى|في بعض|مضمون|و السبب|والسبب|أي خروجه|و في بعض)",
)


def _looks_like_matn_resume(stripped: str) -> bool:
    if not stripped:
        return False
    if HADITH_START_RE.match(stripped):
        return True
    if _FOOTNOTE_LINE.match(stripped) or _NOTE_CONTINUATION.match(stripped):
        return False
    if re.match(r"^[\u0600-\u06FF]", stripped) and _HARKAT.search(stripped):
        return True
    return False


def strip_folklib_footnotes(text: str) -> str:
    """Drop editor footnotes ([1] أي …) and leftover inline [n] markers from matn."""
    out: list[str] = []
    in_note = False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if _FOOTNOTE_LINE.match(line) or _FOOTNOTE_LINE.match(stripped):
            in_note = True
            continue
        if in_note:
            if not stripped:
                continue
            if _looks_like_matn_resume(stripped):
                in_note = False
                out.append(line)
            continue
        out.append(line)
    joined = "\n".join(out)
    joined = _INLINE_FOOTNOTE_REF.sub("", joined)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def page_prefix_and_starts(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split a page into leading text (before the first numbered start) and starts.

    Unlike split_hadith_page, the prefix is kept: on a continuation page it is the
    rest of the previous hadith, not a bab title to drop.
    """
    matches = list(HADITH_START_RE.finditer(text))
    if not matches:
        return text.strip(), []
    leading = text[: matches[0].start()].strip()
    starts: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        token = (match.group("kafi") or match.group("wasail") or match.group(0)).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            starts.append((token, body))
    return leading, starts


def split_hadith_page(text: str) -> list[tuple[str, str]]:
    """Return (start_token, body) for each numbered hadith; drop prefix and short bodies."""
    _, starts = page_prefix_and_starts(text)
    return [(token, body) for token, body in starts if len(body) >= _MIN_HADITH_CHARS]


def next_page_continues(next_text: str | None) -> bool:
    """True if the following page still belongs to the hadith that ended the previous page."""
    if not next_text or not next_text.strip():
        return False
    leading, starts = page_prefix_and_starts(next_text)
    if not starts:
        return True
    return bool(leading)


def _is_footnote_page(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    footnoteish = sum(1 for ln in lines if re.match(r"^\[\d+\]", ln))
    return footnoteish >= max(1, len(lines) // 2)


def hadith_units(units: list[ParsedUnit]) -> list[ParsedUnit]:
    """One Gemini unit per printed page. Intro pages before the first numbered
    hadith are skipped; continuation pages after that stay their own units.
    """
    refined: list[ParsedUnit] = []
    started = False
    for unit in units:
        text = strip_folklib_footnotes(unit.text.strip())
        if not text or _is_footnote_page(text):
            continue
        pieces = split_hadith_page(text)
        if pieces:
            started = True
            refined.append(
                ParsedUnit(locator=unit.locator, text=text, source_path=unit.source_path)
            )
            continue
        if started and len(text) >= _MIN_HADITH_CHARS:
            refined.append(
                ParsedUnit(locator=unit.locator, text=text, source_path=unit.source_path)
            )
    return refined


def tafsir_ayah_units(units: list[ParsedUnit]) -> list[ParsedUnit]:
    ayah = [u for u in units if is_ayah_locator(u.locator)]
    return ayah if ayah else units


def pack_history(
    units: list[ParsedUnit],
    pages_per_call: int,
    max_chars: int,
) -> list[ParsedUnit]:
    packed: list[ParsedUnit] = []
    buf: list[ParsedUnit] = []
    size = 0
    for unit in units:
        next_size = size + len(unit.text)
        if buf and (len(buf) >= pages_per_call or next_size > max_chars):
            packed.append(_merge(buf))
            buf = []
            size = 0
        buf.append(unit)
        size += len(unit.text)
    if buf:
        packed.append(_merge(buf))
    return packed


def _merge(group: list[ParsedUnit]) -> ParsedUnit:
    first = group[0]
    last = group[-1]
    locator = first.locator if first.locator == last.locator else f"{first.locator} … {last.locator}"
    text = "\n\n".join(u.text for u in group)
    return ParsedUnit(locator=locator, text=text, source_path=first.source_path)
