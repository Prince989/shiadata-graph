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


def split_hadith_page(text: str) -> list[tuple[str, str]]:
    """Return (start_token, body) for each hadith on a Folklib page.

    Text before the first marker (bab title) is dropped, not glued onto hadith 1.
    """
    matches = list(HADITH_START_RE.finditer(text))
    if not matches:
        return []
    out: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        token = (match.group("kafi") or match.group("wasail") or match.group(0)).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) >= _MIN_HADITH_CHARS:
            out.append((token, body))
    return out


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
        text = unit.text.strip()
        if not text or _is_footnote_page(text):
            continue
        pieces = split_hadith_page(text)
        if pieces:
            started = True
            refined.append(unit)
            continue
        if started and len(text) >= _MIN_HADITH_CHARS:
            refined.append(unit)
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
