"""Turn parsed units into pipeline-specific Gemini inputs."""

from __future__ import annotations

from src.extractors.epub_parser import ParsedUnit
from src.extractors.txt_parser import is_ayah_locator


def hadith_units(units: list[ParsedUnit]) -> list[ParsedUnit]:
    return [u for u in units if u.text.strip()]


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
