"""Split Folklib-style .txt books on --- [locator] --- banners."""

from __future__ import annotations

import re
from pathlib import Path

from src.extractors.epub_parser import ParsedUnit

BANNER_RE = re.compile(r"---\s*\[(.*?)\]\s*---")
AYAH_BANNER_RE = re.compile(r"سوره|آیات|آيه|آیه", re.I)


def parse_txt(path: Path | str) -> list[ParsedUnit]:
    source = Path(path)
    content = source.read_text(encoding="utf-8")
    parts = BANNER_RE.split(content)
    units: list[ParsedUnit] = []
    for i in range(1, len(parts), 2):
        locator = parts[i].strip()
        text = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if text:
            units.append(
                ParsedUnit(locator=locator, text=text, source_path=str(source))
            )
    if not units and content.strip():
        units.append(
            ParsedUnit(locator=source.stem, text=content.strip(), source_path=str(source))
        )
    return units


def is_ayah_locator(locator: str) -> bool:
    return bool(AYAH_BANNER_RE.search(locator))
