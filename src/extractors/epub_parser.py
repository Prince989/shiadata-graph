"""Parse EPUB (XHTML inside ZIP) into heading-scoped text units."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub


@dataclass(frozen=True)
class ParsedUnit:
    locator: str
    text: str
    source_path: str


def strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def parse_epub(path: Path | str) -> list[ParsedUnit]:
    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    units: list[ParsedUnit] = []
    source = str(path)
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        html = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        heading = soup.find(["h1", "h2", "h3"])
        locator = heading.get_text(" ", strip=True) if heading else item.get_name()
        text = strip_html(html)
        if text:
            units.append(ParsedUnit(locator=locator, text=text, source_path=source))
    return units
