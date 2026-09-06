"""Recover the book's own topical classification: kitab -> bab -> hadith.

Al-Kafi is already a classified corpus. Kulayni grouped every narration under a
kitab and a bab, and those headings are printed in the source text -- 69 of them
in volume 1, 278 in volume 2. Inheriting them costs nothing, needs no model, and
cannot hallucinate.

This matters because it makes connectivity independent of extraction quality.
Whatever the model does or does not emit for a given hadith, every narration in
كتاب العقل و الجهل still shares one high-frequency node with the other thirty-odd
in that kitab. The LLM's semantic_nodes become an enrichment on top of a graph
that is already connected, instead of the only thing holding it together.
"""

from __future__ import annotations

import logging
import re

from src.extractors.chunkers import HADITH_START_RE, iter_line_roles
from src.extractors.epub_parser import ParsedUnit

logger = logging.getLogger(__name__)

_KITAB = re.compile(r"^(?:كِتَابُ|كتاب)\s")
_BAB = re.compile(r"^(?:بَابُ|باب)\s")

# Prose that merely begins with the word "kitab" is not a heading. Real headings
# are bare noun phrases: no sentence punctuation, and short. The preface line
# "كتاب الحجّة و إن لم نكمّله على استحقاقه، لأنّا كرهنا..." is the case this
# rejects, along with footnote glosses like "باب منع أي مشى. و يطلق على...".
_SENTENCE_PUNCT = re.compile(r"[،.:؛!؟]")
_MAX_HEADING_CHARS = 90

# Footnote markers ride along on heading lines ("بَابُ النَّوَادِرِ [1]"), and an
# unterminated one can survive the line break.
_MARKER = re.compile(r"\s*\[\d{0,3}\]?\s*$")
# The basmala opens a kitab on the line after its title; it is not part of it.
_BASMALA = re.compile(r"^(?:بِسْمِ|بسم)\b")

# Headings wrap mid-phrase in this edition, and not always at a conjunction:
# al-Kafi 2 prints "بَابُ طِينَةِ" on one line and "الْمُؤْمِنِ وَ الْكَافِرِ" on the
# next. What is reliable is the layout -- nothing stands between a heading and
# its first numbered hadith except the rest of the heading -- so the title
# absorbs following lines until a hadith start, another heading, or the cap.
_MAX_JOINED_LINES = 2

# Babs named "miscellany" carry no topic and must not become a bucket key --
# every volume has one and they would merge unrelated narrations.
_EMPTY_TOPICS = ("النوادر", "نوادر")


def _strip_marks(text: str) -> str:
    return re.sub(r"[ـً-ٟۖ-ۭ]", "", text or "")


def _looks_like_heading(text: str) -> bool:
    if not text or len(text) > _MAX_HEADING_CHARS:
        return False
    if not (_KITAB.match(text) or _BAB.match(text)):
        return False
    return not _SENTENCE_PUNCT.search(text)


def is_empty_topic(heading: str) -> bool:
    """True for بَابُ النَّوَادِرِ and friends: a heading that names no subject."""
    plain = _strip_marks(heading)
    return any(word in plain for word in _EMPTY_TOPICS)


def _join_wrapped(lines: list[str], start: int) -> tuple[str, int]:
    """Reassemble a heading split across lines by the typesetter."""
    title = lines[start].strip()
    consumed = 0
    index = start + 1
    while consumed < _MAX_JOINED_LINES and index < len(lines):
        nxt = lines[index].strip()
        if not nxt:
            index += 1
            continue
        if HADITH_START_RE.match(nxt) or _looks_like_heading(nxt):
            break
        # Tested on the de-vocalized form: "بِسْمِ" carries marks that defeat a
        # word-boundary match.
        if _SENTENCE_PUNCT.search(nxt) or _BASMALA.match(_strip_marks(nxt)):
            break
        if len(title) + len(nxt) > _MAX_HEADING_CHARS:
            break
        title = f"{title} {nxt}"
        consumed += 1
        index += 1
    return clean_heading(title), index


def clean_heading(title: str) -> str:
    """Drop footnote residue and the zero-width joiner this edition leaves behind."""
    text = _MARKER.sub("", (title or "").strip())
    text = text.replace("‌", "").replace("ـ", "")
    return re.sub(r"\s+", " ", text).strip()


def page_headings(text: str) -> list[tuple[str, str]]:
    """(level, title) for each heading on this page, in order.

    Only matn lines are considered: a heading-shaped line inside a footnote is
    the editor glossing a bab title, not the bab itself.
    """
    matn = [line for line, role, _ in iter_line_roles(text) if role == "matn"]
    found: list[tuple[str, str]] = []
    index = 0
    while index < len(matn):
        stripped = matn[index].strip()
        if not _looks_like_heading(stripped):
            index += 1
            continue
        title, index = _join_wrapped(matn, index)
        found.append(("kitab" if _KITAB.match(title) else "bab", title))
    return found


def attach_sections(units: list[ParsedUnit]) -> list[ParsedUnit]:
    """Stamp every unit with the kitab and bab in force at that point.

    A heading applies from where it is printed until the next one of the same or
    higher level, so this is a running scan over the volume in reading order. A
    new kitab clears the current bab.
    """
    out: list[ParsedUnit] = []
    kitab = ""
    bab = ""
    for unit in units:
        for level, title in page_headings(unit.text):
            if level == "kitab":
                kitab, bab = title, ""
                logger.info("kitab: %s", title)
            else:
                bab = title
        out.append(
            ParsedUnit(
                locator=unit.locator,
                text=unit.text,
                source_path=unit.source_path,
                quran_refs=unit.quran_refs,
                kitab=kitab,
                bab=bab,
            )
        )
    return out


def section_nodes(kitab: str, bab: str) -> list[dict]:
    """The classification as graph nodes, skipping topic-free headings."""
    nodes: list[dict] = []
    if kitab and not is_empty_topic(kitab):
        nodes.append({"node": kitab, "type": "kitab"})
    if bab and not is_empty_topic(bab):
        nodes.append({"node": bab, "type": "bab"})
    return nodes
