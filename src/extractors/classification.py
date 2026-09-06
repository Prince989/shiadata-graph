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

# Wasa'il numbers every bab and closes the line with a colon:
#     ١ ـ باب وجوب العبادات الخمس :
# Requiring the line to START with باب, and rejecting any line containing a
# colon, made all 11,805 of its headings invisible -- the single richest topical
# index in the corpus, silently skipped by two characters of regex.
_NUMBER_PREFIX = r"(?:[٠-٩0-9]{1,4}\s*[ـ\-–—]\s*)?"
_KITAB = re.compile(rf"^{_NUMBER_PREFIX}(?:كِتَابُ|كتاب)\s")
_BAB = re.compile(rf"^{_NUMBER_PREFIX}(?:بَابُ|باب)\s")
# The separator is optional here: `clean_heading` strips tatweel, so by the time
# a title reaches `heading_topic` the `١ ـ باب` has already become `١ باب`.
_HEADING_PREFIX = re.compile(
    r"^(?:[٠-٩0-9]{1,4}\s*[ـ\-–—]?\s*)?(?:كِتَابُ|كتاب|بَابُ|باب)\s+"
)

# Prose that merely begins with the word "kitab" is not a heading. Real headings
# are bare noun phrases: no sentence punctuation, and short. The preface line
# "كتاب الحجّة و إن لم نكمّله على استحقاقه، لأنّا كرهنا..." is the case this
# rejects, along with footnote glosses like "باب منع أي مشى. و يطلق على...".
# A TRAILING colon is punctuation of the heading itself, not of a sentence.
_SENTENCE_PUNCT = re.compile(r"[،.؛!؟:]")
_TRAILING_COLON = re.compile(r"\s*:\s*$")
_MAX_HEADING_CHARS = 120

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

# A long kitab is printed across several volumes, and the publisher stamps the
# division into the title: `كتاب الصلاة القسم الثالث`. Wasa'il also titles the
# fihrist that opens or closes a volume `كتاب الحج فهرس`. Neither the part
# number nor the word fihrist names a subject -- they are apparatus -- and
# leaving them attached forged five parents (الصلاة القسم الثالث/الرابع/الخامس,
# الطهارة القسم الثاني, الحج فهرس) out of books the corpus already had under
# their real names. Stripping them merges those spans onto the one kitab they
# always were.
#
# A trailing bare conjunction is the same kind of damage from the other end: a
# title that wrapped mid-phrase where `_join_wrapped` could not follow it, as in
# the fihrist row `كتاب الفرائض و` whose المواريث sits past a page number. No
# Arabic noun phrase ends in a dangling و, so the truncation is recoverable
# without guessing what was cut off.
_APPARATUS = re.compile(r"\s*(?:القسم\s+\S+|فهرست?)\s*$")
_DANGLING_CONJUNCTION = re.compile(r"\s+و\s*$")


def _strip_marks(text: str) -> str:
    return re.sub(r"[ـً-ٟۖ-ۭ]", "", text or "")


def _names_a_book(topic: str) -> bool:
    """True when the words after كتاب form a title rather than a sentence.

    كتاب is also an ordinary noun -- a letter, a scripture, a written thing --
    and prose using it that way was opening spurious books mid-volume, each one
    stealing the babs of the real kitab it interrupted: `كتاب اللَّه حقّ` ("the
    Book of God is truth") took 26, `كتاب جليل و إذا فيه` ("a magnificent book,
    and therein...") took 36.

    A title is a definite noun phrase. It is definite itself (الصلاة) or is the
    head of an annexation to a definite noun (فضل العلم, معاني الأخبار), and
    every later word is definite too, conjoined, or governed by a preposition.
    A bare indefinite word in final position is a predicate, which is what makes
    the line a sentence. Checked against all 80 kitab titles the corpus prints:
    none is rejected.
    """
    words = _strip_marks(topic).split()
    if not words:
        return False
    annexed = len(words) > 1 and words[1].startswith("ال")
    if not words[0].startswith("ال") and not annexed:
        return False
    last = words[-1]
    if len(words) > 1 and not last.startswith(("ال", "و", "ب", "ل", "ف")):
        return False
    return True


def _looks_like_heading(text: str) -> bool:
    if not text or len(text) > _MAX_HEADING_CHARS:
        return False
    if not (_KITAB.match(text) or _BAB.match(text)):
        return False
    if _SENTENCE_PUNCT.search(_TRAILING_COLON.sub("", text)):
        return False
    if _KITAB.match(text) and not _names_a_book(heading_topic(text)):
        return False
    return True


def heading_topic(title: str) -> str:
    """The heading with its number, its كتاب/باب word and its colon removed.

    `١ ـ باب وجوب العبادات الخمس :` becomes `وجوب العبادات الخمس`, which is the
    part that names a subject.
    """
    text = _TRAILING_COLON.sub("", clean_heading(title))
    text = _HEADING_PREFIX.sub("", text).strip()
    text = _APPARATUS.sub("", text).strip()
    return _DANGLING_CONJUNCTION.sub("", text).strip()


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

    The fihrist pages are read like any other. They are dense -- a Wasa'il index
    lists 300-odd bab titles in a dozen pages -- but they are the book's own
    table of contents, grouped under the book's own kitab, and for the volumes
    whose body headings this parser cannot see they are the only record of those
    chapters at all: dropping them cost Wasa'il vol 4 all but two of its 181
    titles, and cost the catalog seven real parents (الرهن, الضمان, الوصية,
    المزارعة والمساقاة among them) to remove one malformed one.
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
