"""Resolve Qur'anic citations in a hadith page to "sura:ayah" strings.

Two independent signals, because neither covers the corpus alone:

* The Folklib editor footnotes a great many quotations ("[2] البقرة: 269"),
  and those footnotes are authoritative -- they even flag where al-Kafi's
  wording differs from the mushaf, which phrase matching cannot see.
* But the editor does not footnote everything. Hadith 5's
  «فَاعْتَبِرُوا يا أُولِي الْأَبْصارِ» carries no marker at all, so it can only be
  found by matching the matn against the actual text of the Qur'an.

Refs are deliberately kept out of the Gemini schema and injected here, so the
model has no channel through which to invent a citation.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

from config.paths import quran_corpus_path

logger = logging.getLogger(__name__)

# Qur'anic orthography writes some letters as marks. U+0670 is an alef and must
# become one; every other mark in these ranges is a vowel, a recitation symbol
# or tatweel and is dropped. Getting this wrong is not cosmetic: delete U+0670
# instead of folding it and ٱلْأَبْصَٰرِ stops matching الأبصار entirely.
_SUPERSCRIPT_ALEF = "ٰ"
_MARKS = re.compile(r"[ً-ٟۖ-ۭـ]")
_NON_ARABIC = re.compile(r"[^؀-ۿ ]")


def fold(text: str) -> str:
    """Collapse one Arabic string to its comparison key."""
    s = (text or "").replace(_SUPERSCRIPT_ALEF, "ا")
    s = _MARKS.sub("", s)
    s = re.sub(r"[ٱآأإ]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    s = _NON_ARABIC.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# (name, ayah count). Counts are the real ones, so an out-of-range number
# rejects the parse -- the corpus contains cross-reference footnotes such as
# "قد مر الحديث ص 321" whose "ص 321" would otherwise read as Sad:321.
_SURAS: dict[int, tuple[str, int]] = {
    1: ("الفاتحة", 7), 2: ("البقرة", 286), 3: ("آل عمران", 200), 4: ("النساء", 176),
    5: ("المائدة", 120), 6: ("الأنعام", 165), 7: ("الأعراف", 206), 8: ("الأنفال", 75),
    9: ("التوبة", 129), 10: ("يونس", 109), 11: ("هود", 123), 12: ("يوسف", 111),
    13: ("الرعد", 43), 14: ("إبراهيم", 52), 15: ("الحجر", 99), 16: ("النحل", 128),
    17: ("الإسراء", 111), 18: ("الكهف", 110), 19: ("مريم", 98), 20: ("طه", 135),
    21: ("الأنبياء", 112), 22: ("الحج", 78), 23: ("المؤمنون", 118), 24: ("النور", 64),
    25: ("الفرقان", 77), 26: ("الشعراء", 227), 27: ("النمل", 93), 28: ("القصص", 88),
    29: ("العنكبوت", 69), 30: ("الروم", 60), 31: ("لقمان", 34), 32: ("السجدة", 30),
    33: ("الأحزاب", 73), 34: ("سبأ", 54), 35: ("فاطر", 45), 36: ("يس", 83),
    37: ("الصافات", 182), 38: ("ص", 88), 39: ("الزمر", 75), 40: ("غافر", 85),
    41: ("فصلت", 54), 42: ("الشورى", 53), 43: ("الزخرف", 89), 44: ("الدخان", 59),
    45: ("الجاثية", 37), 46: ("الأحقاف", 35), 47: ("محمد", 38), 48: ("الفتح", 29),
    49: ("الحجرات", 18), 50: ("ق", 45), 51: ("الذاريات", 60), 52: ("الطور", 49),
    53: ("النجم", 62), 54: ("القمر", 55), 55: ("الرحمن", 78), 56: ("الواقعة", 96),
    57: ("الحديد", 29), 58: ("المجادلة", 22), 59: ("الحشر", 24), 60: ("الممتحنة", 13),
    61: ("الصف", 14), 62: ("الجمعة", 11), 63: ("المنافقون", 11), 64: ("التغابن", 18),
    65: ("الطلاق", 12), 66: ("التحريم", 12), 67: ("الملك", 30), 68: ("القلم", 52),
    69: ("الحاقة", 52), 70: ("المعارج", 44), 71: ("نوح", 28), 72: ("الجن", 28),
    73: ("المزمل", 20), 74: ("المدثر", 56), 75: ("القيامة", 40), 76: ("الإنسان", 31),
    77: ("المرسلات", 50), 78: ("النبأ", 40), 79: ("النازعات", 46), 80: ("عبس", 42),
    81: ("التكوير", 29), 82: ("الانفطار", 19), 83: ("المطففين", 36), 84: ("الانشقاق", 25),
    85: ("البروج", 22), 86: ("الطارق", 17), 87: ("الأعلى", 19), 88: ("الغاشية", 26),
    89: ("الفجر", 30), 90: ("البلد", 20), 91: ("الشمس", 15), 92: ("الليل", 21),
    93: ("الضحى", 11), 94: ("الشرح", 8), 95: ("التين", 8), 96: ("العلق", 19),
    97: ("القدر", 5), 98: ("البينة", 8), 99: ("الزلزلة", 8), 100: ("العاديات", 11),
    101: ("القارعة", 11), 102: ("التكاثر", 8), 103: ("العصر", 3), 104: ("الهمزة", 9),
    105: ("الفيل", 5), 106: ("قريش", 4), 107: ("الماعون", 7), 108: ("الكوثر", 3),
    109: ("الكافرون", 6), 110: ("النصر", 3), 111: ("المسد", 5), 112: ("الإخلاص", 4),
    113: ("الفلق", 5), 114: ("الناس", 6),
}

# Classical names this edition actually prints. المؤمن for غافر and
# بني إسرائيل for الإسراء both occur in al-Kafi vol. 1.
_ALTERNATE_NAMES: dict[str, int] = {
    "المؤمن": 40,
    "بني إسرائيل": 17,
    "بنى إسرائيل": 17,
    "الإسرى": 17,
    "حم السجدة": 41,
    "المنافقين": 63,
    "الرحمان": 55,
    "الملائكة": 35,
    "التوبة براءة": 9,
    "براءة": 9,
    "الأنسان": 76,
    "الدهر": 76,
    "المطفّفين": 83,
    "النبيّ": 66,
}

_STRIP_SURA_WORD = re.compile(r"^(?:سورة|سوره|سورهٔ)\s+")


@lru_cache(maxsize=1)
def _name_index() -> dict[str, int]:
    table: dict[str, int] = {}
    for number, (name, _) in _SURAS.items():
        table[fold(name)] = number
    for name, number in _ALTERNATE_NAMES.items():
        table.setdefault(fold(name), number)
    return table


def ayah_count(sura: int) -> int:
    entry = _SURAS.get(sura)
    return entry[1] if entry else 0


def sura_number(name: str) -> int | None:
    folded = fold(_STRIP_SURA_WORD.sub("", (name or "").strip()))
    return _name_index().get(folded)


# "[2] البقرة: 269 و فيها …", "[5] الرعد 41.", "يونس، 39." -- the separator is
# a colon, an Arabic comma, or nothing at all, and commentary may trail.
_FOOTNOTE_REF_RE = re.compile(
    r"^\s*(?:\[\d{1,3}\]\s*)?"
    r"(?P<name>[؀-ۿ][؀-ۿ\s]{0,24}?)"
    r"\s*[:：،,\-]?\s*"
    r"(?P<ayah>\d{1,3})(?!\d)"
)


def parse_footnote_ref(body: str) -> str | None:
    """Return "sura:ayah" when this footnote opens with a Qur'an citation.

    The sura name must be the whole leading segment, which is what keeps
    cross-references like "و يأتي في ج 5 ص 87" from parsing: their leading
    segment is not a sura name, so nothing resolves.
    """
    match = _FOOTNOTE_REF_RE.match(body or "")
    if not match:
        return None
    sura = sura_number(match.group("name"))
    if sura is None:
        return None
    ayah = int(match.group("ayah"))
    if not 1 <= ayah <= ayah_count(sura):
        logger.debug("reject %s:%s -- out of range", sura, ayah)
        return None
    return f"{sura}:{ayah}"


@lru_cache(maxsize=1)
def _quran_index() -> tuple[tuple[str, str], ...]:
    """(ref, folded-and-despaced ayah text) for all 6236 ayat, or () if absent.

    Spaces are removed because the mushaf and the hadith text disagree about
    word division: the mushaf writes يَٰٓأُو۟لِى as one token where al-Kafi prints
    يا أُولِي as two.
    """
    path = quran_corpus_path()
    if path is None:
        logger.info("no Qur'an corpus available; phrase matching disabled")
        return ()
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("could not read Qur'an corpus %s: %s", path, exc)
        return ()
    out: list[tuple[str, str]] = []
    for row in rows:
        try:
            ref = f"{int(row['surah_number'])}:{int(row['ayah_number'])}"
            text = fold(str(row["arabic_text"])).replace(" ", "")
        except (KeyError, TypeError, ValueError):
            continue
        if text:
            out.append((ref, text))
    return tuple(out)


# Below this, a "match" is a common turn of phrase rather than a quotation.
# Roughly three Arabic words once spaces are gone.
_GRAM = 16
_MAX_PHRASE_REFS = 3


@lru_cache(maxsize=1)
def _gram_index() -> dict[str, frozenset[str]]:
    """Every _GRAM-character window of every ayah, mapped to the refs holding it.

    Scanning all 6236 ayat per candidate phrase is quadratic and took minutes on
    a single volume. Any quotation long enough to count contains at least one
    full window, so one dictionary lookup per position finds it instead.
    """
    building: dict[str, set[str]] = {}
    for ref, text in _quran_index():
        for i in range(len(text) - _GRAM + 1):
            building.setdefault(text[i : i + _GRAM], set()).add(ref)
    return {gram: frozenset(refs) for gram, refs in building.items()}


def match_quran_phrases(text: str) -> list[str]:
    """Refs for verses quoted verbatim in `text`, longest quotation first.

    Comparison is space-insensitive: the mushaf writes يَٰٓأُو۟لِى as one token
    where al-Kafi prints يا أُولِي as two, and word-aligned matching misses every
    such quotation.

    A phrase can legitimately sit in more than one verse -- إلا أولوا الألباب is
    both 2:269 and 3:7 -- so every locus is returned and the caller decides.
    """
    if not _quran_index():
        return []
    squashed = fold(text).replace(" ", "")
    if len(squashed) < _GRAM:
        return []
    index = _gram_index()
    weights: dict[str, int] = {}
    for i in range(len(squashed) - _GRAM + 1):
        for ref in index.get(squashed[i : i + _GRAM], ()):
            weights[ref] = weights.get(ref, 0) + 1
    if not weights:
        return []
    # More matching windows means a longer verbatim run, so the best-supported
    # ref leads. Ties keep mushaf order, which reads as "earliest sura first".
    ranked = sorted(
        weights.items(),
        key=lambda kv: (-kv[1], int(kv[0].split(":")[0]), int(kv[0].split(":")[1])),
    )
    best = ranked[0][1]
    return [ref for ref, score in ranked if score == best][:_MAX_PHRASE_REFS]


# Roots per window for tier 2. Four content roots is a quotation; three is a
# turn of phrase ordinary Arabic prose also produces.
_ROOT_GRAM = 4
_MAX_ROOT_REFS = 3


@lru_cache(maxsize=1)
def _quran_rows() -> tuple[tuple[str, str], ...]:
    """(ref, raw arabic) for every ayah, or () when the corpus is unavailable."""
    path = quran_corpus_path()
    if path is None:
        return ()
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    out: list[tuple[str, str]] = []
    for row in rows:
        try:
            out.append(
                (
                    f"{int(row['surah_number'])}:{int(row['ayah_number'])}",
                    str(row["arabic_text"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(out)


def _roots_of(text: str) -> list[str]:
    from src.pipelines.morphology import root

    return [r for r in (root(word) for word in fold(text).split()) if len(r) >= 2]


@lru_cache(maxsize=1)
def _root_index() -> dict[tuple[str, ...], frozenset[str]]:
    """Root n-grams of every ayah, mapped to the refs containing them.

    Tier 2. Letter-level matching fails whenever al-Kafi's wording differs from
    the mushaf, and it often does: hadith 11 reads وَ مَا يَتَذَكَّرُ where 2:269 has
    وَمَا يَذَّكَّرُ. Both reduce to the root ذكر, so comparing root sequences finds
    the quotation that comparing letters cannot.
    """
    building: dict[tuple[str, ...], set[str]] = {}
    for ref, text in _quran_rows():
        roots = _roots_of(text)
        for i in range(len(roots) - _ROOT_GRAM + 1):
            building.setdefault(tuple(roots[i : i + _ROOT_GRAM]), set()).add(ref)
    return {gram: frozenset(refs) for gram, refs in building.items()}


def match_quran_roots(text: str) -> list[str]:
    """Refs for verses quoted with wording that differs from the mushaf."""
    index = _root_index()
    if not index:
        return []
    roots = _roots_of(text)
    if len(roots) < _ROOT_GRAM:
        return []
    weights: dict[str, int] = {}
    for i in range(len(roots) - _ROOT_GRAM + 1):
        for ref in index.get(tuple(roots[i : i + _ROOT_GRAM]), ()):
            weights[ref] = weights.get(ref, 0) + 1
    if not weights:
        return []
    ranked = sorted(
        weights.items(),
        key=lambda kv: (-kv[1], int(kv[0].split(":")[0]), int(kv[0].split(":")[1])),
    )
    best = ranked[0][1]
    return [ref for ref, score in ranked if score == best][:_MAX_ROOT_REFS]


def match_quran(text: str) -> list[str]:
    """Every tier, verbatim first. Deduped, order preserved."""
    found = list(match_quran_phrases(text))
    for ref in match_quran_roots(text):
        if ref not in found:
            found.append(ref)
    return found


_LEADING_MARKER_RE = re.compile(r"^\s*\[\d{1,3}\]\s*")
_INLINE_MARKER_RE = re.compile(r"\[(\d{1,3})\]")

CONTINUATION_KEY = "continuation"


def page_quran_refs(raw_text: str) -> dict[str, list[str]]:
    """Map each hadith start token on this page to the verses it cites.

    Must be given the RAW page: `strip_folklib_footnotes` deletes both the
    footnote bodies and the inline [n] markers, and the markers are the only
    thing tying a footnote to the hadith that earned it. Text before the first
    numbered start is keyed CONTINUATION_KEY, since it belongs to the hadith
    carried over from the previous page.
    """
    from src.extractors.chunkers import HADITH_START_RE, iter_line_roles

    segments: dict[str, list[str]] = {CONTINUATION_KEY: []}
    markers: dict[str, list[str]] = {CONTINUATION_KEY: []}
    footnotes: dict[str, list[str]] = {}

    current = CONTINUATION_KEY
    for line, role, note_number in iter_line_roles(raw_text):
        stripped = line.strip()
        if role == "note":
            if not stripped or note_number is None:
                continue
            footnotes.setdefault(note_number, []).append(
                _LEADING_MARKER_RE.sub("", stripped)
            )
            continue
        start = HADITH_START_RE.match(stripped)
        if start:
            current = stripped[: start.end()].strip()
            segments.setdefault(current, [])
            markers.setdefault(current, [])
        segments[current].append(stripped)
        markers[current].extend(_INLINE_MARKER_RE.findall(stripped))

    # A note only yields a ref when it OPENS with a sura name. That is what
    # separates a citation ("[2] البقرة: 269") from commentary that merely
    # quotes scripture in passing ("[3] فهو يعلم ان الوسوسة … «مِنْ شَرِّ …»"),
    # which belongs to the editor and not to the hadith.
    resolved = {
        number: parse_footnote_ref(" ".join(body))
        for number, body in footnotes.items()
    }

    out: dict[str, list[str]] = {}
    for key, lines in segments.items():
        refs: list[str] = []
        for number in markers.get(key, []):
            ref = resolved.get(number)
            if ref and ref not in refs:
                refs.append(ref)
        # Footnotes first: the editor disambiguates wording that sits in more
        # than one verse, and flags where al-Kafi differs from the mushaf.
        for ref in match_quran("\n".join(lines)):
            if ref not in refs:
                refs.append(ref)
        if refs:
            out[key] = refs
    return out
