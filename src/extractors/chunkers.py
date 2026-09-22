"""Turn parsed units into pipeline-specific Gemini inputs."""

from __future__ import annotations

import re
from collections.abc import Iterator

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
_FOOTNOTE_LINE = re.compile(r"^[ \t]*\[(\d{1,3})\]")
_INLINE_FOOTNOTE_REF = re.compile(r"\[\d{1,3}\]")
_HARKAT = re.compile(r"[\u064B-\u0652]")
_NOTE_CONTINUATION = re.compile(
    r"^(أي|اى|في بعض|مضمون|و السبب|والسبب|أي خروجه|و في بعض)",
)
_EDITOR_ASIDE = re.compile(
    r"(يحتمل|تمثيلية|رحمه الل|رضوان الل|الظاهر أنّ?ه|في بعض النسخ|"
    r"يعني الرسوخ|المداقة|الشأن بالهمزة|قال الفيض|اتّحاد الرجلين|"
    r"ابن بندار|\( ?آت\))"
)


def _harakat_count(text: str) -> int:
    return len(_HARKAT.findall(text or ""))


def _looks_like_matn_resume(stripped: str) -> bool:
    if not stripped:
        return False
    if HADITH_START_RE.match(stripped):
        return True
    if _FOOTNOTE_LINE.match(stripped) or _NOTE_CONTINUATION.match(stripped):
        return False
    if _EDITOR_ASIDE.search(stripped):
        return False
    # One shadda on اللّه is not enough — folklib asides mention Allah constantly.
    if re.match(r"^[\u0600-\u06FF]", stripped) and _harakat_count(stripped) >= 4:
        return True
    return False


def _drop_editor_aside_lines(text: str) -> str:
    kept: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped and _EDITOR_ASIDE.search(stripped) and _harakat_count(stripped) < 4:
            continue
        kept.append(line)
    return "\n".join(kept)


def _quote_delta(text: str) -> int:
    return text.count("«") - text.count("»")


def iter_line_roles(text: str) -> Iterator[tuple[str, str, str | None]]:
    """Yield (line, role, note_number) for every line, role in {"matn", "note"}.

    Single source of truth for where an editor footnote starts and stops. Both
    the stripper and the Qur'an-reference reader consume this, so they can never
    disagree about which lines are matn -- a disagreement is exactly how footnote
    text reached the citation scanner and produced a verse attributed to a
    hadith that never quoted it.

    Guillemet balance is tracked across a note because the harakat heuristic
    cannot survive a footnote that quotes vocalized Qur'an. Page 12 of al-Kafi 1
    has one: note [3] quotes «مِنْ شَرِّ الْوَسْواسِ الْخَنَّاسِ…», whose 27
    harakat sail past the threshold, so the note was declared finished mid-quote
    and both the rest of the verse and the editor's following sentence were
    appended to hadith 11's matn.

    Blank lines never end a note: this corpus double-spaces every line, so a
    note's own body is always separated from its opening by one.
    """
    in_note = False
    note_number: str | None = None
    quote_depth = 0
    for line in (text or "").splitlines():
        stripped = line.strip()
        opener = _FOOTNOTE_LINE.match(line) or _FOOTNOTE_LINE.match(stripped)
        if opener:
            in_note = True
            note_number = opener.group(1)
            quote_depth = max(0, _quote_delta(stripped))
            yield line, "note", note_number
            continue
        if in_note:
            if not stripped:
                yield line, "note", note_number
                continue
            if HADITH_START_RE.match(stripped):
                in_note = False
                note_number = None
                quote_depth = 0
                yield line, "matn", None
                continue
            if quote_depth <= 0 and _looks_like_matn_resume(stripped):
                in_note = False
                note_number = None
                yield line, "matn", None
                continue
            quote_depth = max(0, quote_depth + _quote_delta(stripped))
            yield line, "note", note_number
            continue
        yield line, "matn", None


def strip_folklib_footnotes(text: str) -> str:
    """Drop editor footnotes ([1] أي …) and leftover inline [n] markers from matn."""
    out = [line for line, role, _ in iter_line_roles(text) if role == "matn"]
    joined = _drop_editor_aside_lines("\n".join(out))
    joined = _INLINE_FOOTNOTE_REF.sub("", joined)
    joined = re.sub(r"\s*\( ?آت\)", "", joined)
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
    # Local imports: both modules read HADITH_START_RE / iter_line_roles from
    # here, so importing them at module scope would be circular.
    from src.extractors.classification import attach_sections
    from src.extractors.quran_refs import page_quran_refs

    # Sections are resolved over the unfiltered volume, in reading order: a
    # heading can sit on a page this function is about to drop, and the bab it
    # opens still governs every page after it.
    units = attach_sections(units)

    refined: list[ParsedUnit] = []
    started = False
    for unit in units:
        raw = unit.text.strip()
        text = strip_folklib_footnotes(raw)
        if not text or _is_footnote_page(text):
            continue
        # Read citations off the raw page: this is the only place that still has
        # both the footnote bodies and the inline [n] markers linking them to a
        # hadith. `text` has had both deleted.
        refs = page_quran_refs(raw)
        pieces = split_hadith_page(text)
        if pieces:
            started = True
            refined.append(
                ParsedUnit(
                    locator=unit.locator,
                    text=text,
                    source_path=unit.source_path,
                    quran_refs=refs,
                    kitab=unit.kitab,
                    bab=unit.bab,
                )
            )
            continue
        if started and len(text) >= _MIN_HADITH_CHARS:
            refined.append(
                ParsedUnit(
                    locator=unit.locator,
                    text=text,
                    source_path=unit.source_path,
                    quran_refs=refs,
                    kitab=unit.kitab,
                    bab=unit.bab,
                )
            )
    return refined


def tafsir_ayah_units(units: list[ParsedUnit]) -> list[ParsedUnit]:
    ayah = [u for u in units if is_ayah_locator(u.locator)]
    return ayah if ayah else units


# Al-Mizan section heads sit at line start: بيان, بحث روايتى, بحث فلسفى, …
# Folklib also opens the riwayat block as "رواياتى درباره …" with no
# "بحث روايتى" line. "حديثى از امام رضا" is a subheading inside that
# block and must not start a new unit. "بحث پيرامون كلمه" is prose.
_TAFSIR_HEADING = re.compile(
    r"^(?:بيان\b|"
    r"(?:يك\s+)?بحث\s+"
    r"(?:روايت[ىی]|فلسف[ىی]|تاريخ[ىی]|علم[ىی]|اخلاق[ىی]|اجتماع[ىی])"
    r"|روايات[ىی]\b"
    r"|روايت[ىی]\s+(?:از|درباره|در\s+باره|در\s+ذيل)"
    r")",
    re.MULTILINE,
)

_DUMP_TITLE = re.compile(r"^آيات?\s+\d+")
_DUMP_NUMBERED = re.compile(r"^\d{1,3}\s*[-–—]")
_DUMP_TRAILING_N = re.compile(r"\(\s*\d{1,3}\s*\)\s*$")
_PERSIAN_LETTERS = re.compile(r"[پچژگ]")


def _strong_mizan_dump_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    return bool(
        _DUMP_TITLE.match(s) or _DUMP_NUMBERED.match(s) or _DUMP_TRAILING_N.search(s)
    )


def _weak_mizan_dump_line(line: str) -> bool:
    """Arabic mushaf lines after a dump has already started (no ayah number)."""
    s = (line or "").strip()
    if not s or _PERSIAN_LETTERS.search(s):
        return False
    squashed = re.sub(r"\s+", "", s)
    return 2 <= len(squashed) <= 80


def strip_mizan_mushaf_dump(text: str) -> str:
    """Drop the opening reprint of the banner's ayahs, keep Tabatabai's prose.

    Folklib reprints Arabic + Persian of the range, then commentary. Some
    banners never use a بيان line until later (or open with سبب ابتداء).
    Dropping everything before the first heading would delete real tafsir.
    """
    source = text or ""
    lines = source.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return source
    first = lines[i].strip()
    if not (
        _strong_mizan_dump_line(first)
        or first.startswith("بسم")
        or first.startswith("آيات")
    ):
        return source
    hits = 0
    last_dump = i - 1
    for j in range(i, len(lines)):
        s = lines[j].strip()
        if not s:
            continue
        if _strong_mizan_dump_line(s) or (hits and _weak_mizan_dump_line(s)) or (
            hits == 0 and s.startswith("بسم")
        ):
            hits += 1
            last_dump = j
            continue
        if hits >= 3 and _looks_interpretive(s):
            break
        if hits:
            last_dump = j
    if hits < 3:
        return source
    rest = "\n".join(lines[last_dump + 1 :]).strip()
    return rest if rest else source


def _looks_interpretive(line: str) -> bool:
    s = (line or "").strip()
    if len(s) < 40:
        return False
    if _DUMP_TRAILING_N.search(s) or _DUMP_NUMBERED.match(s) or _DUMP_TITLE.match(s):
        return False
    return bool(_PERSIAN_LETTERS.search(s)) or "مى" in s or "است" in s


def _section_label(heading: str, seen: dict[str, int]) -> str:
    line = (heading or "").strip().split("\n", 1)[0]
    line = re.sub(r"^يك\s+", "", line)
    if line.startswith("بيان"):
        kind = "بيان"
    elif re.match(r"بحث\s+روايت[ىی]", line) or line.startswith("روايات") or re.match(
        r"روايت[ىی]\s+(?:از|درباره|در\s+باره|در\s+ذيل)", line
    ):
        kind = "بحث روايتى"
    elif re.match(r"بحث\s+فلسف[ىی]", line):
        kind = "بحث فلسفى"
    elif re.match(r"بحث\s+تاريخ[ىی]", line):
        kind = "بحث تاريخى"
    elif re.match(r"بحث\s+علم[ىی]", line):
        kind = "بحث علمى"
    elif re.match(r"بحث\s+اخلاق[ىی]", line):
        kind = "بحث اخلاقى"
    elif re.match(r"بحث\s+اجتماع[ىی]", line):
        kind = "بحث اجتماعى"
    else:
        kind = line[:40] or "قسمت"
    seen[kind] = seen.get(kind, 0) + 1
    return kind if seen[kind] == 1 else f"{kind} {seen[kind]}"


def mizan_commentary_body(text: str) -> str:
    """Text used for verse matching: mushaf dump gone, start at first heading."""
    stripped = strip_mizan_mushaf_dump(text or "")
    match = _TAFSIR_HEADING.search(stripped)
    if match:
        return stripped[match.start() :]
    return stripped


def split_mizan_sections(unit: ParsedUnit) -> list[ParsedUnit]:
    """One Gemini unit per بيان / بحث / رواياتى block inside an ayah-range banner.

    The opening mushaf reprint is dropped. Prose before the first heading
    (سبب ابتداء, unlabeled tafsir) stays on the first section.
    """
    text = strip_mizan_mushaf_dump(unit.text or "")
    matches = list(_TAFSIR_HEADING.finditer(text))
    if not matches:
        if text.strip() and text.strip() != (unit.text or "").strip():
            return [
                ParsedUnit(
                    locator=f"{unit.locator} | بيان",
                    text=text,
                    source_path=unit.source_path,
                )
            ]
        return [unit]
    seen: dict[str, int] = {}
    parts: list[ParsedUnit] = []
    preamble = text[: matches[0].start()].strip()
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.start() : end].strip()
        if i == 0 and preamble:
            body = f"{preamble}\n\n{body}"
        if not body:
            continue
        label = _section_label(match.group(0), seen)
        locator = f"{unit.locator} | {label}"
        parts.append(
            ParsedUnit(
                locator=locator,
                text=body,
                source_path=unit.source_path,
            )
        )
    return parts or [unit]


def tafsir_section_units(units: list[ParsedUnit]) -> list[ParsedUnit]:
    out: list[ParsedUnit] = []
    for unit in tafsir_ayah_units(units):
        out.extend(split_mizan_sections(unit))
    return out


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
