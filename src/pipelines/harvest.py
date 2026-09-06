"""Build a concept catalog out of the classification the books already carry.

The vocabulary problem was never that these concepts are hard to name -- it is
that there are thousands of them and nobody can hand-write the list. But nobody
has to: Kulayni, al-Hurr al-Amili and al-Saduq each chaptered their collections,
and those chapter titles ARE the vocabulary, at exactly the granularity a graph
node wants. Roughly 16,000 headings sit in `data/raw_epubs/hadith` -- 11,805 in
Wasa'il alone -- authored by the scholars, already parsed, and free.

Two passes over them:

  Layer 1  a heading that names a subject becomes a concept, with the kitab it
           sits under as its `broader`.
  Layer 2  a long title decomposes into the concepts its words name, against a
           catalog that layer 1 just grew -- so it is run to a fixpoint, each
           pass resolving titles the previous pass made resolvable.

Output goes to `config/derived_ontology.yaml`, kept strictly separate from the
hand-written `base_ontology.yaml`. Re-harvesting must never touch a human's
judgement call, and `قتل النفس -> الانتحار` is a judgement call no heading
implies.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from config.paths import DERIVED_ONTOLOGY_YAML, RAW_EPUBS_DIR
from src.extractors.classification import heading_topic, is_empty_topic, page_headings
from src.extractors.txt_parser import parse_txt
from src.pipelines.ontology import normalize_ar

logger = logging.getLogger(__name__)

# A heading that opens with one of these states a CLAIM, not a subject:
# `باب أن الأرض لا تخلو من حجة` is a proposition about the earth, and as a node
# it would be true of exactly one chapter. 358 of al-Kafi's 2,002 are like this.
_PROPOSITIONAL = re.compile(
    r"^(?:أن|أنّ|ان|إن|أنه|أنها|انه|انها|ما|من|في|فيمن|فيما|إذا|اذا|كيف|هل|لا"
    r"|كم|متى|حكم من|قول)\b"
)

# Rijal volumes index narrators alphabetically, and each letter gets a heading:
# `الثاء ثابت`, `الجيم جابر`, `الخاء خالد`. The shape is exact -- the article,
# a letter NAME, then one word -- so it is recognisable without listing narrators.
_LETTER_INDEX = re.compile(
    r"^ال(?:ألف|الف|باء|تاء|ثاء|جيم|حاء|خاء|دال|ذال|راء|زاي|سين|شين|صاد|ضاد"
    r"|طاء|ظاء|عين|غين|فاء|قاف|كاف|لام|ميم|نون|هاء|واو|ياء)\s+\S+$"
)

# Persian orthography. These four letters do not exist in classical Arabic, so
# any heading carrying one came from a translated footnote, not from the matn:
# `چهارم در ناخن` is a Persian gloss that leaked into the heading stream.
_PERSIAN = re.compile(r"[پچژگ]")

# Bare masdars and verbal nouns. Each names an action with no object, so as a
# standalone node it groups everything and distinguishes nothing -- `ترك` is
# "abandoning", true of a thousand chapters. They are rejected only as the WHOLE
# term; `ترك الصلاة` is a real topic and keeps its head.
_GENERIC_SINGLE = frozenset(
    normalize_ar(w)
    for w in (
        "ترك", "إتيان", "اتيان", "كون", "كيفية", "إخراج", "اخراج", "عمل",
        "إعطاء", "اعطاء", "إعادة", "اعادة", "حضور", "طلب", "دخول", "خروج",
        "طول", "طرح", "مقدار", "وضع", "رفع", "أخذ", "اخذ", "جعل", "نقل",
        "بيع", "شراء", "قطع", "منع", "دفع", "رد", "صرف", "قدر",
    )
)

# `الرجل يقتل` is a sentence: subject then imperfect verb. A ي-initial word in
# any position but the first is nearly always that verb -- these few nouns are
# the exceptions, and the class is closed because Arabic nouns rarely begin ي.
_YA_NOUNS = frozenset(
    normalize_ar(w) for w in ("يوم", "يقين", "يد", "يمين", "يتيم", "يهود", "يونس", "يس")
)
_IMPERFECT = re.compile(r"^ي[^\s]{3,}$")

# `وجوبه` / `وجوبها` are back-references -- "its obligation" -- pointing at the
# previous chapter rather than naming a topic. A real term almost always carries
# the article, and a pronoun-suffixed one never does, which separates وجوبها
# from الفقه without listing either.
_PRONOUN_SUFFIX = re.compile(r"(?:ه|ها|هم|هن|هما)$")

# Descriptive heads. `باب صفة العلماء` is about العلماء, not about صفة.
_META_HEADS = (
    "صفة", "صفات", "فضل", "فضائل", "ثواب", "عقاب", "أحكام", "حكم", "آداب",
    "أبواب", "جملة", "ذكر", "بيان", "معنى", "تفسير", "نوادر", "وجوب",
    "استحباب", "كراهة", "تحريم", "جواز", "عدم", "اشتراط", "بطلان",
)

# A concept term is a short noun phrase. Longer headings are not discarded --
# they go to layer 2, which mines the concepts out of them.
_MAX_TERM_WORDS = 3
_MIN_TERM_CHARS = 3

# Stored labels are de-vocalised. Books disagree about harakat -- al-Kafi prints
# الطَّهَارَةِ where Wasa'il prints الطهارة -- and while `normalize_ar` matches
# them anyway, storing both spellings makes the catalog look like it holds two
# concepts when it holds one.
_HARAKAT = re.compile(r"[ً-ٟۖ-ۭـ]")


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", _HARAKAT.sub("", text or "")).strip()


@dataclass
class Harvest:
    terms: dict[str, str] = field(default_factory=dict)      # term -> broader
    sources: Counter = field(default_factory=Counter)        # term -> heading count
    skipped_propositional: int = 0
    long_titles: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "terms": len(self.terms),
            "long_titles": len(self.long_titles),
            "propositional_skipped": self.skipped_propositional,
        }


def _strip_meta_head(topic: str) -> str:
    """Drop a leading descriptive head: `صفة العلماء` is about العلماء."""
    words = topic.split()
    while words and normalize_ar(words[0]) in {normalize_ar(h) for h in _META_HEADS}:
        words = words[1:]
    return " ".join(words).strip()


# Editorial furniture and bare grammar. A closed class -- pronouns, particles
# and the printer's brackets -- so it does not grow with the corpus the way a
# list of banned topics would.
_NOT_A_TOPIC = re.compile(r"[()\[\]{}«»<>0-9٠-٩|]")
# Structural words naming the BOOK's furniture rather than a subject. They can
# appear anywhere in a title -- "الطلاق عنوان الباب", "الطلاق أبواب الايلاء" --
# so unlike the function words these are checked at every position.
_EDITORIAL = frozenset(
    normalize_ar(w) for w in ("عنوان", "الباب", "باب", "أبواب", "الأبواب", "كتاب", "الكتاب")
)
_FUNCTION_WORDS = frozenset(
    normalize_ar(w)
    for w in (
        "أحدهما", "أحدهم", "هو", "هي", "هم", "هن", "هذا", "هذه", "ذلك", "تلك",
        "الذي", "التي", "الذين", "أخذ", "قال", "قوله", "غير", "سائر", "بعض",
        "كل", "أول", "آخر", "شيء", "أشياء", "أمر", "أمور", "وجه", "وجوه",
        "باب", "أبواب", "كتاب", "كتب", "جملة", "عدة", "نحو", "مثل", "معه",
    )
)


def _is_term(topic: str) -> bool:
    if not topic or len(topic) < _MIN_TERM_CHARS:
        return False
    if len(topic.split()) > _MAX_TERM_WORDS:
        return False
    if _PROPOSITIONAL.match(topic) or _NOT_A_TOPIC.search(topic):
        return False
    if not re.search(r"[؀-ۿ]", topic):
        return False
    if _LETTER_INDEX.match(topic) or _PERSIAN.search(topic):
        return False
    words = topic.split()
    if normalize_ar(words[0]) in _FUNCTION_WORDS:
        return False
    if any(normalize_ar(w) in _EDITORIAL for w in words):
        return False
    # A ي-initial word after the first is an imperfect verb, which makes the
    # heading a clause: `الرجل يقتل`, `الرجل يتعدى`.
    for word in words[1:]:
        folded = normalize_ar(word)
        if _IMPERFECT.match(folded) and folded not in _YA_NOUNS:
            return False
    if len(words) == 1:
        folded = normalize_ar(topic)
        # Measured on the SURFACE form, not the folded one. `normalize_ar` strips
        # the article, so الحج folds to حج and a two-character floor rejected it
        # -- along with الحق, الدم and الأم. That silently killed `كتاب الحج`,
        # and every Hajj chapter after it inherited الصيام as its parent.
        if len(topic) < _MIN_TERM_CHARS:
            return False
        if folded in _GENERIC_SINGLE:
            return False
        if not topic.startswith("ال") and _PRONOUN_SUFFIX.search(topic):
            return False
    return True


# Not every kitab boundary is printed. Faqih vol 2 runs كتاب الصوم straight into
# the Hajj chapters with no heading between them -- the words `كتاب الحج` appear
# nowhere in its body -- so a running kitab variable handed الصوم to 165 chapters
# about إحرام and طواف and said it as fact.
#
# The book cannot be asked where the seam is, but the narrations answer: a kitab
# is discussed throughout its own span. Measured page by page, whether the kitab
# label occurs at all, a sound span stays lit end to end (al-Kafi 4's الحج, 66%;
# its الصيام, 54%) while a span holding two books lights up and then goes dark.
# So look for the one split where a well-attested prefix meets a silent tail.
#
# The tail is orphaned, not re-parented. We can tell these chapters are not
# الصوم; we cannot tell what they are, and an honest gap beats a confident lie.
#
# Every threshold is a floor on evidence, not a tuned constant: enough span to
# split, enough prefix to establish the subject was ever discussed, and a tail
# long enough that its silence means something. Across the corpus's 86 spans
# this fires once -- on Faqih vol 2, cutting between بَاب الاعتكاف (the last
# fasting chapter) and بَاب علل الحج (the first Hajj one).
_SPLIT_MIN_SPAN = 50
_SPLIT_MIN_PREFIX = 15
_SPLIT_MIN_PREFIX_HITS = 8
_SPLIT_MIN_PREFIX_RATE = 0.35
_SPLIT_MAX_SUFFIX_RATE = 0.06
_SPLIT_MIN_SUFFIX = 25
_SPLIT_MIN_SUFFIX_FRACTION = 0.40


def _abandoned_from(flags: list[int]) -> int | None:
    """Index after which the kitab stops being discussed, or None if it never does."""
    total = len(flags)
    if total < _SPLIT_MIN_SPAN:
        return None
    seen = sum(flags)
    best: tuple[float, int] | None = None
    hits = 0
    for cut in range(1, total):
        hits += flags[cut - 1]
        if cut < _SPLIT_MIN_PREFIX or hits < _SPLIT_MIN_PREFIX_HITS:
            continue
        tail = total - cut
        if tail < _SPLIT_MIN_SUFFIX or tail < _SPLIT_MIN_SUFFIX_FRACTION * total:
            continue
        before, after = hits / cut, (seen - hits) / tail
        if before < _SPLIT_MIN_PREFIX_RATE or after > _SPLIT_MAX_SUFFIX_RATE:
            continue
        if best is None or before - after > best[0]:
            best = (before - after, cut)
    return None if best is None else best[1]


def _flush_span(out: Harvest, kitab: str, span: list[tuple[bool, str, int]]) -> None:
    """Record one kitab's chapters, dropping the parent past an unmarked seam."""
    cut = _abandoned_from([flag for _, _, flag in span]) if kitab else None
    if cut is not None:
        logger.info(
            "kitab %r stops being discussed after %d of %d chapters; "
            "orphaning the rest rather than guessing their book",
            kitab,
            cut,
            len(span),
        )
    for position, (is_term, label, _) in enumerate(span):
        parent = "" if cut is not None and position >= cut else kitab
        if is_term:
            out.terms.setdefault(label, parent if parent != label else "")
            out.sources[label] += 1
        else:
            out.long_titles.append((label, parent))


def reset_derived(path: Path | None = None) -> None:
    """Empty the derived catalog before a harvest.

    Without this the harvest reads the file it is about to replace: layer 2
    decomposes against the previous run's output, so any junk term it produced
    is found again and re-emitted. `أخذ` survived three harvests that way, each
    one citing the last as evidence.
    """
    path = path or DERIVED_ONTOLOGY_YAML
    path.write_text("concepts: []\n", encoding="utf-8")
    _reload_catalog()


def scan(root: Path | None = None, pattern: str = "*.txt") -> Harvest:
    """Read every classified book and sort its headings into terms and titles."""
    root = root or (RAW_EPUBS_DIR / "hadith")
    out = Harvest()
    for path in sorted(root.glob(pattern)):
        kitab = ""
        # A chapter's parent cannot be settled until its kitab's whole span has
        # been read, because the span is what shows whether the kitab was ever
        # abandoned. Chapters are held here and written out at the next kitab.
        span: list[tuple[bool, str, int]] = []
        try:
            units = parse_txt(path)
        except (OSError, ValueError) as exc:
            logger.warning("skip unreadable book %s: %s", path, exc)
            continue
        for unit in units:
            headings = page_headings(unit.text)
            if not headings:
                continue
            page: str | None = None
            for level, title in headings:
                topic = plain(heading_topic(title))
                if not topic or is_empty_topic(topic):
                    continue
                if _PROPOSITIONAL.match(topic):
                    out.skipped_propositional += 1
                    continue
                if level == "kitab":
                    _flush_span(out, kitab, span)
                    span = []
                    candidate = _strip_meta_head(topic) or topic
                    # Only a real term may become a parent. Prose beginning with
                    # the word kitab -- a citation line like
                    # "كتاب ( تهذيب الأحكام ) أن النبيذ المسكر..." -- otherwise
                    # became the `broader` of every bab after it.
                    #
                    # And a REJECTED kitab clears the current one rather than
                    # leaving it standing. State that outlives the section it
                    # describes is worse than no state: when `كتاب الحج` was
                    # rejected the scan kept الصيام and handed it to hundreds of
                    # Hajj chapters, which reads as fact rather than as a gap.
                    kitab = candidate if _is_term(candidate) else ""
                    if kitab:
                        out.terms.setdefault(kitab, "")
                        out.sources[kitab] += 1
                    else:
                        logger.info("kitab heading not usable as a parent: %r", topic)
                    continue
                stripped = _strip_meta_head(topic)
                if not stripped:
                    continue
                # A book's own kitab is the natural parent, and it is almost
                # always already a term because kitab titles are short. Whether
                # this page still talks about that kitab is recorded alongside,
                # for `_flush_span` to judge once the span is complete.
                if page is None:
                    page = normalize_ar(unit.text)
                head = normalize_ar(kitab).split()[0] if kitab else ""
                attested = 1 if head and head in page else 0
                span.append((_is_term(stripped), stripped, attested))
        _flush_span(out, kitab, span)
    return out


def mine_long_titles(harvest: Harvest, rounds: int = 3) -> int:
    """Layer 2: pull concepts out of long titles, to a fixpoint.

    `وجوب الإخلاص في العبادة والنية` names الإخلاص, العبادة and النية. Each round
    decomposes against a catalog the previous round grew, so a title that could
    not be mined on the first pass often can be on the second.
    """
    from src.pipelines.resolver import decompose

    added = 0
    for _ in range(rounds):
        before = added
        for title, kitab in harvest.long_titles:
            for _key, pref in decompose(title):
                if pref not in harvest.terms:
                    harvest.terms[pref] = kitab if kitab != pref else ""
                    added += 1
                harvest.sources[pref] += 1
            # Any short residue the meta-head strip left behind is itself a term.
            residue = _strip_meta_head(title)
            if _is_term(residue) and residue not in harvest.terms:
                harvest.terms[residue] = kitab if kitab != residue else ""
                added += 1
        if added == before:
            break
        _reload_catalog()
    return added


def promote_recurring(harvest: Harvest, min_df: int = 2, root: Path | None = None) -> int:
    """Layer 3: terms the corpus keeps reaching for that no chapter is named after.

    Headings only tell you what a book chose to chapter on. `النكراء` is a real
    topic that no bab is titled after, and it would stay invisible to layers 1
    and 2 forever. A label several distinct narrations independently used has
    demonstrated the same thing a chapter title demonstrates -- that it groups
    something -- so it earns a place on the same evidence.

    Reads the resolved node table, so it only runs after `resolve-nodes`.
    """
    from config.paths import OUTPUT_DIR
    import json

    path = (root or (OUTPUT_DIR / "phase1")) / "nodes.json"
    if not path.exists():
        logger.info("no node table at %s; skipping frequency promotion", path)
        return 0
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("unreadable node table %s: %s", path, exc)
        return 0

    added = 0
    for entry in table.values():
        if entry.get("type") != "concept" or entry.get("curated"):
            continue
        if int(entry.get("df") or 0) < min_df:
            continue
        label = plain(str(entry.get("label") or ""))
        if not _is_term(label) or label in harvest.terms:
            continue
        harvest.terms[label] = ""
        harvest.sources[label] += int(entry.get("df") or 0)
        added += 1
    logger.info("promoted %d recurring corpus terms", added)
    return added


def _reload_catalog() -> None:
    """Drop the cached catalog so the next decompose() sees new terms."""
    from src.pipelines.ontology import clear_catalog_caches

    clear_catalog_caches()


def _yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write(harvest: Harvest, path: Path | None = None, min_sources: int = 1) -> int:
    """Emit `derived_ontology.yaml`. Never touches the hand-written catalog."""
    from src.pipelines.ontology import lookup_concept

    path = path or DERIVED_ONTOLOGY_YAML
    rows: list[str] = []
    seen: set[str] = set()
    for term, broader in sorted(harvest.terms.items()):
        folded = normalize_ar(term)
        if not folded or folded in seen:
            continue
        if harvest.sources.get(term, 0) < min_sources:
            continue
        seen.add(folded)
        parent = ""
        if broader and normalize_ar(broader) != folded:
            parent = f", broader: {_yaml_quote(broader)}"
        rows.append(
            f"  - {{id: {_yaml_quote(term)}, pref: {_yaml_quote(term)}{parent}}}"
            f"  # x{harvest.sources.get(term, 0)}"
        )

    header = (
        "# GENERATED by `python main.py harvest-ontology`. Do not hand-edit.\n"
        "#\n"
        "# Concepts mined from the chapter headings the books themselves carry --\n"
        "# roughly 16,000 kitab/bab titles across al-Kafi, Wasa'il, al-Faqih,\n"
        "# al-Istibsar and the rest. The scholars wrote this vocabulary; nobody\n"
        "# needs to invent it.\n"
        "#\n"
        "# Hand-written aliases and judgement calls belong in base_ontology.yaml,\n"
        "# which this file never touches and which wins on conflict.\n"
        "concepts:\n"
    )
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    logger.info("wrote %d derived concepts to %s", len(rows), path)
    _reload_catalog()
    return len(rows)


def report(harvest: Harvest, limit: int = 20) -> str:
    lines = [str(harvest.summary()), "", "most-chaptered terms:"]
    for term, count in harvest.sources.most_common(limit):
        parent = harvest.terms.get(term) or "-"
        lines.append(f"   x{count:<5} {term:<28} broader= {parent}")
    return "\n".join(lines)
