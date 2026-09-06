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
    r"^(?:أن|أنّ|ان|إن|أنه|انه|ما|من|في|إذا|اذا|كيف|هل|لا|كم|متى|حكم من|قول)\b"
)

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
    words = topic.split()
    if normalize_ar(words[0]) in _FUNCTION_WORDS:
        return False
    # A lone short word is almost always a fragment of a wrapped heading rather
    # than a subject. Multi-word phrases carry their own evidence of being one.
    if len(words) == 1 and len(normalize_ar(topic)) < 3:
        return False
    return True


def scan(root: Path | None = None, pattern: str = "*.txt") -> Harvest:
    """Read every classified book and sort its headings into terms and titles."""
    root = root or (RAW_EPUBS_DIR / "hadith")
    out = Harvest()
    for path in sorted(root.glob(pattern)):
        kitab = ""
        try:
            units = parse_txt(path)
        except (OSError, ValueError) as exc:
            logger.warning("skip unreadable book %s: %s", path, exc)
            continue
        for unit in units:
            for level, title in page_headings(unit.text):
                topic = plain(heading_topic(title))
                if not topic or is_empty_topic(topic):
                    continue
                if _PROPOSITIONAL.match(topic):
                    out.skipped_propositional += 1
                    continue
                if level == "kitab":
                    candidate = _strip_meta_head(topic) or topic
                    # Only a real term may become a parent. Prose beginning with
                    # the word kitab -- a citation line like
                    # "كتاب ( تهذيب الأحكام ) أن النبيذ المسكر..." -- otherwise
                    # became the `broader` of every bab after it.
                    if _is_term(candidate):
                        kitab = candidate
                        out.terms.setdefault(kitab, "")
                        out.sources[kitab] += 1
                    continue
                stripped = _strip_meta_head(topic)
                if _is_term(stripped):
                    # A book's own kitab is the natural parent, and it is almost
                    # always already a term because kitab titles are short.
                    out.terms.setdefault(stripped, kitab if kitab != stripped else "")
                    out.sources[stripped] += 1
                elif stripped:
                    out.long_titles.append((stripped, kitab))
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


def _reload_catalog() -> None:
    """Drop the cached catalog so the next decompose() sees new terms."""
    from src.pipelines.ontology import load_concept_catalog, load_entity_catalog

    load_concept_catalog.cache_clear()
    load_entity_catalog.cache_clear()


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
