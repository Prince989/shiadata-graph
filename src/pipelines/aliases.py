"""Enrich the catalog's aliases -- the surface forms one concept is written in.

The catalog had 1,625 concepts and zero aliases, which means every narration
that phrases a topic differently from its chapter title misses it entirely.

Unlike concepts and parentage, synonymy is NOT written down anywhere in these
books. The chapter titles gave us the vocabulary and the chapter nesting gave us
the hierarchy, but nobody wrote `الصوم = الصيام`. Three free sources were tried
and measured before reaching for a model:

  intra-catalog string similarity   ~20 real pairs, and the cheap signals pair
                                    الزاني with الزانية and ميراث الزوج with
                                    ميراث الزوجة -- opposites, not synonyms.
  the mention stream                nothing to mine until phase 1 runs at scale;
                                    `proposals.py` is the path when it does.
  the corpus's own glosses          the markers are there in quantity (1,637
                                    `أي`, 1,371 `وهو`) but isolating the two
                                    operands needs real phrase parsing; window
                                    extraction ran at roughly 5% precision.

So this asks a model -- but never trusts it. A proposal is kept only if the
corpus actually uses that wording, which is checkable and cheap, and which no
amount of fluent invention can fake. The model widens the net; the text decides.

Cost is bounded and paid once: the catalog is finite, batches are 40 concepts,
and every answer is cached by concept so re-running spends nothing.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

from config.paths import CONFIG_DIR, RAW_EPUBS_DIR
from src.extractors.classification import heading_topic, page_headings
from src.extractors.txt_parser import parse_txt
from src.models import AliasBatch
from src.pipelines.harvest import plain
from src.pipelines.ontology import load_concept_catalog, normalize_ar

logger = logging.getLogger(__name__)

BATCH_SIZE = 40
ALIASES_YAML = CONFIG_DIR / "derived_aliases.yaml"
CACHE_PATH = CONFIG_DIR / "alias_cache.json"

# An alias has to earn its place: appear in the corpus, and appear often enough
# that the shape of its usage means something. Five pages, because the check
# below compares distributions and a three-page word has no distribution --
# الوضاءة (radiance) passed every other test on three pages.
_MIN_ATTESTATION = 5

# Attestation alone cannot reject a real word used for something else. الطوفان
# is all over the corpus -- Noah's flood -- and looks like a fine alias for
# الطواف; الوضاءة means radiance, not ablution. What separates them is WHERE they
# are used: a true synonym is discussed alongside its concept, in the same
# books, while a homograph lives somewhere else entirely. So the two page
# distributions have to overlap, compared as vectors so that the PROPORTIONS
# matter and not merely whether the books coincide at all.
#
# Measured on nine pairs: الطوفان scores 0.05 against الطواف and الوضاءة 0.25
# against الوضوء, while real synonyms run 0.22 (الحجى/العقل) to 0.85
# (الطهور/الوضوء). The cut sits under the lowest true pair -- the two false ones
# are caught here and by the page floor respectively, rather than by squeezing
# this threshold until it clips a real synonym.
_MIN_CONTEXT_OVERLAP = 0.20

_ARABIC_WORD = re.compile(r"[ء-ي]+")
_CLITIC_HEAD = re.compile(r"^(?:[وف])?(?:[بلك])?(?:ال)?")

ALIAS_PROMPT = """\
You are given concepts from classical Shi'i hadith literature, each the title of
a chapter in al-Kafi, Wasa'il al-Shi'a, Man la yahduruhu al-Faqih or al-Istibsar.

For each one, list other Arabic wordings a narration might use for THE SAME
concept: a different masdar of the same root, a synonym, a common construct
form. Give the bare Arabic wording, no explanation.

Rules:
- Same concept only. الزاني and الزانية are different, so are ميراث الزوج and
  ميراث الزوجة, and so are الصيد and الذبائح.
- A narrower or broader topic is not an alias. صلاة الجمعة is not الصلاة.
- Classical Arabic as the books write it, not modern paraphrase.
- Return an empty list rather than a doubtful guess.

Concepts:
{concepts}
"""


def load_cache(path: Path | None = None) -> dict[str, list[str]]:
    path = path or CACHE_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("unreadable alias cache %s: %s", path, exc)
        return {}


def save_cache(cache: dict[str, list[str]], path: Path | None = None) -> None:
    path = path or CACHE_PATH
    path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )


def match_key(text: str) -> str:
    """Comparison key for corpus text: normalised, de-cliticised word by word.

    `normalize_ar` sheds only a leading ال, which is right for a label and wrong
    for prose -- every word in a narration carries its own article and clitics.
    """
    words = []
    for word in normalize_ar(text).split():
        bare = _CLITIC_HEAD.sub("", word, count=1)
        words.append(bare if len(bare) >= 3 else word)
    return " ".join(words)


def attest(
    candidates: set[str], root: Path | None = None, pattern: str = "*.txt"
) -> dict[str, Counter]:
    """Where each candidate wording occurs: {candidate: {kitab: pages}}."""
    root = root or (RAW_EPUBS_DIR / "hadith")
    index: dict[str, str] = {}
    for candidate in candidates:
        key = match_key(candidate)
        if key:
            index.setdefault(key, candidate)
    if not index:
        return {}
    longest = max(len(k.split()) for k in index)

    found: dict[str, Counter] = {}
    for path in sorted(root.glob(pattern)):
        try:
            units = parse_txt(path)
        except (OSError, ValueError) as exc:
            logger.warning("skip unreadable book %s: %s", path, exc)
            continue
        kitab = path.stem
        for unit in units:
            for level, title in page_headings(unit.text):
                if level == "kitab":
                    topic = plain(heading_topic(title))
                    if topic:
                        kitab = topic
            words = [match_key(w) for w in _ARABIC_WORD.findall(normalize_ar(unit.text))]
            here = set()
            for size in range(1, longest + 1):
                for start in range(len(words) - size + 1):
                    gram = " ".join(words[start : start + size])
                    if gram in index:
                        here.add(index[gram])
            for candidate in here:
                found.setdefault(candidate, Counter())[kitab] += 1
    return found


def _overlap(a: Counter, b: Counter) -> float:
    """Cosine of two page distributions over books.

    Not the share of the alias's pages that land in books the concept touches:
    a common concept touches nearly every book, so that share is near 1 for
    anything -- الطوفان scored 0.63 against الطواف by it. Cosine weighs the
    proportions and puts the same pair at 0.05.
    """
    if not a or not b:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if not na or not nb:
        return 0.0
    return sum(a.get(k, 0) * b.get(k, 0) for k in set(a) | set(b)) / (na * nb)


def propose(gemini, concepts: list[str], cache: dict[str, list[str]],
            limit: int | None = None, batch_size: int = BATCH_SIZE) -> dict[str, list[str]]:
    """Ask for candidate wordings, one batch at a time, skipping cached ones."""
    todo = [c for c in concepts if c not in cache]
    if limit is not None:
        todo = todo[:limit]
    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        prompt = ALIAS_PROMPT.format(concepts="\n".join(f"- {c}" for c in chunk))
        try:
            batch = gemini.structured(
                prompt, AliasBatch, system="Return one entry per concept."
            )
        except Exception as exc:  # one bad batch must not lose the rest
            logger.warning("alias batch %d failed: %s", start // batch_size, exc)
            continue
        answered = {entry.concept: entry.aliases for entry in batch.entries}
        for concept in chunk:
            cache[concept] = [a.strip() for a in answered.get(concept, []) if a.strip()]
        logger.info("proposed aliases for %d concepts", start + len(chunk))
    return cache


def ground(cache: dict[str, list[str]], root: Path | None = None) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Keep only proposals the corpus actually uses, and that are still free.

    Two ways a fluent-sounding alias is still wrong: the books never say it, or
    they say it about something else that is already its own concept. The first
    is what the attestation check is for; the second would silently merge two
    concepts, which is worse than missing an alias.
    """
    catalog = load_concept_catalog()
    taken = {normalize_ar(c.pref) for c in catalog}
    taken |= {normalize_ar(a) for c in catalog for a in c.aliases}

    wanted: set[str] = set()
    for concept, proposals in cache.items():
        if proposals:
            wanted.add(concept)
        for alias in proposals:
            if normalize_ar(alias) and normalize_ar(alias) != normalize_ar(concept):
                wanted.add(alias)
    seen = attest(wanted, root=root)

    kept: dict[str, list[str]] = {}
    stats: Counter = Counter()
    claimed: dict[str, str] = {}
    for concept in sorted(cache):
        for alias in cache[concept]:
            folded = normalize_ar(alias)
            if not folded or folded == normalize_ar(concept):
                stats["same_as_concept"] += 1
            elif folded in taken:
                stats["already_a_concept"] += 1
            elif sum(seen.get(alias, Counter()).values()) < _MIN_ATTESTATION:
                stats["not_in_corpus"] += 1
            elif _overlap(seen.get(concept, Counter()), seen.get(alias, Counter())) < _MIN_CONTEXT_OVERLAP:
                stats["different_context"] += 1
            elif folded in claimed:
                # Two concepts cannot share one alias; the graph would fuse them.
                stats["contested"] += 1
            else:
                claimed[folded] = concept
                kept.setdefault(concept, []).append(alias)
                stats["kept"] += 1
    return kept, dict(stats)


def _yaml_quote(value: str) -> str:
    return '"%s"' % value.replace("\\", "\\\\").replace('"', '\\"')


def write(kept: dict[str, list[str]], path: Path | None = None) -> int:
    path = path or ALIASES_YAML
    rows = []
    for concept in sorted(kept):
        forms = ", ".join(_yaml_quote(a) for a in sorted(kept[concept]))
        rows.append(f"  - {{pref: {_yaml_quote(concept)}, aliases: [{forms}]}}")
    header = (
        "# GENERATED by `python main.py enrich-aliases`. Do not hand-edit.\n"
        "#\n"
        "# Surface forms proposed by the adjudicator and then CHECKED against the\n"
        "# corpus: an alias appears here only if the books actually use that\n"
        "# wording on at least two pages, and only if no other concept already\n"
        "# claims it. Hand-written aliases belong in base_ontology.yaml.\n"
        "aliases:\n"
    )
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    logger.info("wrote aliases for %d concepts to %s", len(kept), path)
    return sum(len(v) for v in kept.values())


def report(kept: dict[str, list[str]], stats: dict[str, int], limit: int = 20) -> str:
    lines = [f"grounded aliases: {stats}"]
    for concept in sorted(kept, key=lambda c: -len(kept[c]))[:limit]:
        lines.append(f"   {concept:<26} {' · '.join(kept[concept])}")
    return "\n".join(lines)


def catalog_concepts() -> list[str]:
    """Every concept worth asking about, commonest first."""
    return [c.pref for c in load_concept_catalog() if plain(c.pref)]
