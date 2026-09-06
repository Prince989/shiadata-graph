"""Layer 4: ask a model about the labels nothing else could resolve.

Layers 1-3 are free and deterministic -- headings, decomposition, frequency.
What survives them is a genuine residue: a label that names something, matches no
catalog term, shares no root with one, and never recurred. `الحجة الباطنة`,
`طاعة الشيطان`, `فضول الكلام` are the shape of it -- real topics that no chapter
happened to be titled after.

Three properties keep this from becoming the fragile thing we were avoiding:

  * asked once per distinct LABEL, never per hadith. Whether `فضول الكلام` names
    a topic is a fact about the string; a thousand narrations using it ask once.
  * asked offline, between runs, never on the extraction path. A failure here
    costs a term, not a pipeline.
  * cached permanently in `config/adjudicated.json`, and the accepted answers are
    written into the derived catalog, so the next run resolves them for free and
    the residue shrinks toward nothing.

The model is an adjudicator, not an author. For each orphan it may only say:
this is the same idea as an existing term (alias), this is a real topic of its
own (new), or this names nothing indexable (drop). It never invents a hierarchy
and never sees a hadith.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from config.paths import ADJUDICATED_JSON, DERIVED_ONTOLOGY_YAML, OUTPUT_DIR
from src.models import Adjudication, AdjudicationBatch
from src.pipelines.ontology import lookup_concept, normalize_ar

logger = logging.getLogger(__name__)

BATCH_SIZE = 40

ADJUDICATOR_PROMPT = """\
You are curating a controlled vocabulary of Shi'i hadith topics.

For each Arabic label below, decide what it is. You may ONLY choose:

  alias  - it means the same as an existing topic. Give that topic in `target`,
           copied exactly from the KNOWN TOPICS list. Use this when the label is
           a wording variant, a synonym, or a narrower phrasing of a listed
           topic. الشيطنة is an alias of النكراء; حساب العباد of الحساب.
  new    - it is a real topic in its own right that the list is missing. Give
           the citation form in `target`: the shortest standard Arabic term a
           scholar would index it under, normally with the definite article.
           طاعة الشيطان -> الطاعة. فضول الكلام -> الكلام. الحجة الباطنة ->
           الحجة الباطنة, which is a technical term and stays whole.
  drop   - it names nothing a reader would ever look up: a fragment, a sentence,
           a proper name mistaken for a topic, or grammar. Leave `target` empty.

Judge the LABEL only. You are not shown a hadith and must not imagine one.
Prefer `alias` over `new` whenever a listed topic genuinely covers it -- a
vocabulary that grows without merging is worth nothing.

KNOWN TOPICS (a relevant sample, not the whole list):
{known}

LABELS TO JUDGE:
{labels}
"""


@dataclass
class Orphan:
    label: str
    count: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass
class Plan:
    aliases: dict[str, str] = field(default_factory=dict)   # label -> existing pref
    new_terms: dict[str, str] = field(default_factory=dict)  # label -> citation form
    dropped: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "alias": len(self.aliases),
            "new": len(self.new_terms),
            "drop": len(self.dropped),
        }


def load_cache(path: Path | None = None) -> dict[str, dict]:
    path = path or ADJUDICATED_JSON
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("unreadable adjudication cache %s: %s", path, exc)
        return {}


def save_cache(cache: dict[str, dict], path: Path | None = None) -> None:
    path = path or ADJUDICATED_JSON
    path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def find_orphans(root: Path | None = None) -> dict[str, Orphan]:
    """Mentions across the corpus that resolved to nothing the catalog knows.

    Read from phase-1 payloads rather than from the node table, because the ones
    that matter are exactly those that produced no node at all.
    """
    from src.pipelines.resolver import decompose, is_tautology
    from src.pipelines.resolve_pass import _mentions_of, NODES_FILENAME

    root = root or (OUTPUT_DIR / "phase1")
    orphans: dict[str, Orphan] = {}
    for path in sorted(root.rglob("*.json")):
        if path.name == NODES_FILENAME:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        for item in _mentions_of(payload):
            text = str(item.get("text") or "").strip()
            node_type = str(item.get("type") or "concept")
            if not text or node_type != "concept" or is_tautology(text):
                continue
            if lookup_concept(text) or decompose(text):
                continue
            folded = normalize_ar(text)
            entry = orphans.setdefault(folded, Orphan(label=text))
            entry.count += 1
            locator = str(payload.get("locator") or path.stem)
            if len(entry.examples) < 3 and locator not in entry.examples:
                entry.examples.append(locator)
    return orphans


def _known_sample(orphan_labels: list[str], limit: int = 220) -> list[str]:
    """Catalog terms most likely to be the right merge target.

    The whole catalog is thousands of terms and does not belong in a prompt, so
    the sample is biased toward the roots the orphans actually use.
    """
    from src.pipelines.morphology import root_signature
    from src.pipelines.ontology import load_concept_catalog

    wanted = {r for label in orphan_labels for r in root_signature(label) if r}
    scored: list[tuple[int, str]] = []
    for concept in load_concept_catalog():
        overlap = len(wanted & {r for r in root_signature(concept.pref) if r})
        scored.append((-overlap, concept.pref))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [pref for _, pref in scored[:limit]]


def adjudicate(
    agent,
    orphans: dict[str, Orphan],
    cache: dict[str, dict] | None = None,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
) -> dict[str, dict]:
    """Ask about every orphan not already in the cache. Returns the full cache."""
    cache = dict(cache or {})
    pending = [o for key, o in sorted(orphans.items()) if key not in cache]
    if limit is not None:
        pending = pending[:limit]
    logger.info("adjudicating %d orphans (%d cached)", len(pending), len(cache))

    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        labels = [o.label for o in chunk]
        prompt = ADJUDICATOR_PROMPT.format(
            known="\n".join(f"  {t}" for t in _known_sample(labels)),
            labels="\n".join(f"  {label}" for label in labels),
        )
        try:
            result = agent.complete_structured(
                prompt, AdjudicationBatch, system="Return one verdict per label."
            )
        except Exception as exc:  # noqa: BLE001 - one bad batch must not stop the rest
            logger.error("adjudication batch failed: %s", exc)
            continue
        by_label = {normalize_ar(v.label): v for v in result.verdicts}
        for orphan in chunk:
            verdict = by_label.get(normalize_ar(orphan.label))
            if verdict is None:
                continue
            cache[normalize_ar(orphan.label)] = {
                "label": orphan.label,
                "verdict": verdict.verdict,
                "target": verdict.target,
                "count": orphan.count,
            }
    return cache


def build_plan(cache: dict[str, dict]) -> Plan:
    """Sort cached verdicts into what can actually be written."""
    plan = Plan()
    for entry in cache.values():
        label = str(entry.get("label") or "").strip()
        target = str(entry.get("target") or "").strip()
        verdict = entry.get("verdict")
        if not label:
            continue
        if verdict == "alias" and target:
            hit = lookup_concept(target)
            # An alias must point at something real. A model naming a target the
            # catalog does not have is proposing a new term, not an alias.
            if hit:
                plan.aliases[label] = hit.pref
            else:
                plan.new_terms[label] = target
        elif verdict == "new" and target:
            plan.new_terms[label] = target
        else:
            plan.dropped.append(label)
    return plan


_SECTION = "  # --- adjudicated orphans (python main.py adjudicate --apply) ---"


def _yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def apply_plan(plan: Plan, path: Path | None = None) -> int:
    """Append accepted verdicts to the derived catalog.

    Appended, not merged into the harvested rows, so a re-harvest overwrites the
    heading-derived section without losing adjudicated terms -- and so a human
    can see at a glance which terms came from a model rather than from a book.
    """
    path = path or DERIVED_ONTOLOGY_YAML
    existing = path.read_text(encoding="utf-8").rstrip("\n") if path.exists() else "concepts:"
    if _SECTION in existing:
        existing = existing.split(_SECTION)[0].rstrip("\n")

    rows: list[str] = [_SECTION]
    by_target: dict[str, list[str]] = {}
    for label, target in plan.aliases.items():
        by_target.setdefault(target, []).append(label)
    for target, labels in sorted(by_target.items()):
        rendered = ", ".join(_yaml_quote(a) for a in sorted(labels))
        rows.append(
            f"  - {{id: {_yaml_quote(target)}, pref: {_yaml_quote(target)}, "
            f"aliases: [{rendered}]}}"
        )
    for label, citation in sorted(plan.new_terms.items()):
        aliases = "" if normalize_ar(label) == normalize_ar(citation) else (
            f", aliases: [{_yaml_quote(label)}]"
        )
        rows.append(
            f"  - {{id: {_yaml_quote(citation)}, pref: {_yaml_quote(citation)}{aliases}}}"
        )

    path.write_text(existing + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    from src.pipelines.ontology import clear_catalog_caches

    clear_catalog_caches()
    return len(rows) - 1


def report(orphans: dict[str, Orphan], plan: Plan | None = None, limit: int = 25) -> str:
    ranked = sorted(orphans.values(), key=lambda o: (-o.count, o.label))
    lines = [f"orphan labels: {len(orphans)}", ""]
    for orphan in ranked[:limit]:
        lines.append(f"   x{orphan.count:<4} {orphan.label}")
    if len(ranked) > limit:
        lines.append(f"   ... {len(ranked) - limit} more")
    if plan:
        lines += ["", f"plan: {plan.summary()}"]
        for label, target in list(plan.aliases.items())[:10]:
            lines.append(f"   alias  {label}  ->  {target}")
        for label, target in list(plan.new_terms.items())[:10]:
            lines.append(f"   new    {label}  ->  {target}")
    return "\n".join(lines)


def counts_by_verdict(cache: dict[str, dict]) -> Counter:
    return Counter(str(entry.get("verdict")) for entry in cache.values())
