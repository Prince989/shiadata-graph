"""Grow the vocabulary from the corpus instead of guessing at it.

Whether a term is a useful index term is not a property of the string, it is a
property of the corpus: a term is useful exactly when more than one narration
reaches for it. That is measurable, so it is measured here rather than predicted
by rules in the extractor.

The model writes anything the vocabulary cannot express into `proposed_nodes`,
which never reaches the graph. This module counts those proposals across
distinct hadiths and sorts them into three piles:

  promote  df >= threshold  -- several narrations independently wanted it, so it
                              groups something real. Goes into the catalog.
  repair   df == 1 but it names a term already in the catalog (عقل المرء) --
                              folded onto that term, no new entry.
  drop     df == 1 and anchored to nothing (اجتهاد المجتهدين) -- noise.

خلق العقل is the case this exists for: a genuine topic that parallel narrations
share, which no hand-written rule could have distinguished from عقل المرء.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from config.paths import ONTOLOGY_YAML, OUTPUT_DIR
from src.pipelines.ontology import lookup_concept, normalize_ar, repair_to_vocabulary

logger = logging.getLogger(__name__)

# Two distinct narrations is the minimum evidence that a term groups anything.
# Raise it once the corpus is large; at that point a term wanted by only two
# hadiths out of 36,000 is likelier to be a coincidence than a topic.
DEFAULT_THRESHOLD = 2


@dataclass
class Proposal:
    """One proposed term and every hadith that asked for it."""

    term: str
    sources: list[str] = field(default_factory=list)
    variants: set[str] = field(default_factory=set)

    @property
    def df(self) -> int:
        return len(set(self.sources))


def _hadith_key(payload: dict) -> str:
    return f"{payload.get('marker') or ''}|{payload.get('locator') or ''}"


def collect(payloads: list[dict]) -> dict[str, Proposal]:
    """Group proposals by normalized form, keeping the most common surface form."""
    by_key: dict[str, Proposal] = {}
    for payload in payloads:
        source = _hadith_key(payload)
        items = payload.get("proposed_nodes") or []
        for item in items:
            term = str(item or "").strip()
            folded = normalize_ar(term)
            if not folded:
                continue
            # Anything already in the catalog is not a proposal; the model just
            # routed it to the wrong field.
            if lookup_concept(term) is not None:
                continue
            entry = by_key.setdefault(folded, Proposal(term=term))
            entry.sources.append(source)
            entry.variants.add(term)
    return by_key


def scan_output_dir(root: Path | None = None) -> list[dict]:
    """Read every phase-1 hadith payload written to disk."""
    root = root or (OUTPUT_DIR / "phase1")
    payloads: list[dict] = []
    for path in sorted(root.rglob("*.json")):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            logger.warning("skip unreadable payload %s: %s", path, exc)
    return payloads


@dataclass
class Plan:
    promote: list[Proposal] = field(default_factory=list)
    repair: list[tuple[Proposal, str]] = field(default_factory=list)
    drop: list[Proposal] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"promote={len(self.promote)} repair={len(self.repair)} drop={len(self.drop)}"
        )


def build_plan(
    proposals: dict[str, Proposal],
    threshold: int = DEFAULT_THRESHOLD,
) -> Plan:
    plan = Plan()
    for entry in sorted(proposals.values(), key=lambda p: (-p.df, p.term)):
        if entry.df >= threshold:
            plan.promote.append(entry)
            continue
        anchor = repair_to_vocabulary(entry.term)
        if anchor:
            plan.repair.append((entry, anchor))
        else:
            plan.drop.append(entry)
    return plan


_SECTION_HEADER = "  # --- promoted from corpus frequency (see pipelines/proposals.py) ---"


def _yaml_escape(value: str) -> str:
    return value.replace('"', '\\"')


def apply_promotions(
    plan: Plan,
    catalog_path: Path | None = None,
    threshold: int = DEFAULT_THRESHOLD,
) -> int:
    """Append promoted terms to the concept catalog.

    Appended as text rather than re-serialized: the catalog is hand-maintained
    and its comments carry the reasoning behind several entries, which a YAML
    round-trip would erase.
    """
    if not plan.promote:
        return 0
    path = catalog_path or ONTOLOGY_YAML
    existing = path.read_text(encoding="utf-8").rstrip("\n")
    lines: list[str] = []
    if _SECTION_HEADER not in existing:
        lines.append(_SECTION_HEADER)
    for entry in plan.promote:
        aliases = sorted(v for v in entry.variants if v != entry.term)
        alias_part = ""
        if aliases:
            rendered = ", ".join(f'"{_yaml_escape(a)}"' for a in aliases)
            alias_part = f", aliases: [{rendered}]"
        lines.append(
            f'  - {{id: "{_yaml_escape(entry.term)}", '
            f'pref: "{_yaml_escape(entry.term)}"{alias_part}}}  # df={entry.df}'
        )
    path.write_text(existing + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    logger.info("promoted %d terms into %s", len(plan.promote), path)
    return len(plan.promote)


def repair_map(plan: Plan) -> dict[str, str]:
    """Singleton proposal -> the catalog term it folds onto."""
    return {entry.term: anchor for entry, anchor in plan.repair}


def report(plan: Plan, limit: int = 25) -> str:
    out: list[str] = [f"proposals: {plan.summary()}", ""]
    if plan.promote:
        out.append(f"PROMOTE (df >= threshold) - {len(plan.promote)}")
        for entry in plan.promote[:limit]:
            out.append(f"   df={entry.df:<4} {entry.term}")
        out.append("")
    if plan.repair:
        out.append(f"REPAIR onto an existing term - {len(plan.repair)}")
        for entry, anchor in plan.repair[:limit]:
            out.append(f"   {entry.term}  ->  {anchor}")
        out.append("")
    if plan.drop:
        out.append(f"DROP (df=1, anchored to nothing) - {len(plan.drop)}")
        for entry in plan.drop[:limit]:
            out.append(f"   {entry.term}")
    truncated = max(
        len(plan.promote) - limit, len(plan.repair) - limit, len(plan.drop) - limit, 0
    )
    if truncated:
        out.append(f"\n({truncated} more not shown)")
    return "\n".join(out)


def counts_by_source(proposals: dict[str, Proposal]) -> dict[str, int]:
    """How many proposals each hadith made -- a proxy for vocabulary gaps."""
    per_source: dict[str, int] = defaultdict(int)
    for entry in proposals.values():
        for source in set(entry.sources):
            per_source[source] += 1
    return dict(per_source)
