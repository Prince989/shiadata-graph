"""Concept catalog: canonicalize hadith tags into grouping IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import yaml

from config.paths import ONTOLOGY_YAML

_ALEF = re.compile(r"[إأآٱ]")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Concept:
    id: str
    pref: str
    aliases: tuple[str, ...]
    broader: str | None
    group: bool


def normalize_ar(text: str) -> str:
    s = _WS.sub(" ", (text or "").strip())
    s = _ALEF.sub("ا", s)
    s = s.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
    if s.startswith("ال") and len(s) > 3:
        s = s[2:]
    return s


def _parse_concept(raw) -> Concept | None:
    if isinstance(raw, str):
        label = raw.strip()
        if not label:
            return None
        return Concept(id=label, pref=label, aliases=(), broader=None, group=False)
    if not isinstance(raw, dict):
        return None
    pref = str(raw.get("pref") or raw.get("id") or "").strip()
    cid = str(raw.get("id") or pref).strip()
    if not pref:
        return None
    aliases = tuple(str(a).strip() for a in (raw.get("aliases") or []) if str(a).strip())
    broader = raw.get("broader")
    broader_s = str(broader).strip() if broader else None
    group = bool(raw.get("group", False))
    return Concept(id=cid, pref=pref, aliases=aliases, broader=broader_s, group=group)


@lru_cache(maxsize=1)
def load_concept_catalog() -> tuple[Concept, ...]:
    data = yaml.safe_load(ONTOLOGY_YAML.read_text(encoding="utf-8")) or {}
    items = data.get("concepts") or []
    out: list[Concept] = []
    seen: set[str] = set()
    for raw in items:
        concept = _parse_concept(raw)
        if concept is None or concept.id in seen:
            continue
        seen.add(concept.id)
        out.append(concept)
    return tuple(out)


def grouping_prefs() -> list[str]:
    return [c.pref for c in load_concept_catalog() if c.group]


def load_ontology() -> list[str]:
    """Backward-compatible: grouping prefs, else all prefs."""
    grouped = grouping_prefs()
    if grouped:
        return grouped
    return [c.pref for c in load_concept_catalog()]


def _index() -> dict[str, Concept]:
    table: dict[str, Concept] = {}
    for concept in load_concept_catalog():
        keys = (concept.id, concept.pref, *concept.aliases)
        for key in keys:
            table[normalize_ar(key)] = concept
    return table


def lookup_concept(label: str) -> Concept | None:
    folded = normalize_ar(label)
    if not folded:
        return None
    return _index().get(folded)


def canonicalize_tag(label: str) -> str:
    """Map a raw label to catalog pref, or return the stripped original."""
    stripped = (label or "").strip()
    if not stripped:
        return ""
    hit = lookup_concept(stripped)
    return hit.pref if hit else stripped


def is_grouping_label(label: str) -> bool:
    hit = lookup_concept(label)
    if hit is None:
        return True
    return hit.group


def remap_tag_list(tags: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in tags or []:
        mapped = canonicalize_tag(str(raw))
        if not mapped or mapped in seen:
            continue
        seen.add(mapped)
        out.append(mapped)
    return out


def remap_hadith_payload(payload: dict) -> dict:
    """Return a copy with hadith tags passed through the alias table."""
    data = dict(payload)
    items = data.get("hadiths")
    if isinstance(items, list):
        remapped_items = []
        for item in items:
            if isinstance(item, dict):
                row = dict(item)
                row["tags"] = remap_tag_list(row.get("tags") or [])
                remapped_items.append(row)
            else:
                remapped_items.append(item)
        data["hadiths"] = remapped_items
    if data.get("tags") is not None:
        data["tags"] = remap_tag_list(list(data.get("tags") or []))
    return data
