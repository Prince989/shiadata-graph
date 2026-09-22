"""Al-Mizan prepare + persist helpers.

Section split happens before Gemini. `tafsir_chunk` is the source unit, never
model output. `ayah_anchor` stays the Folklib range. `quran_refs` /
`quran_cites` are mushaf matches on this section, not the whole banner.
"""

from __future__ import annotations

import logging

from src.extractors.chunkers import tafsir_section_units
from src.extractors.epub_parser import ParsedUnit
from src.extractors.quran_refs import resolve_tafsir_quran_refs
from src.pipelines.grounding import ground_mentions

logger = logging.getLogger(__name__)


def prepare(units: list[ParsedUnit]) -> list[ParsedUnit]:
    return tafsir_section_units(units)


def finalize_payload(unit: ParsedUnit, data: dict) -> dict:
    """Attach source text and verse refs; drop ungrounded mentions."""
    out = dict(data)
    range_locator = (unit.locator or "").split("|", 1)[0].strip()
    out["ayah_anchor"] = str(out.get("ayah_anchor") or range_locator).strip() or range_locator
    out["tafsir_chunk"] = unit.text
    out["core_concepts"] = []
    out["referenced_hadith"] = ""
    comments, cites = resolve_tafsir_quran_refs(
        unit.locator or str(out.get("ayah_anchor") or ""),
        unit.text,
        out.get("quotes"),
    )
    out["quran_refs"] = comments
    out["quran_cites"] = cites
    mentions, dropped = ground_mentions(
        out.get("mentions") or [],
        unit.text,
        quotes=out.get("quotes"),
        cut_isnad=False,
    )
    if dropped:
        logger.info("tafsir grounded drop %s on %s", len(dropped), unit.locator)
    out["mentions"] = mentions
    return out
