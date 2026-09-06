"""Merge per-page hadith extracts into complete narrations (no Gemini here)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.extractors.chunkers import next_page_continues, page_prefix_and_starts, strip_folklib_footnotes
from src.pipelines.ontology import enforce_node_policy, semantic_nodes_of


def _extend_unique(target: list, incoming, key, upgrade=None) -> None:
    """Union dict items by `key`, letting a later copy improve an earlier one.

    Strict first-wins loses information across a spanning hadith: page 2 may
    report the same mention with a better evidence span or a higher salience
    than page 1, and dropping it outright throws that away.
    """
    index = {key(item): item for item in target if isinstance(item, dict)}
    for item in incoming or []:
        if not isinstance(item, dict):
            continue
        identity = key(item)
        if not identity:
            continue
        existing = index.get(identity)
        if existing is None:
            index[identity] = item
            target.append(item)
        elif upgrade is not None:
            upgrade(existing, item)


def _resolve_quotes(quotes: list, existing_refs: list[str]) -> list[str]:
    """Turn the model's quoted spans into verse refs, unioned with the page scan.

    The model marks WHERE it saw a quotation; the mushaf decides WHICH verse.
    Without this the `quotes` field was collected and thrown away, and citation
    coverage depended entirely on the page-level scan -- which attributes by
    footnote marker and page segment, and so misses a quotation the model spotted
    inside a narration the scan assigned elsewhere.
    """
    from src.extractors.quran_refs import match_quran

    refs = list(existing_refs)
    for quote in quotes or []:
        if not isinstance(quote, dict) or quote.get("kind", "quran") != "quran":
            continue
        for ref in match_quran(str(quote.get("text") or "")):
            if ref not in refs:
                refs.append(ref)
    return refs


def _upgrade_mention(existing: dict, other: dict) -> None:
    if float(other.get("salience") or 0) > float(existing.get("salience") or 0):
        existing["salience"] = other.get("salience")
    if not str(existing.get("evidence") or "").strip():
        existing["evidence"] = other.get("evidence") or ""


def norm_marker(marker: str) -> str:
    return re.sub(r"\s+", "", (marker or "").strip())


def span_locator(page_start: str, page_end: str) -> str:
    if page_start == page_end:
        return page_start
    prefix_a, sep_a, rest_a = page_start.partition("صفحه")
    prefix_b, sep_b, rest_b = page_end.partition("صفحه")
    if sep_a and sep_b and prefix_a == prefix_b:
        return f"{prefix_a}صفحه {rest_a.strip()} تا {rest_b.strip()}"
    return f"{page_start} تا {page_end}"


def match_gemini_item(items: list[dict], token: str | None) -> dict:
    if not items:
        return {}
    if token is None or token == "continuation":
        for item in items:
            if norm_marker(str(item.get("marker") or "")) in {"continuation", ""}:
                return item
        return items[0]
    want = norm_marker(token)
    for item in items:
        if norm_marker(str(item.get("marker") or "")) == want:
            return item
    return {}


@dataclass
class OpenHadith:
    marker: str
    page_start: str
    page_end: str
    arabic: list[str] = field(default_factory=list)
    fa: list[str] = field(default_factory=list)
    en: list[str] = field(default_factory=list)
    ravis_seed: list[str] = field(default_factory=list)
    semantic_nodes_seed: list = field(default_factory=list)
    quran_refs_seed: list[str] = field(default_factory=list)
    proposed_seed: list[str] = field(default_factory=list)
    # Unioned across pages, not first-wins: a hadith spanning two pages is
    # extracted as two fragments, and each sees only its own half of the matn.
    mentions_seed: list = field(default_factory=list)
    quotes_seed: list = field(default_factory=list)
    kitab: str = ""
    bab: str = ""

    def append_slice(
        self,
        locator: str,
        arabic: str,
        fa: str = "",
        en: str = "",
        ravis: list[str] | None = None,
        semantic_nodes: list | None = None,
        quran_refs: list[str] | None = None,
        proposed: list[str] | None = None,
        mentions: list | None = None,
        quotes: list | None = None,
    ) -> None:
        self.page_end = locator
        if arabic:
            self.arabic.append(arabic)
        if fa:
            self.fa.append(fa)
        if en:
            self.en.append(en)
        if ravis and not self.ravis_seed:
            self.ravis_seed = list(ravis)
        if semantic_nodes and not self.semantic_nodes_seed:
            self.semantic_nodes_seed = list(semantic_nodes)
        # Unioned rather than first-wins: hadith 11 spans pages 12 and 13 and is
        # footnoted on both, so keeping only the first page's refs would drop
        # the البقرة: 269 citation that is the whole point of that narration.
        for ref in quran_refs or []:
            if ref not in self.quran_refs_seed:
                self.quran_refs_seed.append(ref)
        for term in proposed or []:
            if term and term not in self.proposed_seed:
                self.proposed_seed.append(term)
        _extend_unique(
            self.mentions_seed,
            mentions,
            lambda m: (str(m.get("text") or ""), str(m.get("type") or "concept")),
            _upgrade_mention,
        )
        _extend_unique(self.quotes_seed, quotes, lambda q: str(q.get("text") or ""))

    def to_dict(self) -> dict:
        return {
            "marker": self.marker,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "arabic": self.arabic,
            "fa": self.fa,
            "en": self.en,
            "ravis_seed": self.ravis_seed,
            "semantic_nodes_seed": self.semantic_nodes_seed,
            "quran_refs_seed": self.quran_refs_seed,
            "proposed_seed": self.proposed_seed,
            "mentions_seed": self.mentions_seed,
            "quotes_seed": self.quotes_seed,
            "kitab": self.kitab,
            "bab": self.bab,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OpenHadith:
        seed = (
            data.get("semantic_nodes_seed")
            or data.get("concept_nodes_seed")
            or data.get("tags_seed")
            or []
        )
        return cls(
            marker=str(data.get("marker") or ""),
            page_start=str(data.get("page_start") or ""),
            page_end=str(data.get("page_end") or ""),
            arabic=list(data.get("arabic") or []),
            fa=list(data.get("fa") or []),
            en=list(data.get("en") or []),
            ravis_seed=list(data.get("ravis_seed") or []),
            semantic_nodes_seed=list(seed),
            quran_refs_seed=list(data.get("quran_refs_seed") or []),
            proposed_seed=list(data.get("proposed_seed") or []),
            mentions_seed=list(data.get("mentions_seed") or []),
            quotes_seed=list(data.get("quotes_seed") or []),
            kitab=str(data.get("kitab") or ""),
            bab=str(data.get("bab") or ""),
        )

    def assemble(self) -> dict:
        from src.pipelines.grounding import ground_mentions

        matn = "\n".join(p for p in self.arabic if p).strip()
        # Ground against the ASSEMBLED matn, not the page fragment. A mention
        # taken from page 2 of a two-page hadith is not in page 1's text, so
        # checking per fragment would reject half of a spanning narration.
        mentions, _ = ground_mentions(self.mentions_seed, matn, self.ravis_seed)
        refs = _resolve_quotes(self.quotes_seed, self.quran_refs_seed)
        return {
            "marker": self.marker,
            "locator": span_locator(self.page_start, self.page_end),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "hadith": matn,
            "hadith_fa": "\n".join(p for p in self.fa if p).strip(),
            "hadith_en": "\n".join(p for p in self.en if p).strip(),
            # The live contract. Resolved corpus-wide by `main.py resolve-nodes`.
            "mentions": mentions,
            "quotes": list(self.quotes_seed),
            # Re-gate here as well as per page: this is the first point where the
            # assembled hadith's nodes and its full isnad are both in hand.
            "semantic_nodes": enforce_node_policy(
                semantic_nodes_of({"semantic_nodes": self.semantic_nodes_seed}),
                self.ravis_seed,
            ),
            "ravis": list(self.ravis_seed),
            "quran_refs": refs,
            # Never graph nodes. Counted across the corpus by
            # src/pipelines/proposals.py, which promotes what recurs.
            "proposed_nodes": list(self.proposed_seed),
            # The book's own classification. Authored by Kulayni, not inferred.
            "kitab": self.kitab,
            "bab": self.bab,
        }


def _slice_from_item(
    token: str,
    body: str,
    locator: str,
    item: dict,
    quran_refs: list[str] | None = None,
) -> dict:
    return {
        "marker": token,
        "locator": locator,
        "arabic": body or str(item.get("hadith") or ""),
        "fa": str(item.get("hadith_fa") or ""),
        "en": str(item.get("hadith_en") or ""),
        "ravis": list(item.get("ravis") or []),
        "semantic_nodes": semantic_nodes_of(item),
        "proposed_nodes": [str(x) for x in (item.get("proposed_nodes") or []) if str(x).strip()],
        "mentions": [m for m in (item.get("mentions") or []) if isinstance(m, dict)],
        "quotes": [q for q in (item.get("quotes") or []) if isinstance(q, dict)],
        "quran_refs": list(quran_refs or []),
    }


def _single(slice_: dict, locator: str, kitab: str = "", bab: str = "") -> dict:
    buf = OpenHadith(
        marker=slice_["marker"], page_start=locator, page_end=locator, kitab=kitab, bab=bab
    )
    buf.append_slice(
        locator,
        slice_["arabic"],
        slice_["fa"],
        slice_["en"],
        slice_["ravis"],
        slice_["semantic_nodes"],
        slice_.get("quran_refs"),
        slice_.get("proposed_nodes"),
        slice_.get("mentions"),
        slice_.get("quotes"),
    )
    return buf.assemble()


def consume_page(
    locator: str,
    text: str,
    gemini_items: list[dict],
    buffer: OpenHadith | None,
    next_text: str | None,
    quran_refs: dict[str, list[str]] | None = None,
    section: tuple[str, str] | None = None,
) -> tuple[list[dict], OpenHadith | None]:
    """Apply lookahead markers; return complete hadiths and the open buffer.

    `quran_refs` maps this page's start tokens (plus "continuation") to verse
    refs read off the raw page by `page_quran_refs`.
    """
    flushed: list[dict] = []
    refs = quran_refs or {}
    kitab, bab = section or ("", "")
    text = strip_folklib_footnotes(text)
    next_text = strip_folklib_footnotes(next_text) if next_text else next_text
    leading, starts = page_prefix_and_starts(text)
    buf = buffer

    if buf and leading:
        item = match_gemini_item(gemini_items, "continuation")
        # A continuation fragment is the second half of a narration, and it
        # carries observations the first page could not: the matn it quotes, the
        # people it names, and often the isnad's tail. Passing only the
        # translations meant every mention made on page 2 was discarded, so the
        # cross-page union the design depends on never happened for the half of
        # a spanning hadith that most needed it.
        buf.append_slice(
            locator,
            leading,
            str(item.get("hadith_fa") or ""),
            str(item.get("hadith_en") or ""),
            ravis=list(item.get("ravis") or []),
            quran_refs=refs.get("continuation"),
            mentions=[m for m in (item.get("mentions") or []) if isinstance(m, dict)],
            quotes=[q for q in (item.get("quotes") or []) if isinstance(q, dict)],
        )

    if buf and starts:
        flushed.append(buf.assemble())
        buf = None

    for i, (token, body) in enumerate(starts):
        item = match_gemini_item(gemini_items, token)
        slice_ = _slice_from_item(token, body, locator, item, refs.get(token))
        is_last = i == len(starts) - 1
        hold = is_last and next_page_continues(next_text)
        if hold:
            buf = OpenHadith(
                marker=token, page_start=locator, page_end=locator, kitab=kitab, bab=bab
            )
            buf.append_slice(
                locator,
                slice_["arabic"],
                slice_["fa"],
                slice_["en"],
                slice_["ravis"],
                slice_["semantic_nodes"],
                slice_["quran_refs"],
                slice_["proposed_nodes"],
                slice_["mentions"],
                slice_["quotes"],
            )
        else:
            flushed.append(_single(slice_, locator, kitab, bab))

    if not starts and buf and not next_page_continues(next_text):
        flushed.append(buf.assemble())
        buf = None

    return flushed, buf
