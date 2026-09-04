"""Merge per-page hadith extracts into complete narrations (no Gemini here)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.extractors.chunkers import next_page_continues, page_prefix_and_starts, strip_folklib_footnotes


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
    tags_seed: list[str] = field(default_factory=list)

    def append_slice(
        self,
        locator: str,
        arabic: str,
        fa: str = "",
        en: str = "",
        ravis: list[str] | None = None,
        tags: list[str] | None = None,
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
        if tags and not self.tags_seed:
            self.tags_seed = list(tags)

    def to_dict(self) -> dict:
        return {
            "marker": self.marker,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "arabic": self.arabic,
            "fa": self.fa,
            "en": self.en,
            "ravis_seed": self.ravis_seed,
            "tags_seed": self.tags_seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> OpenHadith:
        return cls(
            marker=str(data.get("marker") or ""),
            page_start=str(data.get("page_start") or ""),
            page_end=str(data.get("page_end") or ""),
            arabic=list(data.get("arabic") or []),
            fa=list(data.get("fa") or []),
            en=list(data.get("en") or []),
            ravis_seed=list(data.get("ravis_seed") or []),
            tags_seed=list(data.get("tags_seed") or []),
        )

    def assemble(self) -> dict:
        return {
            "marker": self.marker,
            "locator": span_locator(self.page_start, self.page_end),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "hadith": "\n".join(p for p in self.arabic if p).strip(),
            "hadith_fa": "\n".join(p for p in self.fa if p).strip(),
            "hadith_en": "\n".join(p for p in self.en if p).strip(),
            "tags": list(self.tags_seed),
            "ravis": list(self.ravis_seed),
        }


def _slice_from_item(
    token: str,
    body: str,
    locator: str,
    item: dict,
) -> dict:
    return {
        "marker": token,
        "locator": locator,
        "arabic": body or str(item.get("hadith") or ""),
        "fa": str(item.get("hadith_fa") or ""),
        "en": str(item.get("hadith_en") or ""),
        "ravis": list(item.get("ravis") or []),
        "tags": list(item.get("tags") or []),
    }


def _single(slice_: dict, locator: str) -> dict:
    buf = OpenHadith(marker=slice_["marker"], page_start=locator, page_end=locator)
    buf.append_slice(
        locator,
        slice_["arabic"],
        slice_["fa"],
        slice_["en"],
        slice_["ravis"],
        slice_["tags"],
    )
    return buf.assemble()


def consume_page(
    locator: str,
    text: str,
    gemini_items: list[dict],
    buffer: OpenHadith | None,
    next_text: str | None,
) -> tuple[list[dict], OpenHadith | None]:
    """Apply lookahead markers; return complete hadiths and the open buffer."""
    flushed: list[dict] = []
    text = strip_folklib_footnotes(text)
    next_text = strip_folklib_footnotes(next_text) if next_text else next_text
    leading, starts = page_prefix_and_starts(text)
    buf = buffer

    if buf and leading:
        item = match_gemini_item(gemini_items, "continuation")
        buf.append_slice(
            locator,
            leading,
            str(item.get("hadith_fa") or ""),
            str(item.get("hadith_en") or ""),
        )

    if buf and starts:
        flushed.append(buf.assemble())
        buf = None

    for i, (token, body) in enumerate(starts):
        item = match_gemini_item(gemini_items, token)
        slice_ = _slice_from_item(token, body, locator, item)
        is_last = i == len(starts) - 1
        hold = is_last and next_page_continues(next_text)
        if hold:
            buf = OpenHadith(marker=token, page_start=locator, page_end=locator)
            buf.append_slice(
                locator,
                slice_["arabic"],
                slice_["fa"],
                slice_["en"],
                slice_["ravis"],
                slice_["tags"],
            )
        else:
            flushed.append(_single(slice_, locator))

    if not starts and buf and not next_page_continues(next_text):
        flushed.append(buf.assemble())
        buf = None

    return flushed, buf
