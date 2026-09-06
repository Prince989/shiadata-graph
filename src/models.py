"""Pydantic contracts for Phase 1 structured Gemini output."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# No "ayah": Qur'an citations are read off the printed page by
# src/extractors/quran_refs.py and injected as a separate quran_refs field. The
# model had no reliable way to produce them -- the prompt's own gold example
# taught 39:9 for a verse the editor cites as 2:269 -- so the schema no longer
# offers it a channel to guess.
NodeType = Literal["concept", "person", "place", "group", "event", "work"]


class SemanticNode(BaseModel):
    node: str
    type: NodeType = "concept"
    role: Literal["primary", "secondary"] = "secondary"

    @field_validator("node")
    @classmethod
    def _clean(cls, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if not text:
            raise ValueError("empty node")
        return text


class Mention(BaseModel):
    """One thing this narration is about, as the model saw it.

    A mention is NOT a graph node. It is an observation, in whatever words the
    matn used. Identity -- deciding that عقل المرء and العقل are the same thing,
    and that معاوية is one person across every book -- is resolved afterwards
    over the whole corpus by src/pipelines/resolver.py. Asking the model to
    settle identity one hadith at a time was the original mistake: it cannot see
    the other 15,000 narrations, so it invented a fresh label for each.

    `evidence` is the span of the matn the mention came from. It makes every
    mention checkable against the text, which is a stronger guard against
    invention than any rule about phrasing.
    """

    text: str
    type: NodeType = "concept"
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = ""

    @field_validator("text")
    @classmethod
    def _clean_text(cls, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if not text:
            raise ValueError("empty mention")
        return text


class QuotedSpan(BaseModel):
    """A passage the narration quotes rather than asserts.

    The model marks the span and says what kind it is; it is never asked for a
    verse number. Resolving the span to sura:ayah is done against the actual
    mushaf in src/extractors/quran_refs.py, because a model asked for a citation
    will supply a plausible one -- the old prompt's own gold example taught 39:9
    for a verse the editor cites as 2:269.
    """

    text: str
    kind: Literal["quran", "hadith", "other"] = "quran"


class HadithExtraction(BaseModel):
    """One narration (page fragment or flushed complete hadith)."""

    marker: str = ""
    locator: str = ""
    page_start: str = ""
    page_end: str = ""
    hadith: str
    hadith_fa: str
    hadith_en: str
    # What the model observed, in the matn's own words. Resolved to graph nodes
    # afterwards, corpus-wide. This is the field the extractor should fill.
    mentions: list[Mention] = Field(default_factory=list)
    quotes: list[QuotedSpan] = Field(default_factory=list)
    # Legacy channels, still read so payloads extracted before the resolver
    # landed keep working. New extractions leave both empty.
    semantic_nodes: list[SemanticNode] = Field(default_factory=list)
    proposed_nodes: list[str] = Field(default_factory=list)
    ravis: list[str] = Field(default_factory=list)


class HadithPageExtraction(BaseModel):
    """Internal per-page Gemini extract; not the Phase 1 product."""

    page: str
    hadiths: list[HadithExtraction] = Field(default_factory=list)


class HadithUnify(BaseModel):
    """Fill translations / ravis / semantic_nodes when the page pass left them empty."""

    hadith_fa: str = ""
    hadith_en: str = ""
    mentions: list[Mention] = Field(default_factory=list)
    quotes: list[QuotedSpan] = Field(default_factory=list)
    semantic_nodes: list[SemanticNode] = Field(default_factory=list)
    proposed_nodes: list[str] = Field(default_factory=list)
    ravis: list[str] = Field(default_factory=list)

    @field_validator("mentions")
    @classmethod
    def _dedupe_mentions(cls, value: list[Mention]) -> list[Mention]:
        cleaned: list[Mention] = []
        seen: set[tuple[str, str]] = set()
        for item in value or []:
            key = (item.text, item.type)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(item)
        return cleaned

    @model_validator(mode="after")
    def _must_say_something(self) -> "HadithUnify":
        """Reject a hollow unify so the agent retries.

        Unify only runs on narrations already known to be under-extracted, so an
        answer carrying nothing to index is a failed call, not a valid result.
        Either channel satisfies it: new extractions fill `mentions`, and
        payloads from before the resolver still arrive as `semantic_nodes`.
        """
        # One is enough. Unify also runs when only a translation is missing,
        # and demanding two there pressures the model into inventing a second
        # observation for a narration that genuinely has one. Grounding would
        # usually catch the invention, but not asking for it is better.
        if len(self.mentions) + len(self.semantic_nodes) < 1:
            raise ValueError("need at least 1 mention (or legacy semantic_node)")
        return self


class TafsirExtraction(BaseModel):
    ayah_anchor: str
    # Same mention channel as hadith, so tafsir and hadith resolve into ONE node
    # space. Previously tafsir emitted free-text core_concepts that could never
    # merge with a hadith node, which walled the two pipelines off from each
    # other entirely.
    mentions: list[Mention] = Field(default_factory=list)
    core_concepts: list[str] = Field(default_factory=list)
    referenced_hadith: str = ""
    summary_fa: str
    tafsir_chunk: str


class HistoricalEvent(BaseModel):
    event_title: str
    mentions: list[Mention] = Field(default_factory=list)
    characters_involved: list[str] = Field(default_factory=list)
    historical_concepts: list[str] = Field(default_factory=list)
    historical_chunk: str


class HistoryExtraction(BaseModel):
    events: list[HistoricalEvent] = Field(default_factory=list)


class DuplicateVerdict(BaseModel):
    duplicate: bool
    reason: str = ""


class EdgeRelation(BaseModel):
    relation: Literal["SUPPORTS", "CONTRADICTS", "EXCEPTS", "UNRELATED"]
    reason: str = ""
