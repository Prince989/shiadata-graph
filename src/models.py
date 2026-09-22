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

    @field_validator("salience", mode="before")
    @classmethod
    def _parse_salience(cls, value: object) -> float:
        """Qwen often emits high/medium/low instead of 0.0-1.0."""
        if isinstance(value, bool):
            return 0.9 if value else 0.3
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value or "").strip().lower().replace("_", " ")
        words = {
            "very high": 0.95,
            "high": 0.9,
            "primary": 0.9,
            "medium": 0.6,
            "moderate": 0.6,
            "low": 0.3,
            "secondary": 0.4,
            "very low": 0.2,
        }
        if text in words:
            return words[text]
        try:
            return float(text)
        except ValueError:
            return 0.5

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
    is_encyclopedic: bool = Field(default=False, description="true ONLY if the matn explicitly enumerates a massive list (>5-15 items, attributes, or classes). Otherwise false.")
    # Mentions are required in the prompt and enforced at unify when the page
    # pass leaves them empty (HadithUnifyRequireTopics). A hard gate here used
    # to reject the entire page extract after 6 retries even when FA/ravis were
    # fine, which stalled Phase 1 with errors=1 and nothing flushed.


class HadithPageItem(BaseModel):
    """What the page pass actually asks Gemini for.

    Field order here is not cosmetic: it becomes the JSON-schema property order,
    and a model fills a structured response in schema order regardless of what
    the prompt says to do first. `HadithExtraction` put `hadith`, `hadith_fa`
    and `hadith_en` ahead of `mentions`, so on a six-hadith page the model wrote
    six full Arabic matns and twelve translations before reaching the one field
    that matters -- and arrived there with the budget spent, which is why
    mentions kept coming back empty.

    `hadith` is gone entirely. `_slice_from_item` takes the matn from the page
    split (`body or item["hadith"]`, and body always wins), so echoing it back
    was the single largest field in the response and was discarded on arrival.

    The legacy `semantic_nodes` / `proposed_nodes` are gone too: offering them
    gave the model a second place to put topics, and anything it put there was
    dropped by the resolver.
    """

    marker: str = ""
    mentions: list[Mention] = Field(default_factory=list)
    ravis: list[str] = Field(default_factory=list)
    quotes: list[QuotedSpan] = Field(default_factory=list)
    hadith_fa: str = ""
    hadith_en: str = ""
    is_encyclopedic: bool = Field(
        default=False,
        description=(
            "true ONLY if the matn explicitly enumerates a massive list "
            "(>10-15 items, attributes, or classes). Otherwise false."
        ),
    )


class HadithPageExtraction(BaseModel):
    """Internal per-page Gemini extract; not the Phase 1 product."""

    page: str
    hadiths: list[HadithPageItem] = Field(default_factory=list)


def _dedupe_mention_list(value: list[Mention]) -> list[Mention]:
    cleaned: list[Mention] = []
    seen: set[tuple[str, str]] = set()
    for item in value or []:
        key = (item.text, item.type)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


class HadithUnify(BaseModel):
    """Fill translations / ravis / mentions when the page pass left gaps.

    Soft by default: FA/ravis-only replies are valid when the assembled payload
    already has topics. Use `HadithUnifyRequireTopics` when it does not.
    """

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
        return _dedupe_mention_list(value)


class HadithUnifyRequireTopics(HadithUnify):
    """Unify when the assembled narration still has no topics to resolve.

    `min_length=2` becomes JSON-schema minItems so Gemini cannot emit [].
    """

    mentions: list[Mention] = Field(min_length=2)

    @field_validator("mentions")
    @classmethod
    def _dedupe_mentions(cls, value: list[Mention]) -> list[Mention]:
        return _dedupe_mention_list(value)

    @model_validator(mode="after")
    def _must_say_something(self) -> "HadithUnifyRequireTopics":
        # Mentions only -- legacy semantic_nodes used to satisfy this gate, then
        # enforce_node_policy dropped them and the payload was saved with FA/ravis
        # and empty mentions. resolve-nodes then had nothing to do.
        if len(self.mentions) < 2:
            raise ValueError(
                "need at least 2 mentions with text/type/salience/evidence"
            )
        return self


class MentionsFill(BaseModel):
    """Mentions-only recovery when the page pass left topics empty.

    Separate from unify so the model cannot burn the whole budget on FA/ravis
    and return mentions: []. minItems is enforced in the response schema.
    """

    mentions: list[Mention] = Field(min_length=2, max_length=12)

    @field_validator("mentions")
    @classmethod
    def _dedupe_mentions(cls, value: list[Mention]) -> list[Mention]:
        cleaned = _dedupe_mention_list(value)
        if len(cleaned) < 2:
            raise ValueError(
                "need at least 2 mentions with text/type/salience/evidence"
            )
        return cleaned


class MentionsFillExhaustive(BaseModel):
    """Encyclopedic MentionsFill: inventories may need dozens of mentions per chunk.

    Used only when `is_encyclopedic` is set. Chunked across page fragments so
    each call stays within Gemini structured-output limits. Normal MentionsFill
    stays capped at 12.
    """

    mentions: list[Mention] = Field(min_length=2, max_length=80)

    @field_validator("mentions")
    @classmethod
    def _dedupe_mentions(cls, value: list[Mention]) -> list[Mention]:
        cleaned = _dedupe_mention_list(value)
        if len(cleaned) < 2:
            raise ValueError(
                "need at least 2 mentions with text/type/salience/evidence"
            )
        return cleaned


class CitedHadith(BaseModel):
    """A narration Tabatabai quotes; not a CanonicalHadith.

    `span` is the verbatim evidence from this unit (usually Persian).
    `text_ar` / `text_fa` / `text_en` are the same narration in three
    languages so a citation can meet a hadith payload later.
    """

    source_work: str = ""
    speaker: str = ""
    span: str = ""
    text_ar: str = ""
    text_fa: str = ""
    text_en: str = ""


class TafsirExtraction(BaseModel):
    """What the tafsir pass asks Gemini for.

    Field order is the JSON-schema property order. `tafsir_chunk` is gone: the
    Folklib unit is already on disk. The model fills fluent Arabic, Persian,
    and English of THIS unit, and three-language text on every cited hadith.

    `core_concepts` / `referenced_hadith` are not requested. Resolve still
    upcasts legacy payloads that have them.
    """

    ayah_anchor: str
    mentions: list[Mention] = Field(default_factory=list)
    quotes: list[QuotedSpan] = Field(default_factory=list)
    cited_hadiths: list[CitedHadith] = Field(default_factory=list)
    tafsir_ar: str = ""
    tafsir_fa: str = ""
    tafsir_en: str = ""


class HistoricalEvent(BaseModel):
    event_title: str
    mentions: list[Mention] = Field(default_factory=list)
    characters_involved: list[str] = Field(default_factory=list)
    historical_concepts: list[str] = Field(default_factory=list)
    historical_chunk: str


class HistoryExtraction(BaseModel):
    events: list[HistoricalEvent] = Field(default_factory=list)


class Adjudication(BaseModel):
    """One verdict on one orphan label, from the offline vocabulary adjudicator.

    A closed set of three moves. The model may merge a label into an existing
    term, promote it to a term of its own, or reject it -- it may not invent a
    hierarchy, and it never sees a hadith.
    """

    label: str
    verdict: Literal["alias", "new", "drop"]
    target: str = ""


class AdjudicationBatch(BaseModel):
    verdicts: list[Adjudication] = Field(default_factory=list)


class AliasProposal(BaseModel):
    """Other wordings one concept is written in.

    Proposals only. Nothing here reaches the catalog until the corpus is checked
    for the wording -- see `src.pipelines.aliases.ground`.
    """

    concept: str
    aliases: list[str] = Field(default_factory=list)


class AliasBatch(BaseModel):
    entries: list[AliasProposal] = Field(default_factory=list)


class DuplicateVerdict(BaseModel):
    duplicate: bool
    reason: str = ""


class EdgeRelation(BaseModel):
    relation: Literal["SUPPORTS", "CONTRADICTS", "EXCEPTS", "UNRELATED"]
    reason: str = ""
