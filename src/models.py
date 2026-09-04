"""Pydantic contracts for Phase 1 structured Gemini output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HadithExtraction(BaseModel):
    """One narration (page fragment or flushed complete hadith)."""

    marker: str = ""
    locator: str = ""
    page_start: str = ""
    page_end: str = ""
    hadith: str
    hadith_fa: str
    hadith_en: str
    tags: list[str] = Field(default_factory=list)
    ravis: list[str] = Field(default_factory=list)


class HadithPageExtraction(BaseModel):
    """Internal per-page Gemini extract; not the Phase 1 product."""

    page: str
    hadiths: list[HadithExtraction] = Field(default_factory=list)


class HadithUnify(BaseModel):
    """Second-pass tags and ravis for a multi-page assembled matn."""

    tags: list[str] = Field(default_factory=list)
    ravis: list[str] = Field(default_factory=list)


class TafsirExtraction(BaseModel):
    ayah_anchor: str
    core_concepts: list[str] = Field(default_factory=list)
    referenced_hadith: str = ""
    summary_fa: str
    tafsir_chunk: str


class HistoricalEvent(BaseModel):
    event_title: str
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
