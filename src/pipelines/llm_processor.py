"""Phase 1: send one text unit through Gemini structured output via GeminiAgent."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel

from src.agents.gemini import GeminiAgent
from src.extractors.chunkers import split_hadith_page, strip_folklib_footnotes
from src.extractors.epub_parser import ParsedUnit
from src.models import (
    HadithPageExtraction,
    HadithUnify,
    HistoryExtraction,
    MentionsFill,
    TafsirExtraction,
)
from src.pipelines.ontology import (
    enforce_node_policy,
    remap_hadith_payload,
    semantic_nodes_of,
)
from src.pipelines.prompts import (
    HISTORY_PROMPT,
    TAFSIR_PROMPT,
    mentions_fill_prompt,
    hadith_prompt,
    unify_prompt,
)
from src.state_manager import ChunkStatus, StateManager

logger = logging.getLogger(__name__)

SKIP_MARKERS = ("رقم الصفحة", "عناوين الأبواب", "عدد الأحاديث")


def chunk_id(book_id: str, locator: str, text: str) -> str:
    payload = f"{book_id}\n{locator}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def should_skip(text: str, min_chars: int) -> bool:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    return any(marker in stripped for marker in SKIP_MARKERS)


def phase1_filename(source_path: str, locator: str, cid: str, marker: str = "") -> str:
    stem = Path(source_path).stem or "unknown"
    loc = re.sub(r"[^\w\u0600-\u06FF]+", "_", locator, flags=re.UNICODE)
    loc = re.sub(r"_+", "_", loc).strip("_")[:120]
    mark = re.sub(r"[^\w\u0600-\u06FF]+", "_", marker, flags=re.UNICODE)
    mark = re.sub(r"_+", "_", mark).strip("_")[:32]
    if mark and loc:
        return f"{stem}__{mark}__{loc}.json"
    if loc:
        return f"{stem}__{loc}.json"
    return f"{stem}__{cid[:16]}.json"
    stem = Path(source_path).stem or "unknown"
    loc = re.sub(r"[^\w\u0600-\u06FF]+", "_", locator, flags=re.UNICODE)
    loc = re.sub(r"_+", "_", loc).strip("_")[:120]
    if loc:
        return f"{stem}__{loc}.json"
    return f"{stem}__{cid[:16]}.json"


def hadith_system_extra(text: str) -> str:
    tokens = [token for token, _ in split_hadith_page(text)]
    if not tokens:
        return (
            "No numbered start was detected; this page is likely a continuation. "
            "Return hadiths with one continuation item (or more if the page still "
            "contains distinct narrations)."
        )
    listed = ", ".join(tokens)
    return (
        f"Detected numbered starts on this page: {listed}. "
        f"The hadiths array MUST include each of these {len(tokens)} numbered "
        "narrations (plus a leading continuation item if the page starts mid-hadith). "
        "Each numbered item is a SEPARATE claim. semantic_nodes for item N may use "
        "only the matn of item N. Do not copy nodes from a neighbour or from a "
        "kitab/bab heading."
    )


def schema_for(pipeline: str) -> type[BaseModel]:
    if pipeline == "hadith":
        return HadithPageExtraction
    if pipeline == "tafsir":
        return TafsirExtraction
    if pipeline == "history":
        return HistoryExtraction
    raise ValueError(f"Unknown pipeline {pipeline}")


def system_prompt(pipeline: str, extra: str = "") -> str:
    if pipeline == "hadith":
        return hadith_prompt(extra)
    if pipeline == "tafsir":
        return TAFSIR_PROMPT
    return HISTORY_PROMPT


UNIFY_AR_HEAD = 4000
UNIFY_BODY_BUDGET = 20_000



def extract_hadith_page(agent: GeminiAgent, unit: ParsedUnit) -> dict:
    text = strip_folklib_footnotes(unit.text)
    extra = hadith_system_extra(text)
    result = agent.complete_structured(
        f"Locator: {unit.locator}\n\nText:\n{text}",
        HadithPageExtraction,
        system=system_prompt("hadith", extra=extra),
    )
    return remap_hadith_payload(result.model_dump())


_INCOMPLETE_TRANSLATION = re.compile(
    r"(?:\.\.\.|…|ادامه\s+روایت|ادامه\s+سفارش|Continuation\s+of|"
    r"\[\s*بهتر\s+از\s*\]|"
    r"narrated\s*\.\.\.|روایت\s*کرده‌اند\s*که\s*\.\.\.)",
    re.IGNORECASE,
)


def translation_incomplete(text: str) -> bool:
    """Page fragments often land as summaries ending in `...` / ادامه."""
    value = str(text or "").strip()
    if not value:
        return True
    return bool(_INCOMPLETE_TRANSLATION.search(value))


# Short coordinated nouns (الخير و الشر و الايمان …). Sermons use و too, but
# the conjuncts are clauses, not 1–4-word labels.
_MAX_LIST_CONJUNCT_CHARS = 40
_MAX_LIST_CONJUNCT_WORDS = 4
_ENCYCLOPEDIC_SHORT_ITEMS = 12
_ENCYCLOPEDIC_SHORT_ITEMS_SPANNED = 8


def _short_coordinated_items(folded: str) -> int:
    """How many و-conjuncts look like inventory labels, not clauses."""
    n = 0
    for part in folded.split(" و "):
        chunk = part.strip()
        if not chunk:
            continue
        words = chunk.split()
        if 1 <= len(words) <= _MAX_LIST_CONJUNCT_WORDS and len(chunk) <= _MAX_LIST_CONJUNCT_CHARS:
            n += 1
    return n


def looks_encyclopedic(payload: dict) -> bool:
    """Backup detector when the page LLM never set is_encyclopedic.

    Structural only: a dense run of short و-linked labels. Book titles and
    particular concepts are not consulted — any inventory hadith qualifies.
    """
    if payload.get("is_encyclopedic"):
        return True
    from src.pipelines.morphology import fold

    matn = str(payload.get("hadith") or "")
    folded = fold(matn)
    if not folded.strip():
        return False
    short = _short_coordinated_items(folded)
    pages = [p for p in (payload.get("arabic_pages") or []) if str(p).strip()]
    n_pages = len(pages)
    if n_pages == 0 and payload.get("page_start") != payload.get("page_end"):
        n_pages = 2
    if short >= _ENCYCLOPEDIC_SHORT_ITEMS:
        return True
    if n_pages >= 2 and short >= _ENCYCLOPEDIC_SHORT_ITEMS_SPANNED:
        return True
    return False


def _matn_chunks_for_exhaustive(payload: dict) -> list[str]:
    pages = [str(p).strip() for p in (payload.get("arabic_pages") or []) if str(p).strip()]
    if pages:
        return pages
    arabic = str(payload.get("hadith") or "")
    if not arabic:
        return []
    if len(arabic) <= UNIFY_BODY_BUDGET:
        return [arabic]
    step = 6000
    return [arabic[i : i + step] for i in range(0, len(arabic), step)]


def _exhaustive_mentions_fill(agent: GeminiAgent, payload: dict, label: str) -> list[dict]:
    """Chunked + multi-round MentionsFill for encyclopedic inventories.

    Uses the proven MentionsFill schema (max 12) in rounds so Gemini structured
    output stays valid while we accumulate dozens–hundreds of list items.
    """
    from src.agents.errors import ProviderServerError, StructuredOutputError
    from src.pipelines.grounding import ground_mentions

    chunks = _matn_chunks_for_exhaustive(payload)
    if not chunks:
        chunks = [str(payload.get("hadith") or "")]
    merged = list(payload.get("mentions") or [])
    system = mentions_fill_prompt(is_exhaustive=True)
    max_rounds = 10
    for i, chunk in enumerate(chunks):
        stagnant = 0
        for round_i in range(max_rounds):
            known = [str(m.get("text") or "") for m in merged if m.get("text")]
            known_preview = ", ".join(known[-40:]) if known else "(none yet)"
            prompt = (
                f"Arabic hadith fragment {i + 1}/{len(chunks)}, "
                f"extraction round {round_i + 1}.\n"
                f"Already extracted (do NOT repeat these texts): {known_preview}\n\n"
                f"{chunk}\n\n"
                "Return the NEXT batch of distinct enumerated items still missing "
                "from the list above. Prefer up to 12 new mentions this round."
            )
            try:
                filled = agent.complete_structured(
                    prompt,
                    MentionsFill,
                    system=system,
                )
            except (StructuredOutputError, ProviderServerError) as exc:
                raise StructuredOutputError(
                    f"exhaustive mentions fill failed for {label} "
                    f"(fragment {i + 1}/{len(chunks)} round {round_i + 1}): {exc}"
                ) from exc
            before = len(merged)
            merged = _merge_mentions(
                merged, [m.model_dump() for m in filled.mentions]
            )
            gained = len(merged) - before
            logger.info(
                "exhaustive MentionsFill %s fragment %s/%s round %s -> +%s (total %s)",
                label,
                i + 1,
                len(chunks),
                round_i + 1,
                gained,
                len(merged),
            )
            if gained == 0:
                stagnant += 1
                if stagnant >= 2:
                    break
            else:
                stagnant = 0
    grounded, dropped = ground_mentions(
        merged,
        str(payload.get("hadith") or ""),
        payload.get("ravis"),
        quotes=payload.get("quotes"),
    )
    if dropped:
        logger.info(
            "exhaustive MentionsFill %s grounded drop %s",
            label,
            len(dropped),
        )
    return grounded


def needs_enrichment(payload: dict) -> bool:
    if looks_encyclopedic(payload):
        return True
    if payload.get("page_start") != payload.get("page_end"):
        return True
    if translation_incomplete(str(payload.get("hadith_fa") or "")):
        return True
    if translation_incomplete(str(payload.get("hadith_en") or "")):
        return True
    # Either channel counts. Keying on semantic_nodes alone made every payload
    # extracted under the mention contract look hollow, so every one of them
    # went to unify -- a second model call per hadith, for nothing.
    if len(payload.get("mentions") or []) + len(semantic_nodes_of(payload)) < 2:
        return True
    if not payload.get("ravis"):
        return True
    return False


def unify_assembled_hadith(agent: GeminiAgent, payload: dict) -> dict:
    """Fill/repair FA/EN/mentions/ravis; always refresh ravis on multi-page.

    Truncated page translations (`...` / ادامه) and multi-page stubs are
    replaced by unify's full translation of the assembled Arabic. When topics
    are missing, MentionsFill runs first. Soft unify then fills the rest.
    When topics stay missing after MentionsFill, failure is raised so the
    runner marks ERROR instead of writing hollow PROCESSED files.

    Encyclopedic inventories (`is_encyclopedic` / heuristic) use chunked
    MentionsFill so page prefer-3-8 does not permanently truncate long lists.
    """
    from src.agents.errors import ProviderServerError, StructuredOutputError

    payload = dict(payload)
    payload["hadith"] = strip_folklib_footnotes(str(payload.get("hadith") or ""))
    if looks_encyclopedic(payload):
        payload["is_encyclopedic"] = True
    if not needs_enrichment(payload):
        return remap_hadith_payload(payload)
    arabic = str(payload.get("hadith") or "")
    head = arabic[:UNIFY_AR_HEAD]
    if len(arabic) <= UNIFY_BODY_BUDGET:
        body = arabic
    else:
        body = arabic[:UNIFY_BODY_BUDGET]

    is_exhaustive = bool(payload.get("is_encyclopedic"))

    topic_count = len(payload.get("mentions") or []) + len(semantic_nodes_of(payload))
    need_topics = topic_count < 1 or is_exhaustive

    label = payload.get("marker") or payload.get("locator")

    if need_topics:
        try:
            if is_exhaustive:
                payload["mentions"] = _exhaustive_mentions_fill(agent, payload, str(label))
            else:
                filled = agent.complete_structured(
                    f"Arabic hadith:\n{body}",
                    MentionsFill,
                    system=mentions_fill_prompt(is_exhaustive=False),
                )
                payload["mentions"] = _merge_mentions(
                    payload.get("mentions"),
                    [m.model_dump() for m in filled.mentions],
                )
        except (StructuredOutputError, ProviderServerError) as exc:
            raise StructuredOutputError(
                f"mentions fill failed for {label}: {exc}"
            ) from exc
        payload = remap_hadith_payload(payload)
        if not (payload.get("mentions") or []):
            raise StructuredOutputError(
                f"mentions fill returned topics that grounding dropped for {label}"
            )
        if not needs_enrichment(payload):
            return payload

    # Topics present (or MentionsFill already ran): soft unify for FA/EN/ravis.
    # Do not pass is_exhaustive here — exhaustiveness is MentionsFill's job.
    schema = HadithUnify
    system = unify_prompt(require_topics=False, is_exhaustive=False)
    try:
        result = agent.complete_structured(
            f"Isnad / opening:\n{head}\n\nMatn:\n{body}",
            schema,
            system=system,
        )
    except (StructuredOutputError, ProviderServerError) as exc:
        if need_topics and not (payload.get("mentions") or []):
            raise StructuredOutputError(
                f"unify could not extract mentions for {label}: {exc}"
            ) from exc
        logger.warning(
            "unify failed for %s; persisting assembled payload: %s",
            label,
            exc,
        )
        return remap_hadith_payload(payload)
    out = dict(payload)
    if result.ravis:
        out["ravis"] = list(result.ravis)
    if result.mentions and not is_exhaustive:
        # Encyclopedic lists are already filled exhaustively; soft unify must
        # not collapse them by merging a short re-summarization.
        out["mentions"] = _merge_mentions(
            payload.get("mentions"), [m.model_dump() for m in result.mentions]
        )
    if result.quotes:
        out["quotes"] = _merge_quotes(
            payload.get("quotes"), [q.model_dump() for q in result.quotes]
        )
    existing = semantic_nodes_of(payload)
    if result.semantic_nodes and len(existing) < 2:
        out["semantic_nodes"] = enforce_node_policy(
            [n.model_dump() for n in result.semantic_nodes],
            out.get("ravis"),
        )
    # Page extract often leaves truncated FA/EN (`...` / ادامه). Multi-page
    # assemble joins those fragments; unify must replace them with a full
    # translation of the assembled Arabic — never keep the stub.
    multi_page = payload.get("page_start") != payload.get("page_end")
    if result.hadith_fa and (
        multi_page
        or translation_incomplete(str(payload.get("hadith_fa") or ""))
    ):
        out["hadith_fa"] = result.hadith_fa
    if result.hadith_en and (
        multi_page
        or translation_incomplete(str(payload.get("hadith_en") or ""))
    ):
        out["hadith_en"] = result.hadith_en
    out = remap_hadith_payload(out)
    if need_topics and not (out.get("mentions") or []):
        raise StructuredOutputError(
            f"unify returned mentions that grounding dropped for {label}"
        )
    return out


def _merge_mentions(existing, extra) -> list[dict]:
    """Union by (text, type), keeping the better salience and any real evidence."""
    merged: dict[tuple[str, str], dict] = {}
    for item in list(existing or []) + list(extra or []):
        if not isinstance(item, dict):
            continue
        key = (str(item.get("text") or "").strip(), str(item.get("type") or "concept"))
        if not key[0]:
            continue
        current = merged.get(key)
        if current is None:
            merged[key] = dict(item)
            continue
        if float(item.get("salience") or 0) > float(current.get("salience") or 0):
            current["salience"] = item.get("salience")
        from src.pipelines.grounding import prefer_evidence_span

        current["evidence"] = prefer_evidence_span(
            key[0],
            str(current.get("evidence") or ""),
            str(item.get("evidence") or ""),
        )
    return list(merged.values())


def _merge_quotes(existing, extra) -> list[dict]:
    merged: dict[str, dict] = {}
    for item in list(existing or []) + list(extra or []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text and text not in merged:
            merged[text] = dict(item)
    return list(merged.values())


def persist_complete_hadith(
    state: StateManager,
    *,
    book_id: str,
    source_path: str,
    payload: dict,
    output_dir: Path,
    status: ChunkStatus = ChunkStatus.PROCESSED_PHASE1,
    error: str | None = None,
    force: bool = False,
) -> str:
    """Write one complete narration and upsert it as an embeddable Phase 1 chunk.

    When `force` is true (targeted --page debug), always rewrite the JSON and
    refresh the stored payload. Otherwise skip only if the chunk is already
    processed *and* the output file is still on disk — so deleting JSON for
    debugging still recreates files on the next run.
    """
    payload["hadith"] = strip_folklib_footnotes(str(payload.get("hadith") or ""))
    locator = str(payload.get("locator") or "")
    marker = str(payload.get("marker") or "")
    arabic = str(payload.get("hadith") or "")
    cid = chunk_id(book_id, f"{marker}|{locator}", arabic)
    dest = output_dir / book_id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / phase1_filename(source_path, locator, cid, marker=marker)
    existing = state.get_chunk(cid)
    if (
        not force
        and existing
        and existing.status not in {ChunkStatus.PENDING, ChunkStatus.ERROR}
        and status == ChunkStatus.PROCESSED_PHASE1
        and path.exists()
    ):
        return cid
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if not existing:
        state.upsert_chunks(
            [
                {
                    "id": cid,
                    "book_id": book_id,
                    "pipeline": "hadith",
                    "locator": locator,
                    "source_path": source_path,
                    "text": arabic,
                }
            ]
        )
    state.mark(cid, status, payload=payload, error=error)
    logger.info(
        "phase1 hadith %s %s %s",
        status.value,
        locator,
        marker,
    )
    return cid


def process_unit(
    agent: GeminiAgent,
    state: StateManager,
    *,
    book_id: str,
    pipeline: str,
    unit: ParsedUnit,
    output_dir: Path,
    min_chars: int,
) -> ChunkStatus:
    cid = chunk_id(book_id, unit.locator, unit.text)
    existing = state.get_chunk(cid)
    if existing and existing.status not in {ChunkStatus.PENDING, ChunkStatus.ERROR}:
        return existing.status

    if should_skip(unit.text, min_chars):
        state.mark(cid, ChunkStatus.SKIPPED, error="too short or table of contents")
        return ChunkStatus.SKIPPED

    extra = hadith_system_extra(unit.text) if pipeline == "hadith" else ""
    result = agent.complete_structured(
        f"Locator: {unit.locator}\n\nText:\n{unit.text}",
        schema_for(pipeline),
        system=system_prompt(pipeline, extra=extra),
    )
    payload = result.model_dump()
    if pipeline == "hadith":
        payload = remap_hadith_payload(payload)
    dest = output_dir / book_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / phase1_filename(unit.source_path, unit.locator, cid)).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    state.mark(cid, ChunkStatus.PROCESSED_PHASE1, payload=payload)
    logger.info("phase1 %s %s %s", book_id, pipeline, unit.locator)
    return ChunkStatus.PROCESSED_PHASE1
