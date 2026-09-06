"""Propose and rank hadith pairs for the phase-2 relation judgement.

Phase 2 asks a model whether two narrations support, contradict or ignore each
other. That is O(pairs), so the only question that matters here is which pairs
are worth paying for.

The shape is standard record linkage: block cheaply for recall, score
deterministically, then spend the model on the top of the ranking. The
important consequence is that no single signal is load-bearing. Earlier designs
made semantic nodes the sole mechanism, so a label the extractor got wrong
deleted a relation outright. Here a missed node still leaves the shared bab, the
shared verse, the shared narrator and textual similarity, and the pair simply
ranks lower instead of disappearing.

Blocking skips signals that are too common to be evidence. A node carried by
2,000 narrations proposes two million pairs and says almost nothing about any of
them; the same node still contributes to the score of pairs raised by something
else. Rarity is the whole signal, which is why every weight here is IDF-based.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# A key carried by more than this share of the corpus is useless for blocking.
MAX_BLOCK_DF_RATIO = 0.02
# ...and never block on a key with more members than this, however large the corpus.
MAX_BLOCK_MEMBERS = 400
# The ratio only means anything once there is a corpus to take a ratio of. On a
# 20-document trial run every key looks extreme, and applying it there returns
# no candidates at all -- which reads as "the linker is broken" rather than
# "the sample is small".
MIN_DOCS_FOR_RATIO = 200

WEIGHTS = {
    "node": 1.0,
    "ayah": 1.4,   # two narrations citing one verse are talking about it
    "bab": 0.8,    # the author put them together on purpose
    "kitab": 0.15,
    "ravi": 0.3,
    "vector": 1.2,
}


@dataclass
class Document:
    """One narration as the linker sees it."""

    doc_id: str
    nodes: list[str] = field(default_factory=list)
    ayahs: list[str] = field(default_factory=list)
    ravis: list[str] = field(default_factory=list)
    bab: str = ""
    kitab: str = ""


@dataclass
class Candidate:
    left: str
    right: str
    score: float
    reasons: dict[str, float] = field(default_factory=dict)
    # The signal that proposed this pair, as opposed to the signals that merely
    # scored it once proposed. Only this one is evidence the pair was found.
    blocked_by: str = ""

    @property
    def pair(self) -> tuple[str, str]:
        return self.left, self.right


def _postings(docs: list[Document]) -> dict[str, list[str]]:
    """Every key to the documents carrying it, namespaced by signal."""
    index: dict[str, list[str]] = defaultdict(list)
    for doc in docs:
        for node in set(doc.nodes):
            index[f"node::{node}"].append(doc.doc_id)
        for ayah in set(doc.ayahs):
            index[f"ayah::{ayah}"].append(doc.doc_id)
        for ravi in set(doc.ravis):
            index[f"ravi::{ravi}"].append(doc.doc_id)
        if doc.bab:
            index[f"bab::{doc.bab}"].append(doc.doc_id)
        if doc.kitab:
            index[f"kitab::{doc.kitab}"].append(doc.doc_id)
    return index


def idf_table(docs: list[Document]) -> dict[str, float]:
    """Inverse document frequency per key.

    This is what replaces the old primary/secondary flag. Two narrations sharing
    الانتحار is strong evidence; sharing العقل inside كتاب العقل is almost none.
    A binary role cannot express that difference; a number can.
    """
    total = max(len(docs), 1)
    return {
        key: math.log(1 + total / len(set(members)))
        for key, members in _postings(docs).items()
    }


def _blocking_keys(
    index: dict[str, list[str]],
    total: int,
    max_ratio: float,
    max_members: int,
    min_docs_for_ratio: int = MIN_DOCS_FOR_RATIO,
) -> list[str]:
    keys: list[str] = []
    for key, members in index.items():
        size = len(set(members))
        if size < 2:
            continue
        if key.startswith("kitab::"):
            # Never block on kitab. Al-Kafi's kitabs run to 1,607 narrations,
            # which is 1.3M pairs from one key, and being in the same kitab is
            # far too weak a reason to compare two hadiths.
            continue
        too_big = size > max_members
        too_common = total >= min_docs_for_ratio and size / total > max_ratio
        if too_big or too_common:
            logger.debug("skip blocking key %s (%d docs)", key, size)
            continue
        keys.append(key)
    return keys


def generate(
    docs: list[Document],
    *,
    max_block_df_ratio: float = MAX_BLOCK_DF_RATIO,
    max_block_members: int = MAX_BLOCK_MEMBERS,
    min_docs_for_ratio: int = MIN_DOCS_FOR_RATIO,
    vectors: dict[str, list[float]] | None = None,
    min_score: float = 0.0,
    limit: int | None = None,
) -> list[Candidate]:
    """Rank candidate pairs, best first.

    `vectors` is optional; when supplied, cosine similarity is added as one more
    scoring signal. It is deliberately not a blocking signal here -- an all-pairs
    cosine over the corpus is quadratic, and the cheap keys already provide the
    recall.
    """
    index = _postings(docs)
    idf = idf_table(docs)
    by_id = {doc.doc_id: doc for doc in docs}
    total = max(len(docs), 1)

    # Remember which signal actually PROPOSED each pair. Nearly every bab-raised
    # pair also picks up a kitab score, so counting scoring signals as "raised
    # by" made kitab look like the dominant source when it never blocks at all.
    proposed: dict[tuple[str, str], str] = {}
    for key in _blocking_keys(
        index, total, max_block_df_ratio, max_block_members, min_docs_for_ratio
    ):
        signal = key.split("::", 1)[0]
        members = sorted(set(index[key]))
        for left, right in itertools.combinations(members, 2):
            proposed.setdefault((left, right), signal)

    logger.info("blocking proposed %d pairs from %d documents", len(proposed), len(docs))

    scored: list[Candidate] = []
    for (left_id, right_id), blocked_by in proposed.items():
        left, right = by_id[left_id], by_id[right_id]
        reasons: dict[str, float] = {}

        for label, weight, left_items, right_items in (
            ("node", WEIGHTS["node"], left.nodes, right.nodes),
            ("ayah", WEIGHTS["ayah"], left.ayahs, right.ayahs),
            ("ravi", WEIGHTS["ravi"], left.ravis, right.ravis),
        ):
            shared = set(left_items) & set(right_items)
            if not shared:
                continue
            namespace = "node" if label == "node" else label
            reasons[label] = weight * sum(
                idf.get(f"{namespace}::{item}", 0.0) for item in shared
            )

        if left.bab and left.bab == right.bab:
            reasons["bab"] = WEIGHTS["bab"] * idf.get(f"bab::{left.bab}", 0.0)
        if left.kitab and left.kitab == right.kitab:
            reasons["kitab"] = WEIGHTS["kitab"] * idf.get(f"kitab::{left.kitab}", 0.0)

        if vectors and left_id in vectors and right_id in vectors:
            similarity = _cosine(vectors[left_id], vectors[right_id])
            if similarity > 0:
                reasons["vector"] = WEIGHTS["vector"] * similarity

        score = sum(reasons.values())
        if score >= min_score:
            scored.append(Candidate(left_id, right_id, score, reasons, blocked_by))

    scored.sort(key=lambda c: (-c.score, c.left, c.right))
    return scored[:limit] if limit else scored


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def summarise(candidates: list[Candidate], docs: list[Document]) -> str:
    """A budget report: how many pairs, and what raised them."""
    if not candidates:
        return "no candidate pairs"
    by_block: dict[str, int] = defaultdict(int)
    by_reason: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        by_block[candidate.blocked_by or "?"] += 1
        for reason in candidate.reasons:
            by_reason[reason] += 1
    possible = len(docs) * (len(docs) - 1) // 2
    share = 100.0 * len(candidates) / possible if possible else 0.0
    lines = [
        f"documents        {len(docs)}",
        f"all-pairs        {possible}",
        f"candidates       {len(candidates)}  ({share:.2f}% of all-pairs)",
        f"score range      {candidates[-1].score:.2f} .. {candidates[0].score:.2f}",
        "proposed by:",
    ]
    for signal, count in sorted(by_block.items(), key=lambda kv: -kv[1]):
        lines.append(f"   {signal:<8} {count}")
    # Kept separate on purpose: almost every bab-proposed pair also picks up a
    # kitab score, so reporting scoring signals as the source made kitab look
    # like the dominant blocker when it never blocks at all.
    lines.append("also scored on (not why the pair was found):")
    for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        lines.append(f"   {reason:<8} {count}")
    return "\n".join(lines)
