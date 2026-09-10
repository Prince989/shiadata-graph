"""Check mentions against the matn they claim to come from.

The prompt tells the model that a person, place, event, group or work must
appear literally in the narration, and that `evidence` must be the words it took
the mention from. Nothing enforced either, which meant `evidence` was decoration
for a human auditor rather than a guard.

It is a guard here. Comparison is on folded text, because the matn is vocalised
and the model's echo of it usually is not.
"""

from __future__ import annotations

import logging
import re

from src.pipelines.morphology import fold, root, root_signature

logger = logging.getLogger(__name__)

# Types the prompt requires to be literally present. A concept may be inferred
# from what the matn asserts, so it is exempt from the text check -- but its
# evidence span, when offered, still has to be real.
GROUNDED_TYPES = ("person", "place", "group", "event", "work")

# Below this an evidence span is too short to be a quotation of anything, so it
# is treated as absent rather than as a claim to verify.
_MIN_EVIDENCE_CHARS = 6

# Whether a mention with no usable evidence at all is dropped. Default on: the
# claim "evidence makes every mention checkable" is only true if a missing span
# fails rather than skips, and the prompt asks for one on every mention.
# Turn it off for a first live run if the model proves unwilling to supply them.
REQUIRE_EVIDENCE = True


def _contains(haystack: str, needle: str) -> bool:
    folded_needle = fold(needle).replace(" ", "")
    if not folded_needle:
        return False
    return folded_needle in fold(haystack).replace(" ", "")


def matn_body(matn: str) -> str:
    """Speech after the first قال / يقول — not the isnad prefix."""
    folded = fold(matn)
    cut = folded.find(" قال ")
    if cut < 0:
        cut = folded.find(" يقول ")
    if cut < 0:
        return folded
    return folded[cut + 1 :]


def evidence_supports_mention(text: str, evidence: str) -> bool:
    """True when the span is about this mention, not a random isnad fragment."""
    folded_text = fold(text)
    folded_ev = fold(evidence)
    if not folded_text or not folded_ev:
        return False
    compact_ev = folded_ev.replace(" ", "")
    if folded_text.replace(" ", "") and folded_text.replace(" ", "") in compact_ev:
        return True
    for word in folded_text.split():
        if len(word) >= 3 and word.replace(" ", "") in compact_ev:
            return True
    text_roots = {r for r in root_signature(text) if r and len(r) >= 2}
    ev_roots = {root(w) for w in folded_ev.split() if w and root(w)}
    return bool(text_roots & ev_roots)


def evidence_usable(mention: dict, matn: str) -> bool:
    """Evidence counts only if it is in the post-قال body.

    In-body spans may infer a concept without repeating the mention label.
    Isnad names never qualify.
    """
    evidence = str(mention.get("evidence") or "").strip()
    if not fold(evidence).strip():
        return False
    return _contains(matn_body(matn), evidence)


def prefer_evidence_span(
    text: str, current: str, other: str, matn: str | None = None
) -> str:
    """Pick the span that actually supports `text`; optionally prefer matn body."""
    cur = str(current or "").strip()
    alt = str(other or "").strip()
    if not alt:
        return cur
    if not cur:
        return alt
    body = matn_body(matn) if matn else ""
    cur_ok = evidence_supports_mention(text, cur)
    alt_ok = evidence_supports_mention(text, alt)
    if matn:
        cur_ok = cur_ok and _contains(body or matn, cur)
        alt_ok = alt_ok and _contains(body or matn, alt)
    if alt_ok and not cur_ok:
        return alt
    return cur


def check_mention(
    mention: dict, matn: str, require_evidence: bool = REQUIRE_EVIDENCE
) -> str | None:
    """Return a reason string when this mention is not supported by the matn."""
    text = str(mention.get("text") or "").strip()
    if not text:
        return "empty"
    node_type = str(mention.get("type") or "concept")
    body = matn_body(matn)
    usable = evidence_usable(mention, matn)

    if not usable and require_evidence and not _contains(matn, text):
        # No usable span AND the term is not in the text either. A concept may
        # be inferred, but something in the narration has to have prompted it;
        # with neither, the mention rests on nothing at all. Previously a blank
        # evidence field simply skipped the check, which let any invented
        # concept through untouched.
        return "no evidence and term not in matn"

    if node_type in GROUNDED_TYPES and not _contains(body or matn, text):
        # The name itself has to be in the speech, not only the isnad.
        return "entity not in matn"
    return None


def _in_a_quote(text: str, quotes: list | None) -> bool:
    """True when a MULTI-WORD mention repeats wording the hadith is quoting.

    A narration that quotes scripture is not thereby about the scripture's
    phrasing. Hadith 11 quotes «وما يتذكر إلا أولوا الألباب» and the extractor
    turned أولو الألباب into a group node -- a Qur'anic epithet standing beside
    the real topic, العقل, duplicating a link the 2:269 citation edge already
    carries better, since that edge also reaches the tafsir of the verse.

    Single words are deliberately exempt. A hadith about الصبر that quotes a
    verse mentioning الصبر is still about الصبر, and dropping it there would cost
    far more than the occasional stray epithet.
    """
    if len(fold(text).split()) < 2:
        return False
    needle = _quote_key(text)
    if not needle:
        return False
    for quote in quotes or []:
        span = quote.get("text") if isinstance(quote, dict) else quote
        if needle in _quote_key(str(span or "")):
            return True
    return False


# A word-final alef after waw is orthographic, not phonemic -- the ألف فارقة of
# قالوا / كتبوا. The quote writes أُولُوا where the mention writes أولو, and
# without folding it away no letter comparison between the two can ever match.
_OTIOSE_ALEF = re.compile(r"وا(?=\s|$)")


def _quote_key(text: str) -> str:
    return _OTIOSE_ALEF.sub("و", fold(text)).replace(" ", "")


def ground_mentions(
    mentions: list,
    matn: str,
    ravis: list[str] | None = None,
    require_evidence: bool = REQUIRE_EVIDENCE,
    quotes: list | None = None,
) -> tuple[list, list[tuple[str, str]]]:
    """Split mentions into (kept, [(text, reason), ...]).

    Ungrounded mentions are dropped rather than repaired: a mention whose
    evidence is not in the text is not evidence of anything.

    Narrators are dropped here too. Under the old per-hadith gate that check
    lived in `enforce_node_policy`; the mention contract took extraction off
    that path, so without this the Imam being quoted becomes a topic again --
    the first bug this project ever fixed, quietly reintroduced.
    """
    from src.pipelines.ontology import lookup_entity, narrator_identities, normalize_ar

    narrators = narrator_identities(ravis) if ravis else set()
    kept: list = []
    rejected: list[tuple[str, str]] = []
    for mention in mentions or []:
        if not isinstance(mention, dict):
            continue
        text = str(mention.get("text") or "")
        node_type = str(mention.get("type") or "concept")
        if narrators:
            # Checked for EVERY type, not just person/group. The model
            # occasionally types a name as a concept, and gating the filter on
            # the declared type let `concept:أبو عبد الله` past -- where the
            # resolver's catalog lookup then retypes it to a person anyway,
            # putting the speaker back in the graph by the back door.
            #
            # Compare canonical identities, not surface forms: an isnad printing
            # أبو عبد الله and a mention saying جعفر بن محمد are one man, and
            # only gazetteer resolution bridges those two strings.
            forms = {normalize_ar(text)}
            for candidate_type in ("person", "group", node_type):
                entity = lookup_entity(text, candidate_type)
                if entity:
                    forms.add(normalize_ar(entity.pref))
            if forms & narrators:
                rejected.append((text, "narrator, not a subject"))
                logger.info("drop narrator-as-mention %r", text)
                continue
        if _in_a_quote(text, quotes):
            rejected.append((text, "quoted scripture, not a topic"))
            logger.info("drop quoted-phrase mention %r", text)
            continue
        reason = check_mention(mention, matn, require_evidence) if matn else None
        if reason:
            rejected.append((text, reason))
            logger.info("drop ungrounded mention %r: %s", text, reason)
            continue
        row = dict(mention)
        if matn and not evidence_usable(row, matn):
            row["evidence"] = ""
        kept.append(row)
    return kept, rejected
