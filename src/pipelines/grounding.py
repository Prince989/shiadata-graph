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

from src.pipelines.morphology import fold

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


def check_mention(
    mention: dict, matn: str, require_evidence: bool = REQUIRE_EVIDENCE
) -> str | None:
    """Return a reason string when this mention is not supported by the matn."""
    text = str(mention.get("text") or "").strip()
    if not text:
        return "empty"
    node_type = str(mention.get("type") or "concept")
    evidence = str(mention.get("evidence") or "").strip()
    usable = len(fold(evidence)) >= _MIN_EVIDENCE_CHARS

    if usable:
        if not _contains(matn, evidence):
            return "evidence not in matn"
    elif require_evidence and not _contains(matn, text):
        # No usable span AND the term is not in the text either. A concept may
        # be inferred, but something in the narration has to have prompted it;
        # with neither, the mention rests on nothing at all. Previously a blank
        # evidence field simply skipped the check, which let any invented
        # concept through untouched.
        return "no evidence and term not in matn"

    if node_type in GROUNDED_TYPES and not _contains(matn, text):
        # The name itself has to be on the page. An inferred person is a
        # hallucinated person.
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
        kept.append(mention)
    return kept, rejected
