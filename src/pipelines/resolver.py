"""Turn corpus-wide mentions into canonical graph nodes.

This is the inversion the whole design turns on. The vocabulary used to be an
INPUT to extraction -- a list the model had to choose from, which could not
contain معاوية or the thousands of other names in these books, and which made a
human the bottleneck for every new topic. Here it is an OUTPUT: mentions are
collected from the whole corpus and clustered into identities afterwards, when
all 15,000 narrations can be seen at once.

Clustering runs on three signals, cheapest first:

  1. curated seeds -- the catalog and gazetteer, demoted from gate to exception
     table. They express what clustering cannot discover: قتل النفس and الانتحار
     share no root and no wording, so only a human-written alias joins them.
  2. root signature -- Arabic derivational morphology. الحساب، حساب العباد،
     يحاسب and محاسبة all key on حسب without anyone listing them.
  3. compound recurrence -- a multi-word mention keeps its own identity only if
     it recurs as a compound. خلق العقل appears across parallel narrations and
     becomes a node under العقل; عقل المرء appears once and folds into العقل.

Point 3 is what no hand-written rule could do. خلق العقل and عقل المرء are the
same shape -- noun plus genitive -- and only the corpus separates them.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from src.pipelines.morphology import head_root, is_tautology, root_signature
from src.pipelines.ontology import (
    CONCEPT_TYPE,
    lookup_concept,
    lookup_entity,
    normalize_ar,
)

logger = logging.getLogger(__name__)

# A compound must recur across this many distinct documents to keep an identity
# of its own rather than folding into its head. Scales with corpus size: two
# co-occurrences out of 30 documents is evidence, two out of 15,000 is a
# coincidence, and a fixed floor would promote a long tail of accidents on a
# full run.
COMPOUND_MIN_DF = 2
COMPOUND_DF_PER_10K = 3


def compound_threshold(doc_count: int, base: int = COMPOUND_MIN_DF) -> int:
    """Minimum df for a compound to keep its own identity, given corpus size."""
    return max(base, round(doc_count * COMPOUND_DF_PER_10K / 10_000))


# Below this, a short entity name is too likely to be a different person to
# absorb into a longer one. زرارة appearing twice is the short form of one man;
# أبو محمد appearing three hundred times is several.
ENTITY_MERGE_MAX_DF = 12


@dataclass
class Mention:
    """A raw observation, before identity is decided."""

    text: str
    type: str
    doc_id: str
    salience: float = 0.5
    evidence: str = ""


@dataclass
class Node:
    """A resolved identity: one graph node and every surface form that reached it."""

    key: str
    label: str
    type: str
    surfaces: set[str] = field(default_factory=set)
    docs: set[str] = field(default_factory=set)
    parent: str | None = None
    curated: bool = False

    @property
    def df(self) -> int:
        return len(self.docs)


_ENTITY_TYPES = ("person", "place", "group", "event", "work")


def _seed_key(text: str, node_type: str) -> tuple[str, str, str] | None:
    """Identity from the curated catalog or gazetteer, if it knows this form.

    The lookup crosses types deliberately. A mistyped mention is otherwise a
    second, unlinkable identity for something the graph already has: the model
    emits `place:الجنة` and `event:يوم القيامة` where the catalog holds
    `concept:الجنة` and `concept:القيامة`, and `concept:الشيطان` where the
    gazetteer holds a person. The old per-hadith gate corrected both directions;
    when extraction moved to mentions that correction was left behind, which
    quietly restored a bug already paid for.
    """
    # The DECLARED type is tried first, so a string listed in both catalogs
    # keeps the reading the model chose. Only when the declared type knows
    # nothing does the search widen -- which is what corrects `place:الجنة` and
    # `concept:الشيطان` without overruling a correct call on a dual-listed name.
    if node_type in _ENTITY_TYPES:
        entity = lookup_entity(text, node_type)
        if entity:
            return f"{node_type}:{normalize_ar(entity.pref)}", entity.pref, node_type

    hit = lookup_concept(text)
    if hit:
        return f"concept:{normalize_ar(hit.pref)}", hit.pref, CONCEPT_TYPE

    for candidate_type in _ENTITY_TYPES:
        if candidate_type == node_type:
            continue
        entity = lookup_entity(text, candidate_type)
        if entity:
            return (
                f"{candidate_type}:{normalize_ar(entity.pref)}",
                entity.pref,
                candidate_type,
            )
    return None


# Measure and relational words. Inside a construct these are grammar -- `على قدر
# العقل` means "in proportion to the intellect", and قدر contributes nothing --
# but several are also catalog terms in their own right (القدر, divine decree),
# and lookup folds away the article that would have told them apart. So they are
# skipped as CONSTITUENTS only; a whole label that is one of them still resolves
# normally. A closed grammatical class, so it does not grow with the corpus.
_MEASURE_WORDS = frozenset(
    normalize_ar(w)
    for w in (
        "قدر", "مقدار", "حد", "حدود", "عدد", "نحو", "مثل", "بعض", "سائر",
        "جملة", "وجه", "باب", "أبواب", "كتاب", "وقت", "أوقات", "كيفية",
        "صفة", "صفات", "معنى", "أنواع", "نوع", "قسم", "أقسام",
    )
)


def decompose(text: str) -> list[tuple[str, str]]:
    """Every catalog concept named by a constituent of a multi-word label.

    Replaces splitting on و, which was operating on a token rather than on
    meaning and corrupted real words: `ولاة العدل` lost its و and became
    `لاة العدل`. Here the label's structure is irrelevant. Each word is simply
    looked up, and `في` / `على` / `قدر` contribute nothing because they name
    nothing -- no rule has to mention them.

        الوسواس في الوضوء والصلاة  ->  الوسواس + الوضوء + الصلاة
        الجزاء على قدر العقل        ->  الجزاء + العقل

    Concepts only. A bare word inside a phrase must never hit the gazetteer:
    `على` folds to `علي` and became the Imam, and `الحجة` in الحجة الباطنة --
    the intellect as God's inner proof -- collapsed onto الإمام المهدي, taking
    الحجة الظاهرة with it. Entities need the whole label to be safe.
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for word in (text or "").split():
        if normalize_ar(word) in _MEASURE_WORDS:
            continue
        for candidate in _clitic_forms(word):
            hit = lookup_concept(candidate)
            if hit is None:
                continue
            key = f"concept:{normalize_ar(hit.pref)}"
            if key not in seen:
                seen.add(key)
                found.append((key, hit.pref))
            break
    return found


def _clitic_forms(word: str) -> tuple[str, ...]:
    """The word, and the word minus a leading conjunction.

    Self-validating rather than guessed: the caller only accepts a stripped form
    if it actually resolves, so `والصلاة` yields الصلاة while `ولاة` yields
    nothing and keeps its و. No length heuristic, which is what mangled ولاة.
    """
    if len(word) > 1 and word.startswith("و"):
        return (word, word[1:])
    return (word,)


def _topic_parent(text: str, node_type: str) -> tuple[str, str, bool, str] | None:
    """Key of the constituent that carries the topic of a compound.

    Not simply the head. Arabic iḍāfa puts the topic on either side: in
    عقل المرء it is the first word, in كمال العقل and قدر العقول it is the
    second. So the rule is the first constituent the catalog recognises, which
    lands all three on العقل. When no constituent is known, fall back to the
    head's root -- and if that head never appears on its own either, the caller
    drops the compound rather than keeping a singleton nobody can reach.
    """
    for word in (text or "").split():
        seed = _seed_key(word, node_type)
        if seed:
            key, label, parent_type = seed
            # A concept must never anchor to an entity. Allowing it was tried on
            # real data and collapsed إكمال الحجة, الحجة الباطنة and الحجة الظاهرة
            # -- three distinct kalam concepts -- onto الإمام المهدي, because
            # الحجة is one of his gazetteer aliases.
            if node_type == CONCEPT_TYPE and parent_type != CONCEPT_TYPE:
                continue
            return key, label, True, parent_type
    head = head_root(text)
    return (f"{node_type}:@{head}", text, False, node_type) if head else None


def _morph_key(
    text: str, node_type: str
) -> tuple[str, tuple[str, str, bool, str] | None]:
    """Identity from morphology: single words key on their root, compounds on all of them."""
    signature = [r for r in root_signature(text) if r]
    if not signature:
        return f"{node_type}:{normalize_ar(text)}", None
    if len(signature) == 1:
        return f"{node_type}:@{signature[0]}", None
    return f"{node_type}:@{'+'.join(signature)}", _topic_parent(text, node_type)


# Generic heads that precede a name without being part of it.
_ENTITY_HEADS = ("واقعة", "وقعة", "يوم", "غزوة", "معركة", "سرية", "مدينة", "بلاد", "أرض", "كتاب")
_NASAB = (" بن ", " ابن ", " بنت ")


def _name_tokens(text: str) -> tuple[str, ...]:
    """Normalised name tokens, with the genealogy chain cut off.

    زرارة and زرارة بن أعين are one man; the nasab chain identifies him further
    rather than naming someone else. Cutting at بن lets the short form and the
    long form meet, and the full string survives as a surface.
    """
    name = normalize_ar(text)
    for head in _ENTITY_HEADS:
        folded_head = normalize_ar(head)
        if name.startswith(folded_head + " "):
            name = name[len(folded_head) + 1 :]
            break
    for particle in _NASAB:
        cut = name.find(normalize_ar(particle).strip().join((" ", " ")))
        if cut > 0:
            name = name[:cut]
            break
    return tuple(t for t in name.split() if t)


def _merge_entity_prefixes(
    nodes: dict[str, Node],
    max_short_df: int = ENTITY_MERGE_MAX_DF,
    redirect: dict[str, str] | None = None,
) -> None:
    """Join entity nodes where one name is a prefix of another.

    Conservative on purpose. Prefix containment catches the real pattern -- a
    figure named in full once and by short name elsewhere -- without asserting
    that every أبو محمد is the same أبو محمد. Genuinely ambiguous kunyas are
    what the gazetteer's `ambiguous` flag is for.
    """
    by_type: dict[str, list[Node]] = defaultdict(list)
    for node in nodes.values():
        if node.type != CONCEPT_TYPE:
            by_type[node.type].append(node)

    for group in by_type.values():
        # Longest names first, so short forms attach to the fullest available.
        ranked = sorted(
            group, key=lambda n: (-len(_name_tokens(n.label)), n.key)
        )
        absorbed: set[str] = set()
        for i, target in enumerate(ranked):
            if target.key in absorbed:
                continue
            target_tokens = _name_tokens(target.label)
            if not target_tokens:
                continue
            for other in ranked[i + 1 :]:
                if other.key in absorbed or other.key == target.key:
                    continue
                other_tokens = _name_tokens(other.label)
                # Equal-length cores must merge, not just shorter ones: زرارة and
                # زرارة بن أعين both reduce to one token once the nasab chain is
                # cut, as do صفين and واقعة صفين once the generic head is.
                if not other_tokens or len(other_tokens) > len(target_tokens):
                    continue
                if target_tokens[: len(other_tokens)] != other_tokens:
                    continue
                # A short form used constantly is not one person referred to
                # briefly -- it is a kunya several people share. أبو محمد would
                # otherwise chain الرازي to العسكري through a single bare
                # mention. Frequency is the cheap discriminator; a curated
                # gazetteer entry overrides it either way.
                if (
                    len(other_tokens) < len(target_tokens)
                    and not other.curated
                    and other.df > max_short_df
                ):
                    logger.debug(
                        "refuse merge: %s is too common (df=%d) to be a short form",
                        other.label,
                        other.df,
                    )
                    continue
                target.surfaces.update(other.surfaces)
                target.docs.update(other.docs)
                if other.curated:
                    target.curated = True
                    target.label = other.label
                absorbed.add(other.key)
                if redirect is not None:
                    redirect[other.key] = target.key
                logger.debug("merge entity %s into %s", other.label, target.label)
        for key in absorbed:
            nodes.pop(key, None)


def _pick_entity_label(surfaces: set[str]) -> str:
    """Longest surface form. For a person the fuller name identifies better."""
    return sorted(surfaces, key=lambda s: (-len(s), s))[0]


def _pick_label(surfaces: set[str]) -> str:
    """Shortest surface form, ties broken alphabetically for determinism.

    Shortest is a good proxy for the citation form: العقل beats عقل المرء and
    كمال العقل, because the modifiers are what the individual sentence added.
    """
    return sorted(surfaces, key=lambda s: (len(s), s))[0]


def resolve_with_assignments(
    mentions: list[Mention],
    compound_min_df: int | None = None,
    entity_merge_max_df: int = ENTITY_MERGE_MAX_DF,
) -> tuple[dict[str, Node], list[list[str]]]:
    """Cluster mentions into nodes, and say which node each mention landed on.

    The assignment list is produced HERE rather than re-derived afterwards from
    a (type, surface) index. Re-derivation was silently wrong: resolution
    retypes a mention (`place:الجنة` becomes `concept:الجنة`), so a later lookup
    keyed on the mention's ORIGINAL type missed the node entirely and the
    narration was written back with no nodes at all. The node table looked
    right -- correct label, correct df -- while the payload that Phase 2 and the
    export actually read had lost the mention.

    Keys move twice more after the first pass, when a rare compound folds into
    its parent and when an entity absorbs a shorter name, so a redirect table is
    threaded through both and followed at the end.
    """
    nodes, redirect, initial = _cluster(
        mentions, compound_min_df, entity_merge_max_df
    )
    assignments: list[list[str]] = []
    for keys in initial:
        resolved: list[str] = []
        for key in keys:
            final = _follow(key, redirect, nodes)
            if final and final not in resolved:
                resolved.append(final)
        assignments.append(resolved)
    return nodes, assignments


def _follow(
    key: str | None, redirect: dict[str, str], nodes: dict[str, Node]
) -> str | None:
    """Chase a key through the fold/merge redirects to where it ended up."""
    seen: set[str] = set()
    while key and key not in nodes and key in redirect and key not in seen:
        seen.add(key)
        key = redirect[key]
    return key if key in nodes else None


def resolve(
    mentions: list[Mention],
    compound_min_df: int | None = None,
    entity_merge_max_df: int = ENTITY_MERGE_MAX_DF,
) -> dict[str, Node]:
    """Cluster mentions into nodes. Deterministic, no model, no network."""
    return resolve_with_assignments(mentions, compound_min_df, entity_merge_max_df)[0]


def _cluster(
    mentions: list[Mention],
    compound_min_df: int | None = None,
    entity_merge_max_df: int = ENTITY_MERGE_MAX_DF,
) -> tuple[dict[str, Node], dict[str, str], list[str | None]]:
    """The clustering itself. Returns (nodes, redirects, per-mention initial key)."""
    if compound_min_df is None:
        compound_min_df = compound_threshold(len({m.doc_id for m in mentions}))
    nodes: dict[str, Node] = {}
    compound_children: dict[str, tuple[str, str, bool, str]] = {}
    redirect: dict[str, str] = {}
    initial: list[list[str]] = []

    for mention in mentions:
        text = (mention.text or "").strip()
        if not text:
            initial.append([])
            continue

        keys: list[str] = []
        # A multi-word concept the catalog does not know as a whole also names
        # its constituents. The mention reaches all of them directly, so nothing
        # is lost when the compound itself turns out to be a one-off; and when
        # the compound recurs it survives alongside them, giving the narration
        # both the specific topic and the general ones.
        constituents: list[tuple[str, str]] = []
        if (
            mention.type == CONCEPT_TYPE
            and len(text.split()) > 1
            and not _seed_key(text, mention.type)
        ):
            constituents = decompose(text)
            for part_key, part_label in constituents:
                node = nodes.setdefault(
                    part_key,
                    Node(key=part_key, label=part_label, type=CONCEPT_TYPE, curated=True),
                )
                node.surfaces.add(part_label)
                node.docs.add(mention.doc_id)
                if part_key not in keys:
                    keys.append(part_key)

        for part in [text]:
            part = part.strip()
            if not part:
                continue
            # A phrase that repeats one root says the same thing twice; it
            # describes this sentence and can never be shared.
            if is_tautology(part):
                logger.debug("drop tautology %s", part)
                continue
            seed = _seed_key(part, mention.type)
            if seed:
                key, label, node_type = seed
                if node_type != mention.type:
                    logger.info(
                        "retype %s from %s to %s (catalog)", part, mention.type, node_type
                    )
                node = nodes.setdefault(
                    key, Node(key=key, label=label, type=node_type, curated=True)
                )
            else:
                if mention.type == CONCEPT_TYPE:
                    key, parent = _morph_key(part, mention.type)
                    if constituents:
                        # Decomposition already gave the mention its parents, so
                        # the compound only has to justify its OWN existence.
                        first_key, first_label = constituents[0]
                        parent = (first_key, first_label, True, CONCEPT_TYPE)
                    if parent and parent[0] != key:
                        compound_children[key] = parent
                else:
                    # Entities key on the whole name; identity is settled
                    # afterwards by prefix merging, not by folding into a head.
                    key = f"{mention.type}:{normalize_ar(part)}"
                node = nodes.setdefault(
                    key, Node(key=key, label=part, type=mention.type)
                )
            node.surfaces.add(part)
            node.docs.add(mention.doc_id)
            if key not in keys:
                keys.append(key)
        initial.append(keys)

    # A compound that never recurred is this narration's phrasing, not a topic.
    for key, (parent_key, parent_label, parent_curated, parent_type) in compound_children.items():
        node = nodes.get(key)
        if node is None or node.curated:
            continue
        if parent_curated and parent_key not in nodes:
            # The catalog knows this parent even though nothing has mentioned it
            # on its own yet. حساب العباد must still reach الحساب, so create it.
            # The parent's OWN type, never the child's: taking the child's put a
            # node under a person: key while typing it concept.
            nodes[parent_key] = Node(
                key=parent_key, label=parent_label, type=parent_type, curated=True
            )
        if node.df >= compound_min_df:
            # The corpus reached for it more than once, so it is a real topic.
            # It keeps its own identity and hangs under its parent: خلق العقل
            # becomes a child of العقل rather than a rival to it.
            node.parent = parent_key if parent_key in nodes else None
            continue
        parent = nodes.get(parent_key)
        if parent is None:
            # Said once, and built on nothing anyone else uses. عتاب الله is the
            # case: no constituent is a known topic and the phrase never
            # recurred, so there is nothing for it to be reachable by.
            logger.debug("drop unreachable singleton %s", node.label)
            del nodes[key]
            continue
        parent.surfaces.update(node.surfaces)
        parent.docs.update(node.docs)
        logger.debug("fold %s into %s (df=%d)", node.label, parent.label, node.df)
        redirect[key] = parent_key
        del nodes[key]

    _merge_entity_prefixes(nodes, entity_merge_max_df, redirect)

    for node in nodes.values():
        if node.curated or not node.surfaces:
            continue
        node.label = (
            _pick_label(node.surfaces)
            if node.type == CONCEPT_TYPE
            else _pick_entity_label(node.surfaces)
        )
    return nodes, redirect, initial


