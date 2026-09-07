"""Concept and entity catalogs: canonicalize semantic_nodes into graph IDs."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from config.paths import DERIVED_ALIASES_YAML, DERIVED_ONTOLOGY_YAML, ENTITIES_YAML, ONTOLOGY_YAML

logger = logging.getLogger(__name__)

_ALEF = re.compile(r"[إأآٱ]")
_WS = re.compile(r"\s+")
_AYAH_RE = re.compile(r"^\d{1,3}:\d{1,3}$")

# U+0670 (superscript alef) is a LETTER written as a mark: ٱلْأَبْصَٰرِ is الأبصار.
# It must become an alef, not be deleted, or vocalized Qur'anic spellings stop
# matching their plain-text twins. Everything else in these ranges is a vowel
# sign, a Qur'anic recitation mark, or tatweel, and is dropped outright.
_SUPERSCRIPT_ALEF = "ٰ"
_DIACRITICS = re.compile(r"[ً-ٟۖ-ۭـ]")

# Honorifics attach to the SAME entity under many spellings, so they must go
# before any comparison: the gazetteer stores أبو عبد الله while an isnad prints
# أَبُو عَبْدِ اللَّهِ (ع).
_HONORIFIC = re.compile(
    r"\(\s*(?:ع|ص|عج|س|ره|عليه السلام)\s*\)"
    r"|(?:عليه|عليها|عليهم|عليهما)\s+السلام"
    r"|صلى\s+الله\s+عليه\s+و?\s*آله?(?:\s+و\s*سلم)?"
    r"|رضوان\s+الله\s+عليه"
    r"|(?<![؀-ۿ])(?:ع|ص|عج)(?![؀-ۿ])"
)

# Kunyas decline: أبي عبد الله (genitive) and أبا عبد الله (accusative) are the
# same man as أبو عبد الله, which is the form the gazetteer indexes. Without this
# hadith 3 stored a raw أبي عبد الله while hadith 6 resolved to الإمام الصادق.
_KUNYA = re.compile(r"^(?:ابي|ابا)(?=\s)")

# Whether the typesetter set the conjunction close or loose is not a difference
# in the word: the corpus prints both `القضايا والأحكام` and `القضاء و الأحكام`,
# and they were two concepts. Folding it here fixes the catalog's own duplicates
# and, more importantly, lets a mention match whichever way it was written.
_LOOSE_WAW = re.compile(r"\s+و\s+")

# Descriptive heads. The closed vocabulary makes these unreachable for concepts
# and groups, but person/place/event/work stay open, so they still need the
# guard: without it "فضيلة محمد" passes as a person node.
BANNED_HEAD = (
    "أهمية",
    "فضيلة",
    "فضل",
    "تعريف",
    "حقيقة",
    "ماهية",
    "بيان",
    "كيفية",
    "آثار",
    "ثمرات",
    "علة",
    "أسباب",
    "ارتباط",
    "مراتب",
)
CONCEPT_TYPE = "concept"
ENTITY_TYPES = ("person", "place", "group", "event", "work")
NODE_TYPES = (CONCEPT_TYPE, *ENTITY_TYPES, "ayah")
MAX_NODES = 6
MAX_PRIMARY = 2

_YAML_TYPE_KEYS = (
    ("persons", "person"),
    ("places", "place"),
    ("groups", "group"),
    ("events", "event"),
    ("works", "work"),
)


@dataclass(frozen=True)
class Concept:
    id: str
    pref: str
    aliases: tuple[str, ...]
    broader: tuple[str, ...]
    group: bool


@dataclass(frozen=True)
class Entity:
    id: str
    pref: str
    aliases: tuple[str, ...]
    type: str
    ambiguous: bool


def normalize_ar(text: str) -> str:
    """Fold one Arabic label to its comparison key.

    Order matters: the superscript alef becomes a real alef before the rest of
    the marks are stripped, honorifics go before the kunya fold (so the trailing
    (ع) cannot block it), and the leading ال is shed last.
    """
    s = (text or "").replace(_SUPERSCRIPT_ALEF, "ا")
    s = _DIACRITICS.sub("", s)
    s = _HONORIFIC.sub(" ", s)
    s = _WS.sub(" ", s.strip())
    s = _ALEF.sub("ا", s)
    s = s.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
    s = _KUNYA.sub("ابو", s)
    s = _LOOSE_WAW.sub(" و", s)
    if s.startswith("ال") and len(s) > 3:
        s = s[2:]
    return s.strip()


def _parse_concept(raw) -> Concept | None:
    if isinstance(raw, str):
        label = raw.strip()
        if not label:
            return None
        return Concept(id=label, pref=label, aliases=(), broader=(), group=False)
    if not isinstance(raw, dict):
        return None
    pref = str(raw.get("pref") or raw.get("id") or "").strip()
    cid = str(raw.get("id") or pref).strip()
    if not pref:
        return None
    aliases = tuple(str(a).strip() for a in (raw.get("aliases") or []) if str(a).strip())
    # A concept may sit under more than one parent: الحساب belongs to both
    # الجزاء الأخروي and القيامة, and forcing a single parent would have meant
    # trading one real link away for the other.
    raw_broader = raw.get("broader")
    if raw_broader is None:
        parents: tuple[str, ...] = ()
    elif isinstance(raw_broader, (list, tuple)):
        parents = tuple(str(b).strip() for b in raw_broader if str(b).strip())
    else:
        parents = tuple(p for p in (str(raw_broader).strip(),) if p)
    group = bool(raw.get("group", False))
    return Concept(id=cid, pref=pref, aliases=aliases, broader=parents, group=group)


def _read_concepts(path: Path) -> list:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("concepts") or []


@lru_cache(maxsize=1)
def load_concept_catalog() -> tuple[Concept, ...]:
    """Hand-written catalog first, then the harvested one.

    Two files on purpose. `base_ontology.yaml` holds judgement calls nothing can
    derive -- قتل النفس and الانتحار share no root, no wording and no chapter, so
    only a human joins them. `derived_ontology.yaml` holds thousands of terms
    mined from the books' own headings and is regenerated wholesale.

    Base is loaded first and wins on conflict, so re-harvesting can never
    overrule a curation decision.
    """
    out: list[Concept] = []
    seen: set[str] = set()
    for path in (ONTOLOGY_YAML, DERIVED_ONTOLOGY_YAML):
        for raw in _read_concepts(path):
            concept = _parse_concept(raw)
            if concept is None or concept.id in seen:
                continue
            folded = normalize_ar(concept.pref)
            if any(folded == normalize_ar(c.pref) for c in out):
                continue
            seen.add(concept.id)
            out.append(concept)
    return _with_derived_aliases(tuple(out))


def _with_derived_aliases(concepts: tuple[Concept, ...]) -> tuple[Concept, ...]:
    """Attach the generated aliases, which live in their own file.

    Not in `derived_ontology.yaml`, because the harvest rewrites that wholesale
    and would drop them; not in `base_ontology.yaml`, because they are generated
    and that file is the human's. Hand-written aliases already on a concept are
    kept -- these are added to them, never over them.
    """
    extra = _read_aliases(DERIVED_ALIASES_YAML)
    if not extra:
        return concepts
    out = []
    for concept in concepts:
        more = extra.get(normalize_ar(concept.pref), ())
        if not more:
            out.append(concept)
            continue
        have = {normalize_ar(a) for a in concept.aliases}
        add = tuple(a for a in more if normalize_ar(a) not in have)
        out.append(
            Concept(
                id=concept.id,
                pref=concept.pref,
                aliases=concept.aliases + add,
                broader=concept.broader,
                group=concept.group,
            )
        )
    return tuple(out)


def _read_aliases(path) -> dict[str, tuple[str, ...]]:
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("unreadable alias file %s: %s", path, exc)
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for row in raw.get("aliases") or []:
        if not isinstance(row, dict):
            continue
        pref = str(row.get("pref") or "").strip()
        forms = tuple(str(a).strip() for a in (row.get("aliases") or []) if str(a).strip())
        if pref and forms:
            out[normalize_ar(pref)] = forms
    return out


def load_ontology() -> list[str]:
    """All concept prefs (catalog still re-exported by pipelines.catalog)."""
    return [c.pref for c in load_concept_catalog()]


@lru_cache(maxsize=1)
def _concept_index() -> dict[str, Concept]:
    """Folded surface form -> concept. Cached: this is rebuilt per lookup
    otherwise, and with a harvested catalog of ~1,900 terms that turned every
    `decompose()` call into thousands of dict insertions. The harvest went from
    ten minutes to seconds."""
    table: dict[str, Concept] = {}
    for concept in load_concept_catalog():
        keys = (concept.id, concept.pref, *concept.aliases)
        for key in keys:
            table[normalize_ar(key)] = concept
    return table


def clear_catalog_caches() -> None:
    """Drop every cached view of the catalogs, after a file on disk changed."""
    load_concept_catalog.cache_clear()
    load_entity_catalog.cache_clear()
    _concept_index.cache_clear()
    _entity_index.cache_clear()


def lookup_concept(label: str) -> Concept | None:
    folded = normalize_ar(label)
    if not folded:
        return None
    return _concept_index().get(folded)


def broader_chain(label: str) -> list[str]:
    """Every ancestor pref above `label`, nearest first, without `label` itself.

    Cycle-guarded, because the catalog is hand-edited YAML and a loop here would
    hang bucket construction rather than fail loudly.
    """
    hit = lookup_concept(label)
    if hit is None:
        return []
    chain: list[str] = []
    seen: set[str] = {normalize_ar(hit.pref)}
    queue = list(hit.broader)
    while queue:
        parent = queue.pop(0)
        folded = normalize_ar(parent)
        if not folded or folded in seen:
            continue
        seen.add(folded)
        parent_hit = lookup_concept(parent)
        pref = parent_hit.pref if parent_hit else parent
        chain.append(pref)
        if parent_hit:
            queue.extend(parent_hit.broader)
    return chain


def vocabulary_block(max_chars: int = 8000) -> str:
    """The selectable vocabulary, rendered for the prompt.

    Showing the model the actual terms turns invention into selection. That is
    the whole point: a term it cannot see, it cannot choose, so عقل المرء and
    اجتهاد المجتهدين stop being possible outputs without any rule naming them.
    It also fixes the reverse failure -- محبة أهل البيت was in the catalog all
    along, and recognising it on a list is far easier than generating it from a
    matn that only says مَحَبَّةٌ.

    At the current catalog size the whole list is a few hundred characters. When
    it outgrows `max_chars`, this is where retrieval by matn similarity replaces
    the full dump; the prompt shape does not change.
    """
    concepts = [c.pref for c in load_concept_catalog()]
    groups = [e.pref for e in load_entity_catalog() if e.type == "group"]
    lines = ["CONCEPTS: " + " · ".join(concepts)]
    if groups:
        lines.append("GROUPS: " + " · ".join(groups))
    block = "\n".join(lines)
    if len(block) > max_chars:
        logger.warning(
            "vocabulary is %d chars, over the %d budget; switch to retrieval",
            len(block),
            max_chars,
        )
        block = block[:max_chars]
    return block


def canonicalize_concept(label: str) -> str:
    """Map a raw label to catalog pref, or return the stripped original."""
    stripped = (label or "").strip()
    if not stripped:
        return ""
    hit = lookup_concept(stripped)
    return hit.pref if hit else stripped


def _parse_entity(raw, node_type: str) -> Entity | None:
    if not isinstance(raw, dict):
        return None
    pref = str(raw.get("pref") or raw.get("id") or "").strip()
    eid = str(raw.get("id") or pref).strip()
    if not pref:
        return None
    aliases = tuple(str(a).strip() for a in (raw.get("aliases") or []) if str(a).strip())
    return Entity(
        id=eid,
        pref=pref,
        aliases=aliases,
        type=node_type,
        ambiguous=bool(raw.get("ambiguous", False)),
    )


@lru_cache(maxsize=1)
def load_entity_catalog() -> tuple[Entity, ...]:
    path: Path = ENTITIES_YAML
    if not path.exists():
        return ()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: list[Entity] = []
    seen: set[tuple[str, str]] = set()
    for key, node_type in _YAML_TYPE_KEYS:
        for raw in data.get(key) or []:
            entity = _parse_entity(raw, node_type)
            if entity is None or (entity.type, entity.id) in seen:
                continue
            seen.add((entity.type, entity.id))
            out.append(entity)
    return tuple(out)


@lru_cache(maxsize=1)
def _entity_index() -> dict[tuple[str, str], Entity]:
    table: dict[tuple[str, str], Entity] = {}
    for entity in load_entity_catalog():
        keys = (entity.id, entity.pref, *entity.aliases)
        for key in keys:
            folded = normalize_ar(key)
            if folded:
                table[(entity.type, folded)] = entity
    return table


def lookup_entity(label: str, node_type: str) -> Entity | None:
    folded = normalize_ar(label)
    if not folded:
        return None
    return _entity_index().get((node_type, folded))


# Types drawn from a closed vocabulary. A concept or a group is an index term:
# it only earns a place in the graph if the catalog already knows it, because a
# term nothing else uses indexes nothing. Anything else the model wants to say
# goes to proposed_nodes and is decided by corpus frequency, not by a rule here.
#
# person/place/event/work stay open. Those are open classes grounded literally
# in the matn -- a gazetteer of every narrator-adjacent name in 36,000 hadiths
# is not something we can enumerate up front.
CLOSED_VOCABULARY_TYPES = (CONCEPT_TYPE, "group")


def repair_to_vocabulary(label: str) -> str | None:
    """Map an off-vocabulary phrase onto the catalog term it is a variant of.

    عقل المرء، حساب العباد، كمال العقل and قدر العقول are the corpus talking about
    العقل and الحساب with this sentence's grammar attached. The modifier never
    decides the topic, so the first constituent that names a catalog concept
    wins.

    Returns None when nothing in the phrase is a known term -- اجتهاد المجتهدين
    and عتاب الله have no vocabulary anchor, and that is exactly the signal that
    they belong in proposed_nodes instead of the graph.
    """
    from src.pipelines.resolver import _MEASURE_WORDS

    words = [w for w in (label or "").split(" ") if w]
    if len(words) < 2:
        return None
    for word in words:
        # Skip measure words for the same reason `decompose` does: `قدر` inside
        # `قدر العقول` is "in proportion to", not the topic, and the harvested
        # catalog now contains it as a term in its own right.
        if normalize_ar(word) in _MEASURE_WORDS:
            continue
        hit = lookup_concept(word)
        if hit:
            logger.info("repair %s -> %s", label, hit.pref)
            return hit.pref
    return None


def resolve_node(raw: str, node_type: str, strict: bool = True) -> str | None:
    """Canonicalize one label, or None if it does not belong in the graph.

    `strict` closes the vocabulary for concepts and groups. It is on for the
    hadith pipeline, where unknown terms are routed to proposals instead. The
    tafsir and history pipelines still pass strict=False: they have no proposal
    channel yet, and silently emptying them would be worse than letting their
    free-text concepts through.
    """
    s = _WS.sub(" ", str(raw or "").strip())
    if not s:
        return None
    if node_type == "ayah":
        return s if _AYAH_RE.match(s) else None
    hit: Concept | Entity | None
    if node_type == CONCEPT_TYPE:
        hit = lookup_concept(s)
    else:
        hit = lookup_entity(s, node_type)
    if hit:
        return hit.pref
    # Shape is checked before repair: الشيطنة والنكراء names two topics, and
    # repairing it would silently keep one and discard the other.
    words = s.split(" ")
    shaped = 1 <= len(words) <= 3 and not any(
        w == "و" or w.startswith("و") for w in words[1:]
    )
    if strict and node_type in CLOSED_VOCABULARY_TYPES:
        # Last chance: map onto the vocabulary rather than drop. This is repair,
        # not invention -- it can only ever return a term already in the catalog,
        # so it cannot fragment anything. It rescues عقل المرء -> العقل when the
        # model slips, and re-canonicalizes payloads extracted before the
        # vocabulary was closed.
        repaired = repair_to_vocabulary(s) if shaped else None
        if repaired is None:
            logger.info("not in vocabulary: %s type=%s", s, node_type)
        return repaired
    if not shaped or words[0] in BANNED_HEAD:
        return None
    return s


# Node types that can collide with an isnad. A place or a work never can.
_SPEAKER_TYPES = ("person", "group")


def narrator_identities(ravis) -> set[str]:
    """Every folded key by which this hadith's own narrators can be recognised.

    Both the printed surface form and the gazetteer pref go in, because the two
    sides of the comparison rarely look alike: hadith 6 emitted the node
    الإمام الصادق while its isnad printed أَبُو عَبْدِ اللَّهِ (ع). Folding alone
    does not bridge that -- only resolving both through the gazetteer does.
    """
    out: set[str] = set()
    for raw in ravis or []:
        label = str(raw or "")
        folded = normalize_ar(label)
        if not folded:
            continue
        out.add(folded)
        for node_type in _SPEAKER_TYPES:
            hit = lookup_entity(label, node_type)
            if hit:
                out.add(normalize_ar(hit.pref))
    return out


def _is_speaker(label: str, resolved: str, node_type: str, narrators: set[str]) -> bool:
    """True when this node is just the narrator/speaker, not what the matn is about.

    The Imam being quoted is not a subject of his own narration; keeping him as
    a node makes every hadith he narrates collide in the graph. Entities the
    matn genuinely talks about -- معاوية in hadith 3, آدم and جبرئيل in hadith 2
    -- are absent from their own isnads and survive this untouched.
    """
    if node_type not in _SPEAKER_TYPES or not narrators:
        return False
    return normalize_ar(label) in narrators or normalize_ar(resolved) in narrators


def bucket_eligible(node_type: str, role: str) -> bool:
    if node_type == CONCEPT_TYPE:
        return True
    if node_type == "ayah":
        return False
    if node_type in ENTITY_TYPES:
        return role == "primary"
    return False


def _as_node_dict(raw) -> dict | None:
    if isinstance(raw, dict):
        node = str(raw.get("node") or "").strip()
        ntype = str(raw.get("type") or CONCEPT_TYPE).strip() or CONCEPT_TYPE
        role = str(raw.get("role") or "secondary").strip() or "secondary"
        if role not in {"primary", "secondary"}:
            role = "secondary"
        return {"node": node, "type": ntype, "role": role}
    text = str(raw or "").strip()
    if not text:
        return None
    return {"node": text, "type": CONCEPT_TYPE, "role": "primary"}


def semantic_nodes_of(data: dict) -> list[dict]:
    """Read semantic_nodes; upcast legacy concept_nodes / tags strings."""
    raw = data.get("semantic_nodes")
    if isinstance(raw, list) and raw:
        out: list[dict] = []
        for item in raw:
            parsed = _as_node_dict(item)
            if parsed:
                out.append(parsed)
        if out:
            return out
    legacy = data.get("concept_nodes")
    if legacy is None:
        legacy = data.get("tags")
    if not isinstance(legacy, list):
        return []
    out = []
    for item in legacy:
        parsed = _as_node_dict(item)
        if parsed:
            out.append(parsed)
    return out


def _catalog_type_override(label: str, node_type: str) -> str:
    """Force a label the concept catalog curates back to type concept.

    `resolve_node` only consults the concept catalog when the model already said
    "concept", so a mistyped label silently became a second, unlinked identity:
    hadith 6 emitted place:الجنة and hadith 7 emitted event:يوم القيامة, neither
    of which can ever merge with concept:الجنة or concept:القيامة no matter how
    many aliases the catalog grows.
    """
    if node_type == CONCEPT_TYPE:
        # The reverse mistake: named beings emitted as ideas. الشيطان as a
        # concept is a bucket key every waswasa hadith would pile into; as a
        # person it only reaches the graph when the matn is really about him.
        for entity_type in ENTITY_TYPES:
            hit = lookup_entity(label, entity_type)
            if hit:
                logger.info("retype %s from concept to %s (gazetteer)", label, entity_type)
                return entity_type
        return node_type
    if node_type not in ENTITY_TYPES:
        return node_type
    if lookup_concept(label) is None:
        return node_type
    logger.info("retype %s from %s to concept (catalog)", label, node_type)
    return CONCEPT_TYPE


def _prefer_concepts_as_primary(cleaned: list[dict]) -> None:
    """Trade an entity out of a primary slot for a curated concept sitting at secondary.

    Hadith 3 is the case: the model made معاوية primary and left النكراء -- the
    thing the matn is actually defining -- secondary. Demoting overflow primaries
    cannot fix that, because there were only two to begin with; the concept has
    to be promoted.

    Assignment is a single ranked pass over every node, NOT a promote step
    followed by a truncate step. Splitting the two left an ordering hole: when
    the model marked معاوية, العقل and النكراء all primary, there were no
    secondaries to promote from, so the overflow cut simply kept the first two in
    list order and demoted النكراء -- the exact failure this is supposed to
    prevent, reappearing whenever the model happened to over-mark primaries.

    Concepts win primary slots outright. An entity holds one only when the
    narration produced no concept at all, because primary is what drives Phase 2
    bucketing and a person-keyed bucket groups nothing worth comparing.
    """
    if not cleaned:
        return

    def rank(node: dict) -> tuple[int, int]:
        # Curated concepts first, then bare concepts, then entities; the model's
        # own primary marking breaks ties inside a rank.
        if node["type"] == CONCEPT_TYPE:
            kind = 0 if lookup_concept(node["node"]) is not None else 1
        else:
            kind = 2
        return kind, 0 if node["role"] == "primary" else 1

    order = sorted(range(len(cleaned)), key=lambda i: (*rank(cleaned[i]), i))
    concepts = [i for i in order if cleaned[i]["type"] == CONCEPT_TYPE]
    if concepts:
        winners = set(concepts[:MAX_PRIMARY])
    else:
        # Entity-only narration: keep whatever the model called primary.
        marked = [i for i in order if cleaned[i]["role"] == "primary"]
        winners = set((marked or order)[:MAX_PRIMARY])
    for index, node in enumerate(cleaned):
        node["role"] = "primary" if index in winners else "secondary"


def enforce_node_policy(
    nodes: list[dict],
    ravis: list[str] | None = None,
    strict: bool = True,
    collect_repairs: list[str] | None = None,
) -> list[dict]:
    """Canonicalize, shape-gate, dedupe, cap primaries and total. Never raises.

    `ravis` is this hadith's own isnad. When supplied, person/group nodes that
    are merely the chain or the speaker are dropped -- see `_is_speaker`.

    `collect_repairs` receives the ORIGINAL wording of every off-vocabulary label
    that had to be repaired onto a catalog term. Without it the proposals loop
    has a blind spot: if the model writes خلق العقل into semantic_nodes rather
    than proposed_nodes, repair silently collapses it to العقل and the term can
    never accumulate the frequency that would promote it. Routing the original
    into proposals means promotion no longer depends on the model putting the
    term in the right field.
    """
    narrators = narrator_identities(ravis)
    cleaned: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for raw in nodes or []:
        parsed = _as_node_dict(raw)
        if parsed is None:
            continue
        ntype = parsed["type"]
        if ntype not in NODE_TYPES:
            logger.info("drop unknown semantic_node type %s", ntype)
            continue
        ntype = _catalog_type_override(parsed["node"], ntype)
        resolved = resolve_node(parsed["node"], ntype, strict=strict)
        if not resolved:
            logger.info("drop semantic_node %s type=%s", parsed["node"], ntype)
            if collect_repairs is not None and ntype in CLOSED_VOCABULARY_TYPES:
                # Dropped for being off-vocabulary: still a candidate, and the
                # corpus decides whether it recurs.
                collect_repairs.append(parsed["node"])
            continue
        if (
            collect_repairs is not None
            and ntype in CLOSED_VOCABULARY_TYPES
            and normalize_ar(resolved) != normalize_ar(parsed["node"])
            and lookup_concept(parsed["node"]) is None
        ):
            collect_repairs.append(parsed["node"])
        if _is_speaker(parsed["node"], resolved, ntype, narrators):
            logger.info("drop narrator-as-node %s type=%s", resolved, ntype)
            continue
        key = (resolved, ntype)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"node": resolved, "type": ntype, "role": parsed["role"]})
    # Trim before assigning roles: dropping a node after the fact could remove
    # every primary and leave the narration with none.
    if len(cleaned) > MAX_NODES:
        cleaned = cleaned[:MAX_NODES]
    _prefer_concepts_as_primary(cleaned)
    return cleaned


def remap_hadith_payload(payload: dict) -> dict:
    """Return a copy with semantic_nodes passed through the alias table and gate."""
    data = dict(payload)
    items = data.get("hadiths")
    if isinstance(items, list):
        remapped_items = []
        for item in items:
            if isinstance(item, dict):
                row = dict(item)
                # Deliberately NOT grounded here. This branch runs on a per-PAGE
                # extract, where `hadith` is only the fragment printed on this
                # page; a mention whose evidence sits on the next page would be
                # rejected before it ever reached the accumulator, and
                # `assemble()` could not recover it. Grounding happens once, in
                # `assemble()`, against the whole narration.
                repairs: list[str] = []
                nodes = enforce_node_policy(
                    semantic_nodes_of(row), row.get("ravis"), collect_repairs=repairs
                )
                row.pop("tags", None)
                row.pop("concept_nodes", None)
                row["semantic_nodes"] = nodes
                row["proposed_nodes"] = _merge_proposals(row.get("proposed_nodes"), repairs)
                remapped_items.append(row)
            else:
                remapped_items.append(item)
        data["hadiths"] = remapped_items
    # Top-level branch: this payload IS one assembled narration (the unify
    # path), so `hadith` is the complete matn and grounding is safe here.
    _ground_row(data)
    repairs: list[str] = []
    nodes = enforce_node_policy(
        semantic_nodes_of(data), data.get("ravis"), collect_repairs=repairs
    )
    data.pop("tags", None)
    data.pop("concept_nodes", None)
    data["semantic_nodes"] = nodes
    data["proposed_nodes"] = _merge_proposals(data.get("proposed_nodes"), repairs)
    return data


def _ground_row(row: dict) -> None:
    """Drop mentions this narration's own text does not support.

    Imported lazily: grounding pulls in the stemmer, and ontology is imported by
    nearly everything.
    """
    mentions = row.get("mentions")
    if not isinstance(mentions, list) or not mentions:
        return
    from src.pipelines.grounding import ground_mentions

    kept, rejected = ground_mentions(
        mentions,
        str(row.get("hadith") or ""),
        row.get("ravis"),
        quotes=row.get("quotes"),
    )
    if rejected:
        logger.info(
            "dropped %d ungrounded mention(s): %s",
            len(rejected),
            "; ".join(f"{text} ({reason})" for text, reason in rejected),
        )
    row["mentions"] = kept


def _merge_proposals(existing, extra: list[str]) -> list[str]:
    """Union of what the model proposed and what the gate had to repair or drop."""
    out: list[str] = []
    seen: set[str] = set()
    for item in list(existing or []) + list(extra or []):
        term = str(item or "").strip()
        folded = normalize_ar(term)
        if not folded or folded in seen:
            continue
        seen.add(folded)
        out.append(term)
    return out
