# shiadata-graph — session handoff

Context for an assistant picking this up cold. Covers what changed, why, and what
is still open. Project root: `D:\shiadata.dev\shiadata-graph`.

## What the project is

An ETL that turns classical Shi'i hadith books (Folklib-format `.txt`, al-Kafi
vols 1–8 on disk) into a knowledge graph. Phase 1 sends each printed page to
Gemini for structured extraction (Arabic matn, Persian/English translations,
isnad, and `semantic_nodes`). Phase 2 embeds, deduplicates, and classifies
edges. `export-neo4j` writes JSONL + Cypher.

`semantic_nodes` are the graph's topic nodes. They are what makes two hadiths in
two different books reachable from each other.

## The problem that was diagnosed

Measured on a real run of eleven hadiths: **25 of 27 distinct nodes had df=1.**
The only nodes with more than one document were `العقل` (8) and `الدين` (2). The
graph was a star with no other edges.

The cause was architectural, not a prompt bug. The system asked the LLM to
**invent** an index term per hadith, and `resolve_node` accepted any 1–3 word
phrase. The model produced accurate *descriptions* — `عقل المرء` ("a man's
intellect"), `حساب العباد`, `اجتهاد المجتهدين`, `عتاب الله`, `التكليف الإلهي` —
each true of exactly one narration and therefore indexing nothing.

Several rounds of adding rules (stoplists, banned heads, compound reduction)
failed to converge: each run produced a *different* set of bad labels. The
decisive observation is that **whether a term groups anything is a property of
the corpus, not of the string.** `خلق العقل` ("the creation of the intellect", a
real recurring topic) and `عقل المرء` (noise) are grammatically identical — both
a noun plus a genitive. No surface rule can separate them. So the system stopped
predicting df and started measuring it.

## The three layers now in place

### Layer 1 — inherit the book's own classification

`src/extractors/classification.py` (new)

Al-Kafi is already a classified corpus: Kulayni grouped every narration under a
kitab and a bab, and those headings are printed in the source text. The parser
reads them with a running scan over the volume in reading order, and every page
inherits the kitab/bab in force at that point. **2002 headings** are recovered
across the eight volumes on disk (67 / 274 / 303 / 340 / 345 / 418 / 254 / 1).

Why it matters: connectivity no longer depends on extraction quality. All twelve
hadiths of `كتاب العقل و الجهل` share one df=12 node regardless of what the model
emits.

Details that are load-bearing:
- Headings wrap mid-phrase. `بَابُ طِينَةِ` continues as `الْمُؤْمِنِ وَ الْكَافِرِ`
  on the next line, and not always at a conjunction. The join rule relies on
  layout: nothing sits between a heading and its first numbered hadith except
  the rest of the heading.
- Prose beginning with the word *kitab* is rejected (sentence punctuation, length
  cap, and footnote-region exclusion). The preface line
  `كتاب الحجّة و إن لم نكمّله على استحقاقه، لأنّا...` is the case this guards.
- `بَابُ النَّوَادِرِ` ("miscellany") is excluded from bucketing — every volume has
  one and they share no subject.

Emits `kitab` / `bab` on each hadith payload and `IN_KITAB` / `IN_BAB` edges.

**Only `bab` becomes a Phase 2 bucket key.** A kitab spans hundreds of
narrations, so as a comparison bucket it is both useless (pairwise classification
across an entire kitab is noise) and liable to trip `EXTREME_DF_RATIO = 0.15`,
which would silently drop it anyway. So the guarantee is precise: the **exported
graph** is connected through `IN_KITAB` regardless of extraction quality;
**Phase 2 edge classification** groups at bab granularity.

### Layer 2 — the model selects from a closed vocabulary

`vocabulary_block()` in `src/pipelines/ontology.py`; prompt in
`src/pipelines/prompts.py` (new).

The catalog is rendered into the prompt as a menu (currently 75 concepts +
5 groups = **798 characters**). `resolve_node(..., strict=True)` closes
`concept` and `group` to the catalog. `person` / `place` / `event` / `work` stay
open — those are open classes grounded literally in the matn and cannot be
enumerated up front.

This removed the need for most of the accumulated rules. **Deleted**:
`TOO_GENERIC`, `_MUST_RESOLVE_COMPOUNDS`, and the exception lists in the prompt.

One escape hatch survives: `repair_to_vocabulary()` maps an off-vocabulary phrase
onto a catalog term it is built from (`عقل المرء → العقل`, `حساب العباد → الحساب`).
This is repair, not invention — it can only ever output a term already in the
catalog, so it cannot fragment the graph. Shape checks run *before* repair, so
`الشيطنة والنكراء` (two topics joined by و) is dropped rather than half-kept.

`BANNED_HEAD` was **restored** for the open types. The closed vocabulary makes it
unreachable for concepts and groups, but `person` / `place` / `event` / `work`
still need it or `فضيلة محمد` passes as a person node.

Repair feeds proposals: `enforce_node_policy(..., collect_repairs=[...])` records
the *original* wording of anything repaired or dropped for being off-vocabulary,
and `remap_hadith_payload` merges it into that hadith's `proposed_nodes`. Without
this, a model that wrote `خلق العقل` into `semantic_nodes` instead of
`proposed_nodes` would have it silently collapsed to `العقل`, and the term could
never accumulate the frequency that promotes it. **Promotion no longer depends on
the model choosing the right field.**

Also fixes the reverse failure: `محبة أهل البيت` had been in the catalog all
along, and the model kept missing it because *generating* an established
technical term from a matn that only says `مَحَبَّةٌ` is hard. Recognising it on a
list is easy.

### Layer 3 — frequency decides what enters the vocabulary

`src/pipelines/proposals.py` (new), CLI `python main.py proposals [--promote]`.

The model writes anything the vocabulary cannot express into `proposed_nodes`,
a separate field that **never reaches the graph**. Proposals are counted across
distinct hadiths and sorted:

| pile | rule | example |
|---|---|---|
| promote | df ≥ threshold (default 2) | `خلق العقل` df=3 → added to catalog |
| repair | df=1 but anchors to a catalog term | `عقل المرء` → `العقل` |
| drop | df=1, anchored to nothing | `اجتهاد المجتهدين` |

Promotion appends to `config/base_ontology.yaml` as text rather than
re-serialising YAML, because the catalog is hand-maintained and its comments
carry reasoning that a round-trip would erase. Verified to stay parseable.

This is the part that stops the whack-a-mole: nothing in the code knows about
`خلق العقل` in advance.

## Other work in the same session

**Arabic normalisation** (`normalize_ar` in `ontology.py`) — harakat and tatweel
removal, honorific stripping (`(ع)`, `عليه السلام`, `صلى الله عليه وآله`), kunya
i'rab folding (`أبي` / `أبا` → `أبو`), and superscript alef `U+0670` mapped **to**
an alef rather than deleted. That last one is not cosmetic: delete it and the
mushaf's `ٱلْأَبْصَٰرِ` stops matching plain-text `الأبصار` entirely.

**Speaker filter** — person/group nodes that resolve to one of the hadith's own
`ravis` are dropped. The Imam being quoted is not a subject of his own narration.
Comparison is on *canonicalised identity*, not surface form: hadith 6 emitted
`الإمام الصادق` while its isnad printed `أَبُو عَبْدِ اللَّهِ (ع)`, and only
gazetteer resolution bridges those. Participants absent from the isnad (`آدم`,
`جبرئيل`, `معاوية`) survive untouched.

**Type overrides, both directions** — a label the concept catalog curates is
forced to `concept` (`place:الجنة`, `event:يوم القيامة` were second, unlinkable
identities). A named being emitted as a concept is forced to its gazetteer type
(`الشيطان` held a primary *concept* slot, which would have clustered every
waswasa hadith under Satan; as a person at secondary it is not a bucket key).

**Primary-slot policy** — assignment is a **single ranked pass** over every node
(curated concept → bare concept → entity, with the model's own marking breaking
ties inside a rank), not a promote step followed by a truncate step. Splitting
the two left an ordering hole: when the model marked `معاوية`, `العقل` and
`النكراء` all primary there were no secondaries to promote from, so the overflow
cut kept the first two in emission order and demoted `النكراء` — the exact
failure the policy exists to prevent, reappearing whenever the model over-marked.
Concepts now win primary slots outright; an entity holds one only when the
narration produced no concept at all. Primary/secondary controls bucketing, so
this decides what clusters.

**SKOS hierarchy activated** — `broader` was parsed and used by nothing. It now
accepts a list (`الحساب` sits under both `الجزاء الأخروي` and `القيامة`),
`broader_chain()` walks it cycle-guarded, and `bucket_keys_for_chunk` emits every
ancestor. `الحساب` / `الثواب` / `الجزاء` / `العقاب` share the parent
`الجزاء الأخروي`, so hadiths 7, 8 and 9 meet in one bucket while staying
distinct nodes — they are *not* synonyms (reckoning ≠ reward ≠ recompense) and
aliasing them would have erased that permanently.

**Qur'an citations** (`src/extractors/quran_refs.py`, new) — a new `quran_refs`
field, deliberately kept **out** of the Gemini schema so the model has no channel
to invent one. Two independent signals:
- Footnote parser: 114-sura table with verse counts, classical alternates
  (`المؤمن` = 40, `بني إسرائيل` = 17), tolerant of the real formats
  (`البقرة: 269`, `يونس، 39`, `الرعد 41`, `ص: 28`, shadda'd names, trailing
  commentary). Verse-count validation rejects the cross-reference
  `قد مر الحديث ص 321`, which would otherwise parse as Sad:321.
- Phrase matcher against the 6236-ayah corpus at
  `../shiadata-rag/data/quran.json`, via an inverted 16-character gram index
  (the naive scan was quadratic and took minutes per volume). Space-insensitive,
  because the mushaf writes `يَٰٓأُو۟لِى` as one token where al-Kafi prints
  `يا أُولِي` as two.

Footnotes win on conflict — the editor disambiguates wording that sits in more
than one verse, and flags where al-Kafi differs from the mushaf. A footnote only
yields a ref when it *opens* with a sura name, which is what separates a citation
from commentary that merely quotes scripture. Exports `CITES → ayah:2:269`,
matching the prefix the tafsir branch already writes, so hadith and tafsir meet
on one node.

**Footnote stripper hardened** — a footnote quoting *vocalised* Qur'an defeated
the harakat heuristic: note `[3]` on page 12 of vol 1 quotes Surat al-Nas (27
harakat), the stripper declared the matn resumed mid-quote, and both the rest of
the verse and the editor's following sentence leaked into hadith 11's matn.
Fixed with guillemet-balance tracking. `iter_line_roles()` is now the single
source of truth for where a note starts and stops, consumed by both the stripper
and the citation reader — they disagreed at first, and that disagreement is
exactly how a verse got attributed to a hadith that never quoted it.

**`ayah` removed from `NodeType`** — the prompt's own gold example taught `39:9`
for a verse the editor cites as `2:269`. Citations now come from the page.

## Capability delta

| | before | now |
|---|---|---|
| topic nodes | invented free-text per hadith | selected from a closed vocabulary |
| vocabulary growth | hand-edited YAML only | measured df promotes proposals automatically |
| connectivity | depended entirely on the LLM | exported graph guaranteed by 2002 inherited headings; Phase 2 buckets at bab granularity |
| off-vocabulary terms | entered the graph as singletons | routed to `proposed_nodes`, never graphed |
| `broader` | parsed, unused | drives bucket expansion via ancestors |
| Qur'an citations | none (and one fabricated example in the prompt) | footnote-parsed + phrase-matched, model cannot invent |
| narrator as topic | Imam being quoted became a node | dropped via canonicalised isnad comparison |
| vocalised Arabic | `مُحَمَّدُ بْنُ يَحْيَى` ≠ `محمد بن يحيى` | folded, incl. honorifics and kunya case |
| footnote leakage | vocalised Qur'an in a note leaked into matn | guillemet-balance guard, one shared classifier |
| mistyped labels | `place:الجنة` ≠ `concept:الجنة` | forced to catalog/gazetteer type both ways |
| tests | 51 | 82 |

## Data contract changes

Hadith payloads now carry, in addition to the previous fields:

```
kitab           str          printed heading, inherited
bab             str          printed heading, inherited
quran_refs      list[str]    ["2:269"], extractor-injected, never model-authored
proposed_nodes  list[str]    vocabulary gaps; NOT graph nodes
```

`ParsedUnit` gained `quran_refs`, `kitab`, `bab` (all defaulted and
`compare=False`, so positional construction and hashing still work).
`OpenHadith` gained matching seeds; `quran_refs_seed` **unions** across pages
because a multi-page hadith can be footnoted on both sides.

## Operating loop

```bash
python main.py reset-book --book hadith
python main.py run-phase1 --book hadith --limit 40
python main.py proposals                    # review the plan
python main.py proposals --promote          # apply, then re-run phase 1
```

The vocabulary is thin (75 concepts), so the first run will propose a lot. That
is intended: proposals from a few hundred pages are how the vocabulary reaches a
realistic size.

## Open items

- **The new system has not been run against Gemini yet.** Everything is verified
  offline against payloads on disk. The output currently in
  `data/output/phase1/hadith/` predates the vocabulary work and should be
  discarded, not used to judge the design.
- **Vocabulary retrieval is not implemented.** `vocabulary_block()` logs a
  warning past 8000 characters; at that point the full dump must be replaced by
  embedding-retrieved top-K candidates per page. The prompt shape does not change.
- **Proposal repair is lexical only.** The design called for mapping unmatched
  singletons onto their nearest catalog term by embedding cosine;
  `repair_to_vocabulary()` only does constituent-word matching. `EmbeddingAgent`
  exists in `src/agents/embeddings.py` and is unused by `proposals.py`.
- **Repair can outrank a genuine proposal.** `خلق العقل` repairs to `العقل` in the
  graph and only reaches the catalog on the *next* cycle, after its proposal
  accumulates df. The graph is therefore one promotion round behind the
  vocabulary for compounds whose head is already catalogued. Acceptable, but it
  means judging a single run understates what the loop converges to.
- **Bab titles are not seeding the vocabulary.** 2002 human-authored topic labels
  are being read and used for grouping, but not normalised into catalog concepts.
  This is the cheapest available vocabulary bootstrap and is not done.
- **Kitab continuation across volume files is not handled.** `attach_sections`
  runs per file; a kitab spanning `al-kafi-1.txt` into `al-kafi-2.txt` restarts
  empty. In practice each volume re-declares its kitab, so this has not bitten.
- **`محبة أهل البيت` in hadith 5 is unverified** under the new design. It is on
  the menu now, which should be enough, but no run has confirmed it.
- **tafsir and history pipelines still pass `strict=False`.** They have no
  proposals channel, and closing their vocabulary would silently empty them.
- **Hadith 12 collects ~40 Qur'an refs.** It is a verse-dense sermon spanning
  pages 13–17; worth confirming it is genuinely one narration and not an
  accumulator span bug.

## Files

New: `src/extractors/classification.py`, `src/extractors/quran_refs.py`,
`src/pipelines/prompts.py`, `src/pipelines/proposals.py`.

Modified: `config/base_ontology.yaml`, `config/entities.yaml`, `config/paths.py`,
`main.py`, `src/models.py`, `src/pipelines/ontology.py`,
`src/pipelines/llm_processor.py`, `src/pipelines/hadith_accumulator.py`,
`src/pipelines/runner.py`, `src/core/vector_engine.py`,
`src/core/neo4j_export.py`, `src/extractors/chunkers.py`,
`src/extractors/epub_parser.py`, `tests/test_graph.py`.

Unrelated pre-existing modifications also sit in the working tree
(`src/agents/*`, `config/settings.py`, `src/core/phase2.py`,
`src/core/edge_classifier.py`) — not from this session.
