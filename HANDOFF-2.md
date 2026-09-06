# shiadata-graph — handoff 2: mentions, corpus-wide resolution, candidate linking

Supersedes the design in `HANDOFF.md`. That document described a closed
vocabulary the model picked from, plus a human-approved proposal queue. **That
approach was replaced.** This document says why, what is there now, and what is
still unverified.

Project root: `D:\shiadata.dev\shiadata-graph`. 130 tests pass
(`tests/test_graph.py` 82, `tests/test_resolution.py` 48). Nothing here has been
run against Gemini — see **Open items**.

> **Revision note.** Two earlier drafts described behaviour the code did not
> have, both times in the same way -- a stage that was correct in isolation and
> disconnected in the pipeline.
>
> 1. `HadithExtraction` carried `mentions`, but the accumulator copied only
>    `semantic_nodes`, so every mention was discarded at the Phase 1 flush.
> 2. Resolution retyped mentions correctly in the node table, but the write-back
>    looked assignments up by the mention's ORIGINAL type, so every retyped
>    mention was dropped from `payload["nodes"]` -- the file Phase 2 and the
>    export actually read. `nodes.json` showed the right label and the right df
>    throughout, so unit tests on the node table passed while the runtime had
>    nothing.
>
> Both are fixed and have regression tests that assert on the **payload**, not
> the node table. The lesson is in the tests now: check what the runtime reads.

## What the graph is actually for

Stated by the project owner, and it determines every trade-off below:

> a huge graph of hadiths, tafsir, history, events, places … that connects
> everything to everything … for the next phases I will query the nodes, group
> them, find the related hadiths for every node, send them **pair to llm** and
> llm says if they're supporting each other or opposing or no relation at all

Plus search: `عقل` and its aliases must find one cluster, and a multi-term query
like `قتل النفس` ∩ `عذاب` must return narrations carrying both.

So nodes are a **candidate generator for pairwise LLM comparison**, and bucket
*size* is a first-class objective. A node with 2,000 narrations is 2M pairs and
is not usable; a node with 8 is ideal.

## Why the previous design was abandoned

It made the vocabulary an **input** to extraction. Three consequences, all fatal:

1. A fixed list cannot contain معاوية, or the thousands of other people in these
   books. Every new entity needed a human to add it.
2. The proposal queue put a human in the loop for every new topic.
3. Forced to pick from a list, the model distorts when the right term is absent.

And the deeper problem, which several rounds of rule-writing failed to solve:
`خلق العقل` (a real recurring topic) and `عقل المرء` (one sentence's grammar) are
**grammatically identical** — noun plus genitive. No surface rule separates them.
Whether a term groups anything is a property of the corpus, not of the string.

So the vocabulary became an **output**.

## Architecture now

### Stage A — extraction reports mentions, not nodes

`src/pipelines/prompts.py`, `src/models.py`

The model returns observations in the matn's own words, with no vocabulary:

```json
{"text": "العقل", "type": "concept", "salience": 0.9,
 "evidence": "ما عبد به الرحمن و اكتسب به الجنان"}
```

- `salience` is continuous, replacing the binary primary/secondary.
- `evidence` is the verbatim span, making every mention checkable against the
  matn — a stronger anti-hallucination guard than any phrasing rule.
- Qur'anic quotations go in a separate `quotes` field as **the quoted words
  only**. The model is never asked for a sura or verse number. `ayah` remains
  absent from `NodeType`.
- `TafsirExtraction` and `HistoricalEvent` gained the **same** `mentions` field,
  so all three pipelines resolve into one node space. Previously tafsir emitted
  free-text `core_concepts` that could never merge with a hadith node — a wall
  between pipelines that defeated "connects everything to everything".
  `_mentions_of` walks **nested** mentions as well: history puts them inside
  `events[]`, so a root-only read found none and the whole pipeline wrote back
  empty `nodes`. It also upcasts the legacy shapes — hadith `semantic_nodes`,
  tafsir `core_concepts`, history `historical_concepts` / `characters_involved`
  — so a part-migrated corpus still resolves. There is an end-to-end test that
  a hadith, a tafsir chunk and a history event mentioning الصبر land on one key.

### Phase 1 flush — mentions have to survive assembly

`src/pipelines/hadith_accumulator.py`, `src/pipelines/llm_processor.py`

Extraction happens per printed page, but a hadith can span pages, so the
accumulator merges fragments and `assemble()` emits the finished narration.
Whatever `assemble()` does not emit does not exist downstream. It previously
emitted only `semantic_nodes`, which meant the entire mention contract was
inert on a live run — the model produced mentions, and they were dropped before
anything could read them.

Now `OpenHadith` carries `mentions_seed` and `quotes_seed`, **unioned** across
pages rather than first-wins, because each fragment sees only its own half of
the matn. A later page can also improve an earlier mention's evidence span or
salience rather than being discarded as a duplicate. Both survive the buffer's
`to_dict`/`from_dict`, so a run resumed mid-hadith does not lose the first
page's mentions.

The **continuation branch** of `consume_page` passes them too. It previously
forwarded only the translations, so every observation made on the second half of
a spanning narration was dropped — the precise case the cross-page union exists
for. Its `ravis` and `quotes` were lost the same way.

`quotes` are resolved into `quran_refs` in `assemble()` via `match_quran`. The
model marks *where* it saw a quotation; the mushaf decides *which* verse. Before
this the field was collected and thrown away, leaving citation coverage entirely
to the page-level scan, which attributes by footnote marker and page segment and
so misses a quotation sitting in a narration the scan assigned elsewhere.

Grounding runs **once**, in `assemble()`, against the assembled matn. The
per-page branch of `remap_hadith_payload` deliberately does not ground: there
`hadith` is only the fragment printed on that page, so a mention whose evidence
sits on the next page would be rejected before the accumulator ever saw it, and
`assemble()` could not recover it. The single-narration branch (the unify path)
still grounds, because there the matn is complete.

Two smaller consequences of the same bug:
- `needs_enrichment` keyed on `semantic_nodes` alone, so every mention-only
  payload looked hollow and went to unify — a second model call per hadith, for
  nothing. It now counts either channel.
- `unify_assembled_hadith` never copied `result.mentions` / `result.quotes` onto
  the payload, so even the recovery path produced nothing. It does now.

### Stage B — identity resolved corpus-wide

`src/pipelines/morphology.py`, `src/pipelines/resolver.py`,
`src/pipelines/resolve_pass.py`, CLI `python main.py resolve-nodes`

Deterministic, no model, no network. Three signals:

**1. Arabic root morphology.** A ~100-line stemmer with no dependencies. It is
used as a **stable hash, not as linguistics** — `الحساب` reduces to `حسب`, which
a lexicographer may dispute, but `حساب العباد` reduces to `حسب عبد`, so the two
agree. Consistency is all that is required, which drops the bar from "needs
CAMeL Tools" to a pattern table.

Verified: `العقل`, `عقول`, `العقول`, `عاقل`, `يعقل`, `المعقول`, `بالعقل`,
`للعقل` → one key. Likewise the `حسب` and `جهد` families.

It also detects tautologies for free: `اجتهاد المجتهدين` is ج-ه-د twice, so it
is noise by construction, with no list naming it.

**1b. Type is corrected from the catalog, both directions.** `_seed_key` looks
across types, not only the one the model declared. Without it `place:الجنة` and
`concept:الجنة` are two unlinkable identities for one thing, and `concept:الشيطان`
never meets the gazetteer's person. The old per-hadith gate corrected this;
moving extraction to mentions left the correction behind until it was restored.

**2. Curated seeds, demoted from gate to exception table.** `base_ontology.yaml`
and `entities.yaml` now express only what clustering *cannot* discover:
`قتل النفس` → `الانتحار` share no root and no letters, so only a human-written
alias joins them.

**3. Compound recurrence.** A multi-word mention keeps its own identity only if
it recurs across enough documents; otherwise it folds into the constituent that
names a known topic. The threshold **scales with corpus size**
(`compound_threshold()`): two co-occurrences out of 30 documents is evidence,
two out of 15,000 is a coincidence, and a fixed floor would promote a long tail
of accidents on a full run.

Concepts and entities use **different identity rules**, which matters:
- concept compounds are *narrower topics* → fold when rare
- entity compounds are *fuller names* → merge by prefix, always

so `زرارة` + `زرارة بن أعين` become one person and `واقعة صفين` + `صفين` one
event, none of them enumerated anywhere. Nasab chains are cut at بن; generic
heads (`واقعة`, `يوم`, `غزوة`, …) are stripped.

Prefix containment alone was not safe enough. A short form used *constantly* is
not one person referred to briefly — it is a kunya several men share, and a
single bare `أبو محمد` would otherwise chain `الرازي` to `العسكري`. So a short
name is only absorbed when its own df is below `ENTITY_MERGE_MAX_DF` (12);
frequency is the cheap discriminator, and a curated gazetteer entry overrides it
either way.

### Stage A-bis — grounding

`src/pipelines/grounding.py`, applied in `assemble()` and in the
single-narration branch of `remap_hadith_payload`

The prompt requires entities to appear literally in the matn and `evidence` to be
the words the mention came from. Nothing enforced either, so `evidence` was
decoration. It is a guard now: a mention whose evidence is not in the matn is
dropped, and a person/place/group/event/work whose own name is not in the matn is
dropped. Comparison is folded and space-insensitive, because the matn is
vocalised and the model's echo usually is not.

A mention with **no** usable evidence is dropped too, unless its own term is in
the matn. Previously a blank evidence field simply skipped the check, so any
invented concept passed untouched and the claim "evidence makes every mention
checkable" was only half true. `REQUIRE_EVIDENCE` turns this off if a live run
shows the model will not supply spans reliably.

**This also restores the narrator filter.** Under the old per-hadith gate,
`enforce_node_policy` dropped person/group nodes that resolved to the hadith's
own isnad. The mention contract took extraction off that path, which silently
reintroduced the very first bug this project fixed — the Imam being quoted
becoming a topic. `ground_mentions` now takes `ravis` and filters them, comparing
**gazetteer-resolved identities** rather than raw strings: an isnad printing
أبو عبد الله and a mention saying جعفر بن محمد are the same man, and a string
compare would let the speaker back in under any of his other names.

Also note Arabic *iḍāfa* puts the topic on either side: `عقل المرء` (first word)
and `كمال العقل` / `قدر العقول` (second). The rule is *the first constituent the
catalog recognises*, not the head. Head-first was tried and is wrong.

Verified end to end: `العقل`, `العقول`, `عقل المرء`, `قدر العقول`, `كمال العقل`
→ one node df=4; `خلق العقل` (df=3) survives as a **child** of it; `عتاب الله`
(df=1, anchored to nothing) is dropped.

### Stage C — candidate pairs, ranked

`src/core/candidates.py`

Standard record linkage: **block cheaply → score deterministically → spend the
model on the top of the ranking.**

The consequence that matters: **no single signal is load-bearing.** Earlier
designs made nodes the sole mechanism, so a mislabelled hadith lost a relation
outright. Now a missed node still leaves the shared bab, shared verse, shared
narrator and vector similarity; the pair ranks lower instead of disappearing.
There is an explicit test for this.

Weights are IDF-based — two narrations sharing `الانتحار` is strong evidence,
sharing `العقل` inside كتاب العقل is nearly none. A binary role cannot express
that; a number can.

`Candidate.blocked_by` records which signal *proposed* a pair, separately from
the signals that merely scored it. Almost every bab-proposed pair also picks up a
kitab score, so counting scoring signals as the source made kitab look like the
dominant blocker when it never blocks at all.

Node keys reaching the linker include each node's morph `parent` and its curated
`broader` chain. Stored only on the node table, the hierarchy was documentation:
`خلق العقل` never met plain `العقل`, and `الحساب` / `الثواب` / `الجزاء` never met
under `الجزاء الأخروي`.

Blocking skips keys that are too common to be evidence (`MAX_BLOCK_DF_RATIO`
0.02, `MAX_BLOCK_MEMBERS` 400, and `MIN_DOCS_FOR_RATIO` 200 so small trial runs
do not return zero). Kitab **never** blocks — al-Kafi's largest is 1,607
narrations, i.e. 1.3M pairs from one key — but still contributes to scores.

### Qur'an citation, three tiers

`src/extractors/quran_refs.py`

1. **verbatim** phrase match against all 6,236 ayat, via an inverted 16-char gram
   index. Space-insensitive, because the mushaf writes `يَٰٓأُو۟لِى` as one token
   where al-Kafi prints `يا أُولِي` as two.
2. **root-sequence** match (new). al-Kafi reads `يَتَذَكَّرُ` where 2:269 has
   `يَذَّكَّرُ`; both reduce to `ذكر`, so this finds quotations letter matching
   cannot. Verified to recover 2:269 independently, and to find 13:11 from a
   paraphrase.
3. **footnotes** — now *confirmation and disambiguation only*, not the source.
   A note yields a ref only when it *opens* with a sura name, which separates a
   citation from commentary that merely quotes scripture.

`U+0670` (superscript alef) must be folded **to** an alef, never deleted, or
`ٱلْأَبْصَٰرِ` stops matching `الأبصار`.

## Measured on the real corpus

All offline, no API calls.

| | |
|---|---|
| hadiths detected, al-Kafi 1–8 | 15,216 |
| kitab/bab headings recovered | 2,002 |
| bab buckets | 1,656 (median 6, mean 8.8, max 174) |
| pairs if every bab compared internally | **119,742** — affordable |
| pairs if every *kitab* compared internally | 5,569,135 — not |
| babs over 30 hadiths | 34 (92% are ≤ 20) |

Candidate generation over volumes 1–2 (3,791 hadiths), using **only** bab and
Qur'an citation — no semantic nodes yet:

```
all-pairs        7,183,945
candidates          27,900   (0.39%)
proposed by:     bab 26,005 | ayah 1,895
also scored on:  kitab 26,733 | bab 26,235 | ayah 1,959
```

Note the two lists differ: kitab **proposes nothing** — it never blocks — but
scores nearly every bab-proposed pair, which is why the two are reported apart.

914 of 3,791 hadiths (24%) carry Qur'an refs with no Gemini call, up from 731
once root-sequence matching was wired into the page scan. Several
top-ranked pairs are **cross-chapter, raised only by a shared verse** — the
cross-cutting links bab can never produce.

## Data contract

Hadith payloads now carry:

```
mentions        list[{text, type, salience, evidence}]   from the model
quotes          list[{text, kind}]                       quoted spans, no refs
nodes           list[{key, label, type, weight}]         written by resolve-nodes
quran_refs      list[str]                                extractor-injected
kitab, bab      str                                      inherited from the book
```

`semantic_nodes` and `proposed_nodes` remain, read-only, so payloads extracted
under the old contract still resolve — `resolve_pass` upcasts them, mapping
primary/secondary onto salience.

## Pipeline

```
run-phase1     → mentions per hadith
resolve-nodes  → corpus-wide identity, writes nodes back + data/output/phase1/nodes.json
run-phase2     → candidate pairs → LLM relation judgement
export-neo4j
```

`resolve-nodes` **must** be a separate pass: knowing `عقل المرء` is `العقل`
requires having seen `العقل` elsewhere, which per-page extraction cannot do.

`python check_nodes.py` demonstrates all of it on real data, free and instantly.

## Integration status

The resolver is **on the runtime path**, as of this revision. Previously it was
not, and the handoff described an architecture that was not what `run-phase2`
actually executed:

- `graph_nodes_for_chunk` and `bucket_keys_for_chunk` now read
  `payload["nodes"]` (resolver output) and only fall back to the old ontology
  gate when the resolver has not run. Resolver keys are canonical, so buckets
  key on the key, not the display label, and there is no alias expansion or role
  gate on that path.
- `run_phase2` calls the new `classify_candidate_pairs`, which ranks every
  candidate with `candidates.generate` and then spends the model top-down under
  `settings.edge_max_pairs` (default 5,000). `classify_concept_groups` remains
  as the legacy path but is no longer called.
- `remap_existing_semantic_nodes` skips any payload that already has resolved
  `nodes`, so the per-hadith gate cannot overwrite canonical identities with the
  old alias table.

Dual identity therefore exists only in one direction: a payload extracted before
the resolver still exports through the legacy gate, which is what keeps a
part-migrated corpus working. Once every payload has been through
`resolve-nodes`, `ontology.resolve_node`'s strict mode, `repair_to_vocabulary`
and `proposals.py` are all dead code and should be deleted.

## Open items — read before trusting anything

- **Nothing has run against Gemini under this contract.** The mention schema and
  the new prompt are unvalidated against a live model. Everything measured above
  uses the deterministic layers only.
- **`data/output/phase1/hadith/` is empty.** No stale payloads to mislead.
- **Structural footnote separation is NOT done.** The stripper still uses
  harakat counting plus guillemet balance. A structural rule (the inline `[n]`
  markers in the matn say exactly which notes to expect, and the notes form a
  numbered trailing block) would be far more robust, but rewriting it risks the
  mid-page-resume behaviour two existing tests depend on. The specific reported
  bug — a footnote quoting vocalised Qur'an leaking into hadith 11 — is fixed
  and has a regression test.
- **Embedding signals are wired but unused.** `candidates.generate()` accepts a
  `vectors` argument and scores cosine when given one; nothing supplies it yet.
  Tier-3 ayah retrieval against the Qur'an collection in `shiadata-rag`
  (6,236 ayat, already embedded in Chroma) is designed but not built — that is
  what would catch allusions sharing no wording with the verse.
- **`salience` is uncalibrated and must not gate anything.** Models produce
  stylistic 0.9/0.7/0.5 clusters rather than a real scale. It is used as a soft
  weight only — IDF and multi-signal ranking dominate — and nothing thresholds
  on it. Do not add a salience cutoff to blocking or graph membership before
  measuring calibration on a live run.
- **Morphology-as-hash will over-merge, by design.** Short or ambiguous roots
  will collapse unrelated topics, and everything derived from `عقل` becomes one
  large node. For *search* that is correct behaviour. For *pair candidates* it
  wastes model calls, so the block caps matter: a morph mega-key hits
  `MAX_BLOCK_MEMBERS` / `MAX_BLOCK_DF_RATIO` and stops proposing pairs while
  still contributing to scores. The false-merge rate has not been measured on
  live mentions.
- **`COMPOUND_MIN_DF` now scales with corpus size, but the slope is a guess.**
  `COMPOUND_DF_PER_10K = 3` was chosen without data.
- **Entity merging is guarded but not solved.** `ENTITY_MERGE_MAX_DF` stops a
  very common short name being absorbed, which handles the `أبو محمد` chain. It
  does not handle two rare men with the same rare name. Shared-context signals
  (same bab, overlapping isnad) would be the next improvement.
- **`quotes` are wired, `vectors` are not.** Quoted spans now resolve to
  citations; `candidates.generate()` still accepts a `vectors` argument that
  nothing supplies, and tier-3 embedding retrieval for allusions is designed but
  unbuilt.
- **`proposals.py` and the closed-vocabulary gate are now dead weight**, kept
  only so payloads extracted under the old contract still export. Delete both
  once the corpus has been through `resolve-nodes`; see **Integration status**.
- **`check_nodes.py` matches this document.** It previously demonstrated the
  abandoned closed-vocabulary design, which would have taught the next session
  the wrong architecture. Sections now cover roots, corpus resolution,
  grounding, chapters, Qur'an tiers, footnotes and pair ranking.
- **Search is not implemented.** Intersection queries (`قتل النفس` ∩ `عذاب`) are
  what the node design is meant to support, but no query layer exists.

## Files

New: `src/pipelines/morphology.py`, `src/pipelines/resolver.py`,
`src/pipelines/resolve_pass.py`, `src/pipelines/grounding.py`,
`src/core/candidates.py`, `tests/test_resolution.py`, `check_nodes.py`.

Rewritten: `src/pipelines/prompts.py`.

Modified: `src/models.py`, `src/pipelines/hadith_accumulator.py`,
`src/pipelines/llm_processor.py`, `src/pipelines/ontology.py`,
`src/core/vector_engine.py`, `src/core/edge_classifier.py`,
`src/core/phase2.py`, `src/extractors/quran_refs.py`, `config/settings.py`,
`main.py`, `tests/test_graph.py`.

From the previous session and still in place: `src/extractors/classification.py`
(kitab/bab inheritance), `src/extractors/quran_refs.py` tiers 1 and 3,
the guillemet-balanced stripper and shared `iter_line_roles`.
