# Ontology Handoff — the concept catalog and how it is built

Written for another agent picking this up. Everything here is current as of the
last `harvest-ontology` run and was verified against the corpus, not recalled.

---

## 1. What problem this solves

`shiadata-graph` turns classical Shi'i hadith books into a knowledge graph. The
end goal is pairwise reasoning: query nodes, group them, pull every narration
attached to a node, and send pairs to an LLM that judges *supports / opposes /
unrelated*. Search has to work too — `عقل` and its aliases must land on one
cluster, and `قتل النفس` ∩ `عذاب` must return narrations carrying both.

That needs a **shared vocabulary**. If one narration is tagged `العقل` and
another `عقل المرء`, they never meet. So there has to be a catalog of concepts
that extraction can select from and resolution can collapse onto.

The catalog cannot be hand-written. Classical fiqh and ethics have thousands of
concepts; enumerating them by hand is both endless and arbitrary. The insight
that unblocked this: **the scholars already wrote the vocabulary.** Al-Kafi,
Wasa'il, al-Faqih and al-Istibsar are *classified* corpora — every narration
sits under a `كتاب` and a `باب`, and those headings are printed in the text.
There are ~16,100 of them across the 77 volumes in `data/raw_epubs/hadith/`.

The catalog is mined from those headings. Nobody invents it.

---

## 2. The dual-catalog design

Two files, and the split matters:

| file | rows | written by | on conflict |
|---|---|---|---|
| `config/base_ontology.yaml` | 56 | **by hand** | **wins** |
| `config/derived_ontology.yaml` | 1,550 | `harvest-ontology` | loses |

`base_ontology.yaml` holds judgement calls and aliases — the things a machine
should not decide (`قتل النفس → الانتحار`). It is small on purpose and is
**never touched** by the harvest.

`derived_ontology.yaml` is regenerated wholesale every run. Do not hand-edit it;
the header says so. Of its 1,550 concepts, 839 carry a `broader` pointing at one
of 46 parents — all of them real kitab titles.

Row format:

```yaml
concepts:
  - {id: "أداء الأمانة", pref: "أداء الأمانة", broader: "المعيشة"}  # x2
```

`# xN` is the corpus document frequency — how many headings proposed this term.
It is evidence, not decoration; layer 3 reads it.

**Regeneration is destructive by design.** `reset_derived()` blanks the file
*before* scanning. Without that the harvest reads the file it is about to
replace, so its own junk cites itself as evidence — `أخذ` survived three
harvests that way, each one pointing at the last.

---

## 3. The four-layer enrichment ladder

Each layer is strictly cheaper and more certain than the one after it. Nothing
reaches an LLM until three deterministic passes have failed on it.

### Layer 1 — headings become concepts (free, deterministic)

`harvest.scan()`. Walk every volume in reading order. A `كتاب` heading opens a
span and becomes a parent; a `باب` heading under it becomes a concept whose
`broader` is that kitab.

Titles too long or too clausal to be a concept are not discarded — they go to
`Harvest.long_titles` for layer 2.

**Yield: ~1,561 terms.**

### Layer 2 — decomposition of long titles (free, deterministic)

`harvest.mine_long_titles()` → `resolver.decompose()`.

`وجوب الإخلاص في العبادة والنية` is not a concept, but it *names* three:
`الإخلاص`, `العبادة`, `النية`. Decomposition pulls out the constituents that
already name known concepts. It runs to a fixpoint (3 rounds), because each
round decomposes against a catalog the previous round grew.

> Do **not** replace this with splitting on `و` / `في`. That was tried and it
> corrupts real words — `ولاة العدل` became `لاة العدل`. Decomposition is
> catalog-driven: a fragment survives only if it matches a concept that exists.

**Yield: +30.**

### Layer 3 — frequency promotion (free, deterministic)

`harvest.promote_recurring(min_df=2)`. A label several distinct headings
independently reach for is a real topic; a label one heading wanted is that
heading's phrasing. `--min-df` is the knob.

**Yield: +1.** (Low because layers 1–2 are now thorough. It used to matter more.)

### Settling — who gets a parent (free, deterministic)

`harvest.settle_parents()`, run from `main.py` once all three layers have voted.
This is where `broader` is actually decided; the layers only record evidence.
See §7 defect 1 — it is the single most important correctness step in the
pipeline and the easiest to get wrong.

### Layer 4 — LLM adjudication (costs money, last resort)

`main.py adjudicate`. Only labels that layers 1–3 could not resolve reach here.
Asked **once per distinct string**, offline, cached in `config/adjudicated.json`
so the same string is never paid for twice. `BATCH_SIZE = 40`.

**Status: never run.** No cache file exists. 20 orphans are waiting:

```
إرشاد المستشير · استثمار المال · الآخرة · الأمر والنهي · الحكمة · الدنيا
الدنيا والآخرة · الصمت · العتاب · الفهم · المروءة · المودة · ذم الكثرة
طاعة الشيطان · طول الأمل · فضول الكلام · كف الأذى · مجالسة الصالحين
مدح القلة · معرفة الله
```

Note what these are: genuine ethical concepts with no kitab of their own.
That is exactly the residue layer 4 exists for. Run `adjudicate --no-ask` to see
the plan without spending anything.

---

## 4. Running it

```bash
python main.py harvest-ontology
```

Layers 1–3, ~2 minutes, writes `derived_ontology.yaml`. Add `--dry-run` to
report without writing.

```bash
python main.py adjudicate --no-ask
```

Layer 4 replay — shows orphans and the plan, spends nothing. Drop `--no-ask` to
actually call Gemini; add `--apply` to write accepted verdicts.

```bash
python main.py resolve-nodes
```

Separate pass, runs between phase 1 and phase 2. Resolves every mention in the
corpus into canonical nodes. Identity is a property of the **whole corpus** —
knowing `عقل المرء` is `العقل` requires having seen `العقل` elsewhere — so it
cannot be settled while extracting a single page.

---

## 5. Invariants that were expensive to learn

Every one of these is a bug that shipped. Do not undo them without reading why.

**Mentions vs nodes.** The model reports *observations* on a page. Identity is
resolved corpus-wide afterwards, never during extraction. This is why
`resolve-nodes` is its own phase.

**Fixes go in the scripts, not the prompt.** A standing constraint from the
project owner. With thousands of narrations, encoding exceptions in the prompt
is fragile and unbounded. The prompt is mature; leave it alone.

**No exception lists.** Every filter must be structurally justified, not a
catalog of observed junk. If you find yourself adding a special case for one
book, the rule is wrong.

**A rejected kitab clears the running one.** State that outlives the section it
describes is worse than no state. When `كتاب الحج` was rejected, the scan kept
`الصيام` and handed it to hundreds of Hajj chapters — which reads as fact
rather than as a gap.

**`_MIN_TERM_CHARS` is measured on the surface form, not the folded one.**
`normalize_ar` strips the article, so `الحج` folds to `حج`; a two-character
floor silently killed `الحج`, `الحق`, `الدم` and `الأم`.

**A kitab title is a definite noun phrase** (`classification._names_a_book`).
`كتاب` also means a letter or a scripture, and prose using it that way was
opening spurious books mid-volume that stole the chapters of the real one —
`كتاب اللَّه حقّ` ("the Book of God is truth") took 26, `كتاب جليل و إذا فيه`
took 36. Validated against all 80 kitab titles the corpus prints; none rejected.

**Volume apparatus is not subject** (`classification._APPARATUS`).
`كتاب الصلاة القسم الثالث` is where the publisher split the book. Stripping
`القسم …` and `فهرس/فهرست` merged five forged parents back onto the books they
always were. The `$` anchor matters: `باب القسم بين النساء` keeps its `القسم`.

**An abandoned kitab stops being inherited** (`harvest._abandoned_from`).
Faqih vol 2 runs `كتاب الصوم` straight into the Hajj chapters — the words
`كتاب الحج` appear **nowhere in its body**, so no detector can find the seam.
Instead: a kitab is discussed throughout its own span. Page by page, whether the
kitab label occurs at all, a sound span stays lit end to end (al-Kafi 4's `الحج`
66%, its `الصيام` 54%) while a span holding two books lights up and goes dark.
Find the one split where a well-attested prefix meets a silent tail.

Across the corpus's 86 spans this fires **twice** — on Faqih vol 2, cutting
exactly between `باب الاعتكاف` (last fasting chapter) and `باب علل الحج` (first
Hajj one), and on Faqih vol 4 where a `الفرائض` span runs into the book's
alphabetical rijal index.

> The tail is **orphaned, not re-parented.** We can prove those chapters are not
> `الصوم`; we cannot prove what they are. In a graph built on edges, an honest
> gap beats a confident lie.

**`morphology.root()` is a stable hash, not a root extractor.** It is built for
consistency, not linguistic correctness — `بالحج` → `لحج`, `للحج` → `للح`,
`يحج` → `يحج`. It is fine for clustering surface variants. It is **not** fine
for "does this text discuss X"; using it that way produced a false positive that
would have orphaned 139 genuine Hajj chapters. The attestation signal above uses
literal substring matching instead, and separates cleanly.

**Superscript alef (U+0670) folds *to* an alef, never deleted.**

---

## 6. A path deliberately not taken

Suppressing the back-of-book `فهرست` pages looks obviously right — they are
heading-shaped table rows, 6,360 of the corpus's ~16,100 headings, and they
inflate spans absurdly (`كتاب النكاح` with 318 babs across fourteen pages).

**It was implemented, measured, and reverted.** It cost 124 real terms and seven
real parents (`الرهن`, `الضمان`, `الوصية`, `المزارعة والمساقاة` among them) to
remove a single malformed one, and it stripped Wasa'il vol 4 of all but two of
its 181 titles. The indexes are the book's own table of contents, grouped under
the book's own kitab, and for the volumes whose body headings this parser cannot
see they are the **only** record those chapters exist.

The reasoning is preserved in the docstring of `classification.page_headings` so
it is not re-tried. The one genuine defect it was covering — `الفرائض و`, a row
that wrapped mid-phrase — is fixed directly by `_DANGLING_CONJUNCTION`.

---

## 7. Known defects, measured

Worst first.

> A note on how this section was arrived at, because it matters for trusting it.
> The first audit checked every `broader` for *bibliographic* accuracy — does
> this bab really appear inside that kitab? — and reported ~97% sound. That was
> the wrong question. `broader` is a SKOS IS-A edge, and the right question is
> whether the child is a *kind of* the parent. Re-measured that way the number
> was 66% wrong, not 3%. Defect 1 below is the result. **When auditing this
> catalog, measure IS-A, not printed-under.**

**1. `broader` conflated location with taxonomy — FIXED.** A heading tells you
where a chapter was **printed**; `broader` has to mean the child **is a kind of**
the parent. The two coincide only for titles specific to their book. `طواف
النساء` is printed under كتاب الحج and is genuinely part of Hajj. `باب الأطفال`
is printed under كتاب الجنائز because of the funeral prayer for children — but
children are not a kind of funeral.

Reading location as taxonomy produced `الميراث`→`النكاح`, `الشهادة`→`الجهاد`,
`القبر`→`الحجة`, `الأربعين`→`القضاء`, `الصيام`→`الطهارة`. Measured over the
parented terms appearing in 4+ kitab-bearing headings, **66% had their assigned
kitab holding under half that term's printings.** Root cause was first-write-wins
(`setdefault`), so the parent was decided by `sorted(glob(...))` — filename order.

**Now:** `harvest.settle_parents()`, called from `main.py` after layer 3. Every
printing is a vote; nothing is decided while scanning. An edge survives only if
all four hold:

| rule | constant | drops |
|---|---|---|
| one book takes a clear majority of the term's printings | `_PARENT_MAJORITY = 0.65` | 121 cross-cutting |
| the term is not itself a book | `Harvest.kitabs` | 63 |
| some book once made it a chapter in its own right | `_REQUIRE_STANDALONE_FOR_PARENT` | 30 |
| books with no kitab structure abstain rather than vote | — | — |

Anything failing them is left **unparented** — an honest gap. Of 1,550 concepts,
831 keep a parent and 719 stand alone.

Both numeric constants are measured, not guessed; the table is in
`docs/broader-audit.md`. 0.60 keeps every bad edge, 0.70 costs good ones.

**2. One book, several printed titles — FIXED, curated.** `الفرائض`,
`المواريث` and `الفرائض والمواريث` are one book; left alone they were three
parents, and worse they **split the vote** so children reached no majority at
all. `_KITAB_ALIASES` merges them, applied at vote-grouping time.

The merges are curated but not guessed: two titles naming the same book never
open in the same volume, which is checkable. Every merge was verified to have no
co-occurrence. `الأطعمة`/`الأشربة`, `الصيد`/`الذبائح` and `البيوع`/`المكاسب` **do**
co-occur (al-Kafi 6, al-Istibsar 3) and are deliberately left separate — merging
them would erase a division the compilers made. Parents: 50 → 41.

> Adding an alias means asserting two titles are one book. Run the
> co-occurrence check first; if they ever open in the same volume, don't.

**3. Single-printing edges keep whatever parent they were printed under.** The
majority vote cannot see them: one printing is 100% pure. `البنات`→`العقيقة`,
`الأجير و الضيف`→`الحدود`, `الحصاد و الجداد`→`الزكاة`, `التكاتب`→`العشرة` are all
real chapters of those books and all fail IS-A.

Three candidate fixes were implemented and measured — a higher threshold, a
minimum printing count, and body-text attestation — and **all three were
reverted**; `docs/broader-audit.md` has the numbers. The short version is that
this residue is semantic, not statistical: the corpus genuinely discusses
daughters in كتاب العقيقة. **Layer 4 is the designed answer and has never been
run.**

**4. Truncated titles — 34 confirmed, and a further source of bad
`broader` edges too.** The parent is right, the id is chopped
mid-phrase: `الوصية إلى`, `مقدار ما`, `الاشهاد على`, `دية لسان`, `دية مفاصل`,
`الرجل`. Concentrated in `الوصية` (**15 of its 22 children are fragments**) and
`الديات`.

Cause: `classification._MAX_JOINED_LINES = 2`, the cap on how many wrapped lines
a heading absorbs.

> **Raising the cap is the wrong fix — this was tested.** At 4, 6,034 of 16,108
> headings change and get *worse*, swallowing chains of transmission:
> `بَابُ وُجُوهِ الْقَتْلِ عَلِيُّ بْنُ` → `… عَلِيُّ بْنُ إِبْرَاهِيمَ قَالَ …`.
> Note the cap-2 form is *already* over-joined. The joiner errs in both
> directions at once; it needs a real stopping rule (stop at an isnad opener,
> stop at a complete noun phrase), not a bigger number.

Nine of the ten `broader` edges that survived the audit are downstream of this:
`تأخير`→`الصلاة` exists only because `٤٨ باب وجوب تأخير` was cut, `أصناف`→`الطهارة`
because of `١ باب أصناف`. No purity rule can reach them — fix the joiner instead.

**5. Cross-kitab leak in Faqih vol 4.** Three `الوصية` topics filed under
`الديات`: `الرجوع عن الوصية`, `رسم الوصية`, `انقطاع يتم اليتيم`.

**6. Formatting artifacts.** `الطرار *`, `الشفتين و -`.

**7. Corpus coverage.** The harvest reads only `data/raw_epubs/hadith/` — 77
volumes — which is the intended scope for now. `bihar-al-anvar/` (105 volumes)
is out of scope, and would need its own heading pattern anyway: Bihar writes
`(باب ٨)`, parenthesized with the number *after* the word and the topic on a
separate line. Vols 2 and 3 currently yield zero headings.

---

## 8. Map of the code

| file | role |
|---|---|
| `src/pipelines/harvest.py` | layers 1–3; span buffering; term filters; `_abandoned_from` |
| `src/pipelines/adjudicate.py` | layer 4; orphan discovery, batching, cache |
| `src/pipelines/resolver.py` | corpus-wide clustering; `decompose`, compound thresholds |
| `src/pipelines/ontology.py` | dual-catalog loading, `normalize_ar`, cached indexes |
| `src/pipelines/morphology.py` | surface-form hashing — **not** a root extractor |
| `src/extractors/classification.py` | kitab/bab detection, `heading_topic`, `_names_a_book` |
| `config/base_ontology.yaml` | 56 hand-written concepts; wins on conflict |
| `config/derived_ontology.yaml` | 1,550 generated concepts; regenerated wholesale |

Full `broader` audit with per-edge verdicts: `docs/broader-audit.md`.
Current catalog grouped for review: `docs/derived-ontology-by-parent.md`
(regenerate it after any harvest — it is a snapshot, not live).

Tests: `tests/test_enrichment.py` covers the ladder and the heading invariants.
Full suite is 167 tests, all passing. Anything in §5 has a test — if you change
one of those behaviours a test will tell you which invariant you broke.

---

## 9. If you are picking this up

Highest value first:

1. **Run layer 4.** Now the highest-value item, not the cheapest one: it is the
   only remaining answer to §7.3, the single-printing edges. 20 orphans cached
   forever after. Consider widening it to adjudicate suspect edges too, not just
   unresolved labels.
2. **Fix the heading joiner** (§7.4). Damages 34 ids and causes most of the rest
   of the bad edges. Read the warning first — raising the cap makes it worse. It has never been run. 20 orphans, one batch, cached
   forever after. Cheapest remaining win.
3. **Nullify known-bad edges in `base_ontology.yaml`** — base wins on conflict,
   so declaring a concept there with no `broader` suppresses the derived one.
   `الاستغفار`, `البنات`, `الأجير و الضيف`, `الحصاد و الجداد`, `التكاتب` to start.
4. **Phase-1 validation.** A real Gemini run under the current lean schema has
   not been done end to end since the schema changed.

Do not: rewrite the prompt, add per-book exceptions, or re-suppress the fihrist.
