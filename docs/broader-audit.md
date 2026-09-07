# `broader` audit — IS-A verdicts

Audit of every surviving `broader` edge in `config/derived_ontology.yaml` after
the majority-vote rewrite. **831 edges across 41 parents.**

The test applied is IS-A, not printed-under: *is the child a kind of the
parent?* A bab genuinely printed inside a kitab still fails if the concept is
merely discussed there — `الأطفال` under `الجنائز` is a real chapter (the
funeral prayer for children) and a false taxonomy edge.

---

## Method

Three independent signals, because vote purity alone has a blind spot: a term
printed under exactly one book scores 100% however generic it is.

| signal | what it catches |
|---|---|
| **vote purity** ≥ 0.65 | terms spread across several books |
| **is-a-kitab** | books made subtopics of other books |
| **standalone attestation** | fragments layer 2 invented, never a chapter anywhere |
| **corpus spread** (audit-only) | wide-spread terms that still slipped through |

Spread is measured outside the pipeline: across all ~16k headings, how many
distinct books mention the term at all. It is the check that finds what the
in-pipeline rules miss.

---

## Verdicts

### KEEP — verified correct (sample of the 831)

| edge | why |
|---|---|
| `الطواف` → `الحج` | tawaf is a rite of Hajj; 23 of 27 printings under it |
| `الاحرام` → `الحج` | ihram exists only in Hajj/ʿumra |
| `الهدي` → `الحج`, `رمي` → `الحج`, `السعي` → `الحج` | Hajj rites |
| `التيمم` → `الطهارة` | 100% purity, a purity ritual by definition |
| `الغسل`, `الوضوء`, `الحيض`, `الخضاب`, `السواك`, `مسح` → `الطهارة` | ritual purity and grooming |
| `القراءة`, `سجود`, `التسبيح`, `الجهر`, `الجمعة` → `الصلاة` | components of prayer |
| `التزويج` → `النكاح` | 41 of 44 printings |
| `الشهادة` → `الشهادات` | 37 of 39; **was `الجهاد` before the fix** |
| `الميراث` → `المواريث` | **was `النكاح`; 124 of its printings are Fara'id/Mawarith** |
| `الحلف` → `الايمان` | oaths |
| `الأقراء` → `الطلاق` | the cycles that count the ʿidda — a divorce topic |
| `الفطرة` → `الزكاة` | zakat al-fitra; `بَابُ الْفِطْرَةِ` sits in Kitab al-Siyam. Defensible |

### NULLIFIED by this pass — 214 edges dropped

| rule | count | examples |
|---|---|---|
| cross-cutting (no book holds 65%) | 121 | `الأطفال`, `الأولاد`, `الأربعين`, `القبر`, `الموت`, `النية` |
| is itself a book | 63 | `الصيام` (was → `الطهارة`), `النكاح` (was → `الديات`) |
| never a chapter anywhere | 30 | `التوحيد` (was → `الصلاة`), `الجهل` (→`الحج`), `القيامة` (→`معاني الأخبار`), `الفقر`/`العقاب` (→`النكاح`) |

One acceptable loss: `التوكل` → `الإيمان و الكفر` was correct but is only ever a
layer-2 fragment, so the standalone rule drops it. An unparented `التوكل` is a
gap, not a false claim.

### CURATED ALIAS — applied, 7 parents removed

Same book, several printed titles. Justified by a corpus test rather than
judgement: **two titles naming the same book never open in the same volume.**

| merged to | from | verified |
|---|---|---|
| `المواريث` | `الفرائض`, `الفرائض و المواريث`, `الفرائض والمواريث` | ✅ no co-occurrence¹ |
| `الحدود` | `الحدود والتعزيرات` | ✅ |
| `الصيام` | `الصوم` | ✅ |
| `الوصية` | `الوصايا` | ✅ |
| `القضاء` | `القضاء و الأحكام`, `القضايا والأحكام` | ✅ |
| `العتق` | `العتق وكيفيته` | ✅ |

¹ Faqih 4 appears to open both `الفرائض` and `الفرائض و المواريث`, but the short
form there is the truncated `كتاب الفرائض و` — defect 3, not a second book.

**Declined**, because the volumes prove them distinct:

| left separate | evidence |
|---|---|
| `الأطعمة` / `الأشربة` / `الأطعمة والأشربة` | al-Kafi 6 opens الأطعمة **and** الأشربة as separate books |
| `الصيد` / `الذبائح` | al-Kafi 6 opens both |
| `البيوع` / `المكاسب` / `التجارات` / `المعيشة` | al-Istibsar 3 opens البيوع and المكاسب |

Merging these would erase a division the compilers deliberately made.

### REMAINING — 10 or so, nearly all one upstream bug

| edge | verdict |
|---|---|
| `تأخير`→`الصلاة`, `مشي`→`الحج`, `ضرب`→`الحج`, `أصناف`→`الطهارة`, `عدد`→`الصلاة`, `انقطاع`→`الطهارة` | **truncation.** Each is standalone only because a heading was cut: `٤٨ باب وجوب تأخير`, `٧ باب كراهة مشي`, `١ باب أصناف`. Fixing the joiner removes them; no purity rule can. |
| `الاشهاد على`, `الحد الذي`, `مقدار ما`, `شارب الخمر` | same — truncated ids |
| `الاستغفار` → `الصلاة` | **genuine residual.** `بَابُ الِاسْتِغْفَارِ` is in al-Kafi's كتاب الدعاء. The only real mono-kitab error left. |
| `التسليم`/`التكبير` → `الصلاة`, `الحلق` → `الطهارة` | **ambiguous.** Each has two senses (prayer-salam vs greeting; shaving in Hajj vs grooming). Marginal either way. |

**Nine of the ten are downstream of the truncated-heading bug**, which the brief
puts out of scope. They are not a purity failure and should not be chased with
threshold changes.

---

## Three things that look like the fix and are not

All three were implemented and measured against 16-25 edges known correct and
10-19 known wrong. **None shipped.** Do not retry them without reading this.

### 1. Raising `_PARENT_MAJORITY` to 0.80

The intuition is right and the arithmetic defeats it: **a term the headings
printed once has 100% purity by definition**, so no threshold can reach it. The
edges most often flagged as unrelated are exactly those — `البنات` 1 printing,
`الحصاد و الجداد` 1, `الأجير و الضيف` 1, `التكاتب` 1, `الدماء` 3.

| threshold | edges | good kept | bad kept |
|---|---|---|---|
| 0.65 | 837 | 24/25 | 16/16 |
| 0.80 | 780 | **19/25** | 13/16 |

0.80 costs five correct edges (`الغسل`, `رمي`, `الحيض`, `مسح`, `الاعتكاف`) to drop
three wrong ones, and fixes none of the flagged single-printing terms.

### 2. Requiring 2+ printings

Directly targets the single-printing blind spot, and collapses the catalog:
**837 → 177 edges.** Most terms are printed once, including correct ones —
`الخلع والمبارات`, `الأقراء`, `شهر رمضان` each appear exactly once.

### 3. Body-text attestation

The narrations are a sample thousands of times larger than the headings, and on
the flagged terms they are decisive: `البنات` spends 4% of its pages in its
assigned book, `الحلق` 4%, `الاستغفار` 13%, `الدماء` 23% — against `الطواف` 60%,
`التيمم` 69%, `الأقراء` 94%.

It was implemented as a second corpus pass and **reverted**, for two reasons.

*Raw share is confounded by book size.* `كتاب الشهادات` is small, so it cannot
hold a majority of a common word's pages even when the edge is right: `الشهادة`
scored 14%, `الحلف` 1%, both correct. A 0.50 gate dropped 234 edges and took
`الشهادة`→`الشهادات` and `الميراث`→`المواريث` with them.

*Correcting for book size fixes the confound but does not separate.* Lift over
the corpus baseline puts `الشهادة` at 27.9x and `الحلف` at 33.0x — correctly high
— but `البنات` also scores 9.9x, above `الطواف`'s 6.2x, because it really does
concentrate in one small book. No cut separates them.

> The finding worth keeping: **the residual bad edges are not a statistical
> problem.** `البنات` genuinely is discussed in كتاب العقيقة; `الأجير و الضيف`
> genuinely is a chapter of كتاب الحدود. They fail because daughters are not a
> kind of birth-sacrifice and a hired worker is not a kind of hudud — a semantic
> judgement no distribution over the corpus contains. That is precisely what
> layer 4 exists for, and layer 4 has never been run.

---

## Threshold — measured, then chosen

Against 20 edges known correct and 19 known wrong:

| `_PARENT_MAJORITY` | edges | good kept | bad kept |
|---|---|---|---|
| 0.60 | 845 | 20/20 | 19/19 |
| **0.65** | **831** | **20/20** | **15/19** |
| 0.70 | 788 | 18/20 — loses `رمي`→`الحج`, `مسح`→`الطهارة` | 14/19 |

0.65 is strictly better than 0.60. 0.70 starts costing real edges to gain one.
**Chose 0.65.** With the standalone rule added on top, bad-kept falls to 8/19.

---

## Numbers

| | before this pass | after |
|---|---|---|
| parented edges | 845 | 831 |
| distinct parents | 50 | **41** |
| purity <25% (measurable subset) | 33% | **10%** |
| purity <50% | 66% | **41%** |

1,550 concepts total; 719 stand unparented — cross-cutting, a book in their own
right, or with no kitab evidence at all. That is the intended outcome, not a
shortfall: a missing edge is recoverable, a false one propagates.
