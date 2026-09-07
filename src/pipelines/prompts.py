"""Phase 1 system prompts.

The hadith prompt asks for OBSERVATIONS, not graph nodes. That is the whole
point of the current design: a model reading one narration cannot know how the
other 15,000 phrase the same idea, so asking it to settle identity per hadith
produced a fresh label every time -- عقل المرء, حساب العباد, اجتهاد المجتهدين --
each true of exactly one narration and therefore indexing nothing.

So identity moved out. The model reports what the matn says in the matn's own
words; src/pipelines/resolver.py decides afterwards, with the whole corpus in
view, that عقل المرء and العقل are one thing and that معاوية is one man across
every book. Nothing here constrains vocabulary, and nothing here needs to.
"""

from __future__ import annotations

HADITH_PROMPT = """\
You extract EVERY hadith on this printed page, not just the first.
Return JSON with page (copy the locator) and hadiths: one object per distinct narration.
If the page starts mid-hadith with no new number, include that fragment first with marker 'continuation'.
Then one object per numbered hadith (e.g. '3 -', '8-', '[ ١٥٤٩٥ ] ١ ـ').
Ignore editor footnotes and bracketed editor asides.

Do NOT copy the Arabic matn back. It is already known from the page; repeating
it wastes the whole response. Return only:
  1. marker      the printed number, e.g. "3 -"
  2. mentions    REQUIRED for every numbered hadith -- prefer 3-8, never []
  3. ravis       isnad order
  4. quotes      Qur'anic spans only (kind "quran"), no sura/verse numbers
  5. hadith_fa   fluent Persian of the FULL matn on this page (no `...`, no summary)
  6. hadith_en   precise English of the FULL matn on this page (no `...`, no summary)
Translate every sentence of the matn present on the page. Never truncate with
ellipsis, never write "ادامه" / "Continuation of", never paraphrase half and drop
the rest. If the page is only isnad with no matn yet, leave fa/en empty.

WHAT A MENTION IS

A mention is something this narration is ABOUT. You report an observation;
identity across the corpus is decided later. Each mention:
  text      Arabic term for the subject
  type      concept | person | place | group | event | work
  salience  0.0-1.0 (subject near 1.0; passing name near 0.2)
  evidence  verbatim matn span that prompted it

KEEP vs STRIP

KEEP the real subject as a unit:
  - خلق العقل stays خلق العقل (do NOT split into خلق + العقل alone as the only topics)
  - محبة أهل البيت stays the compound when that is what the matn discusses
STRIP sentence grammar only:
  - عقل المرء → العقل
  - حساب العباد → الحساب
Never emit a bare verb like خلق as a mention by itself.

When a type-noun is qualified by a relative clause (or similar
attribute) that carries the claim, that qualifier is the high-salience
mention; the bare type-noun is secondary.

Other rules (short):
- person/place/group/event/work must appear in the matn; evidence is their words
- concepts may be inferred from what THIS matn asserts (not the chapter title)
- NARRATORS ARE NOT MENTIONS: isnad + the Imam/Prophet being quoted go in ravis
  معاوية in "فالذي كان في معاوية" IS a mention; أبو عبد الله answering is not
- INDEX WHAT IS ASSERTED, NOT WHAT IS DENIED
- Qur'an goes in quotes as the quoted words only -- never a verse number, never a mention

EXAMPLES

"ما العقل قال ما عبد به الرحمن ... فالذي كان في معاوية فقال تلك النكراء"
  mentions: العقل/0.95, النكراء/0.8, معاوية/person/0.4, العبادة/0.3

"صديق كل امرئ عقله و عدوه جهله"
  mentions: العقل/0.9, الجهل/0.9  (NOT عقل المرء)

"لما خلق الله العقل استنطقه ثم قال له أقبل فأقبل"
  mentions: خلق العقل/0.9, العقل/0.7
  NOT: خلق as a standalone mention

"مؤمن يخالط الناس ويصبر على أذاهم أفضل من مؤمن لا يخالط الناس ولا يصبر"
  mentions: مخالطة الناس مع الصبر/0.95, المؤمن/0.5
  NOT: المؤمن alone as the top subject (the relative clause carries the claim)

"إن عندنا قوما لهم محبة ... فاعتبروا يا أولي الأبصار"
  mentions: محبة أهل البيت/0.8, العزيمة/0.6
  quotes: {text: "فاعتبروا يا أولي الأبصار", kind: "quran"}

Empty mentions on a numbered hadith is invalid. Continuations may omit
translations when the fragment is isnad with no matn yet. Never fabricate a hadith.
"""

UNIFY_PROMPT = """\
From THIS assembled Arabic hadith only (ignore chapter titles and other narrations),
return Persian (hadith_fa), English (hadith_en), ravis, mentions, and quotes.

hadith_fa / hadith_en MUST cover the entire assembled matn end-to-end.
Forbidden: ellipsis (`...`), "ادامه روایت", "Continuation of the previous",
summaries that skip dialogue, or translating only the opening line.
If earlier page fragments left a stub translation, replace it with a complete one.

Mentions = what the matn is ABOUT: {text, type, salience, evidence}.
type is concept|person|place|group|event|work. Prefer plain forms (العقل not عقل المرء)
but KEEP real compounds (خلق العقل). Narrators and the speaker go in ravis only.
Put Qur'anic quotations in quotes as the words only (no verse numbers).
"""

UNIFY_REQUIRE_TOPICS = """\
CRITICAL: mentions are missing. Your FIRST job is mentions (at least 2), then
translations and ravis. An empty mentions array is invalid JSON for this call.
Do not use semantic_nodes. Example shape:
  "mentions": [
    {"text": "العقل", "type": "concept", "salience": 0.9, "evidence": "لما خلق الله العقل"},
    {"text": "خلق العقل", "type": "concept", "salience": 0.8, "evidence": "لما خلق الله العقل استنطقه"}
  ]
"""

UNIFY_TOPICS_OPTIONAL = """\
Mentions are already present. You may add better ones; empty mentions is OK when
you are only filling translations or ravis. Still return complete hadith_fa and
hadith_en for the full Arabic — replace any truncated stubs.
"""

MENTIONS_FILL_PROMPT = """\
List what THIS Arabic hadith is ABOUT. Return JSON with only `mentions`
(at least 2 objects). Each: {text, type, salience, evidence}.

type: concept | person | place | group | event | work
salience: 0.0-1.0
evidence: short verbatim span from the matn

KEEP compounds that are the subject (خلق العقل). STRIP grammar possessors
(عقل المرء → العقل). Do NOT emit bare خلق. Do NOT put narrators or the Imam
speaker in mentions -- only subjects discussed in the matn.
If a type-noun is qualified by a relative clause that carries the claim,
prefer that qualifier over the bare type as the top mention.
Prefer 3-6 mentions. Never return mentions: [].
"""


def unify_prompt(*, require_topics: bool = False) -> str:
    extra = UNIFY_REQUIRE_TOPICS if require_topics else UNIFY_TOPICS_OPTIONAL
    return f"{UNIFY_PROMPT}\n{extra}"


def mentions_fill_prompt() -> str:
    return MENTIONS_FILL_PROMPT


TAFSIR_PROMPT = """\
You extract one Al-Mizan tafsir unit anchored to a Qur'anic ayah range.
Copy ayah_anchor from the locator. Extract any quoted hadith.
Write a two-line Persian summary. Keep tafsir_chunk as the main Arabic/Persian text.

Also return `mentions`: what this passage is ABOUT, in the same shape the hadith
pipeline uses -- text, type (concept | person | place | group | event | work),
salience 0.0-1.0, and the verbatim evidence span. These resolve into the SAME
node space as the hadith mentions, which is what lets a tafsir passage and a
narration on the same subject meet in the graph. Say it plainly: الصبر, not
صبر المؤمنين.
"""

HISTORY_PROMPT = """\
You extract historical events from a long classical Arabic narrative.
Split into distinct events with titles, characters, concepts, and the
paragraphs covering each event. Do not invent events absent from the text.

For each event also return `mentions`: what it is ABOUT, as text + type
(concept | person | place | group | event | work) + salience 0.0-1.0 + the
verbatim evidence span. These share one node space with the hadith and tafsir
pipelines, so name people and places exactly as the narrative does and say
concepts plainly.
"""


def hadith_prompt(extra: str = "") -> str:
    return f"{HADITH_PROMPT}\n{extra}" if extra else HADITH_PROMPT
