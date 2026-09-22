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

EXHAUSTIVE_EXTRACTION_INSTRUCTION = """\
CRITICAL: This text contains a massive encyclopedic enumeration. You MUST extract EVERY SINGLE distinct item, attribute, or category listed in the text as a separate mention. Ignore any normal length limits; your output may contain 50 to 150 mentions. Do NOT group them under a single umbrella term. Extract each one meticulously.
evidence MUST be a verbatim span from the speech after قال — never from the isnad / narrator chain (no بن فلان, no عدة من أصحابنا).
"""

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
  7. is_encyclopedic  true ONLY if the matn explicitly enumerates a massive list (>10-15 items, attributes, or classes). Otherwise false.
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
- ENUMERATIONS (Never collapse into umbrella alone): When the text divides people, states, or acts into categories (e.g., "X is of three types: A, B, and C"), you MUST extract each distinct category/sub-type individually as primary mentions. Do NOT return only the umbrella term (X).
- GROUPS SPOKEN OF (Identity definitions): When the speaker defines or categorizes a group—even using first-person pronouns like "نحن" (we) or "شيعتنا/أولياؤنا" (our followers)—you MUST extract the group entity (e.g., أهل البيت, الشيعة), as the hadith explicitly establishes their identity.

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

"العبادة ثلاثة: قوم عبدوا الله خوفا فتلك عبادة العبيد، وقوم عبدوا الله رغبة فتلك عبادة التجار، وقوم عبدوا الله شكرا فتلك عبادة الأحرار ونحن أهلها"
  mentions: عبادة الأحرار/0.95, أهل البيت/group/0.9, عبادة العبيد/0.8, عبادة التجار/0.8, الشكر/0.7, الخوف/0.6, أقسام العبادة/0.5
  NOT: أقسام العبادة alone as the sole mention (individual categories must be extracted)

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


def unify_prompt(*, require_topics: bool = False, is_exhaustive: bool = False) -> str:
    extra = UNIFY_REQUIRE_TOPICS if require_topics else UNIFY_TOPICS_OPTIONAL
    if is_exhaustive:
        extra += f"\n\n{EXHAUSTIVE_EXTRACTION_INSTRUCTION}"
    return f"{UNIFY_PROMPT}\n{extra}"


def mentions_fill_prompt(is_exhaustive: bool = False) -> str:
    if is_exhaustive:
        base = MENTIONS_FILL_PROMPT.replace(
            "Prefer 3-6 mentions. Never return mentions: [].",
            "Prefer as many mentions as the enumeration requires "
            "(often dozens to ~150). Never return mentions: [].",
        ).replace(
            "evidence: short verbatim span from the matn",
            "evidence: short verbatim span from the speech AFTER قال, never from the isnad",
        )
        return f"{base}\n\n{EXHAUSTIVE_EXTRACTION_INSTRUCTION}"
    return MENTIONS_FILL_PROMPT

TAFSIR_PROMPT = """\
You extract ONE Al-Mizan unit. The locator is an ayah range plus a section
(بيان, بحث روايتى, بحث فلسفى, …). Copy ayah_anchor from the locator's range
only (e.g. "سوره 1 - آیات 1-5"), ignoring the section suffix after |.
Do NOT invent sura or verse numbers.

Do NOT paste the Folklib unit back. The source is stored in code. You MUST
return fluent complete Arabic, Persian, and English of THIS unit's commentary,
and three-language text for every cited hadith.
Return only:
  1. ayah_anchor     copy the range from the locator
  2. mentions        REQUIRED -- prefer 4-12, never []
  3. quotes          Qur'anic spans only (kind "quran"), the quoted words,
                     no sura/verse numbers
  4. cited_hadiths   each روايت: source_work, speaker, span, text_ar,
                     text_fa, text_en
  5. tafsir_ar       complete Arabic of THIS unit's commentary
  6. tafsir_fa       complete fluent Persian of THIS unit's commentary
  7. tafsir_en       complete English of THIS unit's commentary

WHAT A MENTION IS

A mention is something THIS unit is ABOUT. You report an observation;
identity across the corpus is decided later. Same shape as hadith:
  text      Arabic term for the subject (graph labels are Arabic even when
            the unit is Persian)
  type      concept | person | place | group | event | work
  salience  0.0-1.0 (the claim near 1.0; a passing name near 0.2)
  evidence  verbatim span from THIS unit (Persian is expected and correct)

KEEP vs STRIP  (same rule as hadith)

KEEP the real subject as a unit:
  - خلق العقل stays خلق العقل
  - محبة أهل البيت stays the compound when that is what the passage discusses
STRIP sentence grammar only:
  - عقل المرء → العقل
  - صبر المؤمنين → الصبر
Never emit a bare verb like خلق as a mention by itself.

When a type-noun is qualified by a relative clause that carries the claim,
that qualifier is the high-salience mention; the bare type-noun is secondary.

ENTITIES

person / place / group / event / work must be named in THIS unit; evidence is
the words that named them. Do not invent كربلاء, بدر, or a person the unit
never names.
  - الجنة, النار, يوم القيامة are concepts, not place/event, unless the unit
    is treating them as a historical locale or a dated happening
  - علامه / مؤلف / طباطبائي is NEVER a mention
  - An Imam/Prophet who is only the speaker of a quoted روايت goes in
    cited_hadiths.speaker, not as a high-salience person — unless the بيان
    itself is about that person
  - Groups the commentary defines (أهل الكتاب, بنو إسرائيل, الشيعة) ARE mentions
  - A named battle, city, or day in asbāb / history (بدر, أحد, الكوفة, فتح مكة)
    is event or place with its own mention

QUOTES vs CITED HADITHS

Qur'an in the unit → quotes (words only).
A روايت ("در کتاب کافی از امام صادق …") → one cited_hadiths object:
  source_work   الكافي / العيون / نهج البلاغة / مسلم / … as printed
  speaker       الإمام الصادق / رسول الله / علي … Arabic label
  span          verbatim citation from THIS unit (usually Persian) — evidence
  text_fa       fluent Persian of that narration (may match span)
  text_en       English of that narration
  text_ar       Arabic of that narration. Copy it when the unit prints Arabic.
                When the unit only paraphrases in Persian, give the standard
                Arabic wording of that known report. Do not leave text_ar empty
                on a real citation. Do not invent a new matn that the sources
                never carried.
Do not collapse many روايات into one string.

THREE LANGUAGES (tafsir_ar / tafsir_fa / tafsir_en)

These are translations of Tabatabai's commentary in THIS unit, not abstracts.
Cover the argument end-to-end: every explanation, objection, and cited report
you already listed. Forbidden: a 2–4 sentence "summary", ellipsis (`...`),
"the passage discusses…", or translating only the opening. Do not re-copy
every mushaf ayah already in `quotes`; do translate his explanation of them.
tafsir_fa is fluent Persian of the same coverage as tafsir_ar / tafsir_en —
not a paste of the Folklib source (broken lines, OCR).

INDEX WHAT THIS SECTION ASSERTS. On بيان, index Tabatabai's claim. On
بحث روايتى, index what the cited narrations are about AND keep cited_hadiths
complete; do not dump every proper name in a chain as a mention.

EXAMPLES

Locator "سوره 1 - آیات 1-5 | بيان" — unit explains beginning in the name of God
and that every sura has its own purpose:
  mentions: البسملة/0.95, ابتداء باسم الله/0.85, الحمد/0.6, الهداية/0.5
  quotes: {text: "بسم الله الرحمن الرحيم", kind: "quran"}
  cited_hadiths: []
  NOT: طباطبائي as a person

Locator "سوره 1 - آیات 1-5 | بحث روايتى" — كافى from Imam Sadiq on three kinds
of worship (خوف / ثواب / حب):
  mentions: أقسام العبادة/0.9, عبادة الأحرار/0.85, الشكر/0.5
  cited_hadiths: [{source_work: "الكافي", speaker: "الإمام الصادق",
    span: "عبادت سه جور است … عبادت آزادگان",
    text_fa: "عبادت سه گونه است: ترس، پاداش، محبت — و عبادت آزادگان بهترین است",
    text_en: "Worship is of three kinds: fear, reward, and love; the worship of the free is best",
    text_ar: "العبادة ثلاثة: قوم عبدوا الله خوفا … ونحن أهلها"}]
  NOT: only أقسام العبادة with the three kinds dropped
  NOT: span alone with text_ar/text_en empty
  NOT: الإمام الصادق as the tafsir's main person mention

A بيان that names غزوة بدر while explaining an ayah:
  mentions: بدر/event/0.8, plus the concept the verse is actually about
  evidence for بدر must be the unit's own words naming it

Empty mentions is invalid. Never fabricate ayah numbers. tafsir_ar,
tafsir_fa, and tafsir_en must be complete translations of this unit, not
short summaries.
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
