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

For each item give: the original Arabic (hadith), fluent Persian (hadith_fa),
precise English (hadith_en), the narrators in isnad order (ravis), the mentions,
and any quotes.

WHAT A MENTION IS

A mention is something this narration is ABOUT, written the way the matn writes
it. You are not choosing from a list and you are not naming a category. You are
reporting an observation, and something else decides later how it lines up with
the rest of the corpus.

Each mention has:
  text      the term itself, in standard Arabic, as short as it can be while
            still being the thing the matn discusses
  type      concept | person | place | group | event | work
  salience  0.0 to 1.0 -- how much of THIS narration is about it. Use the full
            range. The subject of the hadith is near 1.0; something named once
            in passing is near 0.2.
  evidence  the words of the matn that made you say it, copied verbatim

RULES

1. SAY IT PLAINLY. Write العقل, not عقل المرء. Write الحساب, not حساب العباد.
   A possessor or a genitive that belongs to this sentence's grammar is not part
   of the term. If the matn is about the creation of the intellect, خلق العقل is
   right, because that is a distinct subject and not a rewording of العقل.

2. GROUND EVERYTHING. person, place, event, group and work must appear literally
   in the matn, and `evidence` must be the words you took them from. A concept
   may be inferred, but only from what this matn actually asserts -- never from
   the chapter heading, never from a neighbouring hadith.

3. NARRATORS ARE NOT MENTIONS. Everyone in the isnad belongs in ravis, and so
   does the Imam or Prophet whose words are being quoted. A hadith is not about
   the man who narrated it. معاوية in "فالذي كان في معاوية" IS a mention,
   because the matn discusses him. أبو عبد الله in "قلت لأبي عبد الله" is not,
   because he is the one answering.

4. INDEX WHAT IS ASSERTED, NOT WHAT IS DENIED. "ليس أولئك ممن عاتب الله" denies
   that the group is blamed, so blame is not a mention of this narration.

5. QUOTES ARE SEPARATE. When the matn quotes the Qur'an, put the quoted words in
   `quotes` with kind "quran". Do NOT give a sura name or a verse number and do
   NOT make the verse a mention -- the citation is resolved against the actual
   mushaf afterwards. Copy the quoted span and nothing else.

6. BE COMPLETE. Give 3 to 8 mentions for a hadith of any substance. Missing a
   real subject costs more than including a minor one, because salience already
   says which is which.

EXAMPLES

"ما العقل قال ما عبد به الرحمن و اكتسب به الجنان ... فالذي كان في معاوية فقال تلك النكراء"
  mentions: العقل/concept/0.95 evidence "ما العقل قال ما عبد به الرحمن"
            النكراء/concept/0.8 evidence "تلك النكراء تلك الشيطنة"
            معاوية/person/0.4 evidence "فالذي كان في معاوية"
            العبادة/concept/0.3 evidence "ما عبد به الرحمن"
  The Imam answering is a narrator, not a mention.

"صديق كل امرئ عقله و عدوه جهله"
  mentions: العقل/concept/0.9, الجهل/concept/0.9
  NOT عقل المرء or جهل المرء -- the possessor is grammar, not subject.

"لما خلق الله العقل استنطقه ثم قال له أقبل فأقبل"
  mentions: خلق العقل/concept/0.9 evidence "لما خلق الله العقل استنطقه"
            العقل/concept/0.7
  Here the compound IS the subject: this narration is about the creation of the
  intellect, which other narrations also recount.

"إن عندنا قوما لهم محبة و ليست لهم تلك العزيمة ... إنما قال الله فاعتبروا يا أولي الأبصار"
  mentions: محبة أهل البيت/concept/0.8 evidence "إن عندنا قوما لهم محبة"
            العزيمة/concept/0.6 evidence "ليست لهم تلك العزيمة"
  quotes:   {text: "فاعتبروا يا أولي الأبصار", kind: "quran"}
  No verse number. أبو الحسن is being asked, so he is a narrator.

JSON must be complete and compact: copy each Arabic matn once, do not repeat
sentences, do not pad translations. Every numbered hadith needs non-empty
hadith_fa, hadith_en, ravis and mentions. Continuation fragments may omit
translations only when the fragment is isnad with no matn yet.
Never fabricate a hadith.
"""

UNIFY_PROMPT = """\
From THIS assembled Arabic hadith only (ignore chapter titles and any other
narration), return Persian (hadith_fa), English (hadith_en), ravis, mentions
and quotes.

A mention is something the narration is ABOUT, in the matn's own words, with a
type (concept | person | place | group | event | work), a salience from 0.0 to
1.0, and the verbatim `evidence` span it came from. You are not choosing from a
vocabulary; identity is resolved elsewhere against the whole corpus.

Say it plainly -- العقل, not عقل المرء. Ground every person, place, event, group
and work literally in the matn. Narrators and the speaker being quoted go in
ravis and never in mentions. Never index something the matn denies. Put Qur'anic
quotations in `quotes` as the quoted words only, with no sura name and no verse
number.

Give 3 to 8 mentions. Never return an empty list.
"""

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


def unify_prompt() -> str:
    return UNIFY_PROMPT
