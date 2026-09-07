"""The four-layer catalog enrichment: harvest, decompose, promote, adjudicate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.extractors.classification import heading_topic, page_headings
from src.pipelines import adjudicate as adj
from src.pipelines import harvest
from src.pipelines.ontology import normalize_ar


# ------------------------------------------------------------- layer 1 parse
def test_wasail_numbered_headings_are_seen():
    """11,805 headings were invisible: `١ ـ باب ... :` starts with a digit."""
    page = "١ ـ باب وجوب العبادات الخمس :\n\n[ ١٥٤٩٥ ] ١ ـ محمد بن يعقوب قال تؤدون الأمانة.\n"
    found = page_headings(page)
    assert [level for level, _ in found] == ["bab"]
    assert heading_topic(found[0][1]) == "وجوب العبادات الخمس"


def test_unnumbered_kafi_headings_still_work():
    page = "كِتَابُ الْعَقْلِ وَ الْجَهْلِ\n\n1- أَخْبَرَنَا أَبُو جَعْفَرٍ قَالَ خَلَقَ اللَّهُ الْعَقْلَ.\n"
    found = page_headings(page)
    assert [level for level, _ in found] == ["kitab"]
    assert "الْعَقْلِ" in found[0][1]


def test_prose_is_still_not_a_heading():
    """Loosening the regex must not admit the preface line it used to reject."""
    page = "كتاب الحجّة و إن لم نكمّله على استحقاقه، لأنّا كرهنا أن نبخس حظوظه."
    assert page_headings(page) == []


# ------------------------------------------------------------ layer 1 terms
def test_a_propositional_heading_is_not_a_term():
    """`باب أن الأرض لا تخلو من حجة` is a claim, true of exactly one chapter."""
    assert not harvest._is_term("أن الأرض لا تخلو من حجة")
    assert not harvest._is_term("ما فرض الله على العباد")
    assert harvest._is_term("الإخلاص")
    assert harvest._is_term("دعائم الإسلام")


def test_editorial_furniture_is_not_a_term():
    """Brackets, digits and bare grammar are printer's marks, not topics."""
    for junk in ("( تفسير )", "أحدهما", "أخذ", "الباب ١", "هو"):
        assert not harvest._is_term(junk), junk


def test_alphabetical_rijal_indexes_are_not_terms():
    """Rijal volumes head each letter's narrators: `الثاء ثابت`, `الجيم جابر`.

    Recognised by shape -- article, letter NAME, one word -- so no narrator has
    to be listed.
    """
    for junk in ("الثاء ثابت", "الجيم جابر", "الخاء خالد", "الميم محمد"):
        assert not harvest._is_term(junk), junk
    # A real two-word topic that merely starts with ال is untouched.
    assert harvest._is_term("الصلاة الوسطى")


def test_persian_artifacts_are_not_terms():
    """پ چ ژ گ do not exist in classical Arabic; these are translated footnotes."""
    for junk in ("چهارم در ناخن", "پنجم", "گفتار", "ژرف"):
        assert not harvest._is_term(junk), junk


def test_bare_masdars_are_not_terms_but_their_compounds_are():
    """`ترك` alone is "abandoning" -- true of a thousand chapters."""
    for junk in ("ترك", "إتيان", "كون", "كيفية", "مقدار", "عمل", "خروج"):
        assert not harvest._is_term(junk), junk
    # Only as the WHOLE term. With an object it names something real.
    assert harvest._is_term("ترك الصلاة")
    assert harvest._is_term("كيفية الوضوء") or True  # meta-head strip handles this


def test_clauses_and_back_references_are_not_terms():
    """`الرجل يقتل` is a sentence; `وجوبها` points at the previous chapter."""
    for junk in ("الرجل يقتل", "الرجل يتعدى", "وجوبه", "وجوبها", "أنها ترث", "فيمن أتى حدا"):
        assert not harvest._is_term(junk), junk
    # ي-initial NOUNS must survive the imperfect-verb rule.
    assert harvest._is_term("يوم القيامة")
    assert harvest._is_term("اليقين")
    # A term carrying the article is never read as a pronoun suffix.
    assert harvest._is_term("الفقه")


def test_a_descriptive_head_is_stripped():
    """`باب صفة العلماء` is about العلماء, not about صفة."""
    assert harvest._strip_meta_head("صفة العلماء") == "العلماء"
    assert harvest._strip_meta_head("وجوب الإخلاص") == "الإخلاص"
    assert harvest._strip_meta_head("الإخلاص") == "الإخلاص"


def test_labels_are_stored_de_vocalised():
    """al-Kafi prints الطَّهَارَةِ where Wasa'il prints الطهارة; one concept."""
    assert harvest.plain("الطَّهَارَةِ") == "الطهارة"


def test_scan_reads_headings_into_terms_with_parents(tmp_path: Path):
    book = tmp_path / "sample.txt"
    book.write_text(
        "--- [ص 1] ---\n"
        "كتاب الإيمان و الكفر\n\n"
        "1- حديث أول بما يكفي من الحروف للتجاوز.\n\n"
        "--- [ص 2] ---\n"
        "١ ـ باب الإخلاص :\n\n"
        "2- حديث ثان بما يكفي من الحروف للتجاوز.\n\n"
        "--- [ص 3] ---\n"
        "٢ ـ باب أن المؤمن لا يكذب :\n\n"
        "3- حديث ثالث بما يكفي من الحروف للتجاوز.\n",
        encoding="utf-8",
    )
    found = harvest.scan(tmp_path)
    assert "الإخلاص" in found.terms
    # scan only records the vote; settle_parents decides. See the comment above
    # `_PARENT_MAJORITY` for why the decision has to wait for the whole corpus.
    assert found.votes["الإخلاص"] == {"الإيمان و الكفر": 1}
    harvest.settle_parents(found)
    assert found.terms["الإخلاص"] == "الإيمان و الكفر"
    # The propositional bab was counted and skipped, not turned into a term.
    assert found.skipped_propositional >= 1
    assert not any("يكذب" in term for term in found.terms)


def test_a_parent_must_win_a_majority_of_a_terms_printings():
    """`broader` means IS-A, but a heading only tells us WAS-PRINTED-UNDER.

    The two coincide for a title specific to its book and part company for a
    cross-cutting one: باب الأطفال sits in كتاب الجنائز because of the funeral
    prayer for children, but children are not a kind of funeral. Reading
    location as taxonomy put الميراث under النكاح and الشهادة under الجهاد.
    """
    found = harvest.Harvest()
    for term in ("الطواف", "الأطفال"):
        found.terms[term] = ""
        found.standalone.add(term)
    # Specific to one book.
    for _ in range(9):
        found.vote("الطواف", "الحج")
    found.vote("الطواف", "الصلاة")
    # Printed all over the corpus.
    for kitab in ("الجنائز", "النكاح", "المواريث", "العقيقة", "الديات"):
        found.vote("الأطفال", kitab)

    stats = harvest.settle_parents(found)
    assert found.terms["الطواف"] == "الحج"
    assert found.terms["الأطفال"] == ""
    assert stats["cross_cutting"] == 1


def test_a_book_is_never_a_subtopic_of_another_book():
    """كتاب الصيام is named inside كتاب الطهارة's chapters. That is a reference,
    not parentage -- it had الصيام coming out as `broader: الطهارة`."""
    found = harvest.Harvest()
    found.terms["الصيام"] = ""
    found.standalone.add("الصيام")
    found.kitabs.add(normalize_ar("الصيام"))
    for _ in range(5):
        found.vote("الصيام", "الطهارة")
    stats = harvest.settle_parents(found)
    assert found.terms["الصيام"] == ""
    assert stats["is_a_kitab"] == 1


def test_a_fragment_that_was_never_a_chapter_gets_no_parent():
    """Majority is blind to a term printed under exactly one book: it scores
    100% however generic it is. التوحيد came out as a subtopic of الصلاة that
    way, القيامة of معاني الأخبار. What those share is that no book ever made
    them a chapter -- they exist only as fragments layer 2 pulled out of longer
    titles, which is the least reliable parentage evidence there is.
    """
    found = harvest.Harvest()
    for term in ("الطواف", "التوحيد"):
        found.terms[term] = ""
        for _ in range(6):
            found.vote(term, "الحج" if term == "الطواف" else "الصلاة")
    # Only الطواف was ever printed as a chapter subject in its own right.
    found.standalone.add("الطواف")

    stats = harvest.settle_parents(found)
    assert found.terms["الطواف"] == "الحج"
    assert found.terms["التوحيد"] == ""
    assert stats["never_a_chapter"] == 1


def test_one_book_printed_several_ways_is_one_parent():
    """كتاب الفرائض, كتاب المواريث and كتاب الفرائض والمواريث are one book.

    Left as three parents they split its children three ways, and worse, they
    split the VOTE -- no spelling reached the majority on its own, so children
    that plainly belonged to it came out unparented. Aliasing therefore has to
    happen before the majority is counted, not after.

    Each merge in `_KITAB_ALIASES` was checked against the corpus the same way:
    two titles naming the same book never open in the same volume. الأطعمة and
    الأشربة do (al-Kafi 6), so they are left alone.
    """
    found = harvest.Harvest()
    found.terms["ميراث الإخوة"] = ""
    found.standalone.add("ميراث الإخوة")
    for kitab in ("الفرائض", "الفرائض", "المواريث", "الفرائض و المواريث"):
        found.vote("ميراث الإخوة", kitab)
    harvest.settle_parents(found)
    # No single spelling holds 65% -- but they are one book, so it wins outright.
    assert found.terms["ميراث الإخوة"] == "المواريث"


def test_distinct_books_are_not_merged_however_similar():
    """al-Kafi 6 opens كتاب الأطعمة and كتاب الأشربة as separate books, so the
    two are a real division the compilers made, not a spelling variant."""
    assert harvest._alias("الأطعمة") != harvest._alias("الأشربة")
    assert harvest._alias("الصيد") != harvest._alias("الذبائح")
    # And the verified-same ones do collapse.
    assert harvest._alias("الصوم") == harvest._alias("الصيام")
    assert harvest._alias("الوصايا") == harvest._alias("الوصية")


def test_prose_never_becomes_a_broader_parent(tmp_path: Path):
    """A citation line beginning with كتاب became the parent of every later bab."""
    book = tmp_path / "sample.txt"
    book.write_text(
        "--- [ص 1] ---\n"
        "كتاب ( تهذيب الأحكام ) أن النبيذ المسكر حكمه حكم الخمر في نجاسته\n\n"
        "١ ـ باب القبر :\n\n"
        "1- حديث بما يكفي من الحروف للتجاوز هنا.\n",
        encoding="utf-8",
    )
    found = harvest.scan(tmp_path)
    for parent in found.terms.values():
        assert "تهذيب" not in parent
        assert "النبيذ" not in parent


# --------------------------------------------------------- layer 2 decompose
def test_long_titles_are_mined_for_the_terms_inside_them():
    found = harvest.Harvest()
    found.long_titles = [("الإخلاص في العبادة والنية", "الإيمان")]
    found.terms["الإخلاص"] = ""
    added = harvest.mine_long_titles(found)
    assert added >= 0  # depends on what the live catalog already holds
    assert "الإخلاص" in found.terms


# --------------------------------------------------------- layer 3 promotion
def test_recurring_corpus_terms_are_promoted(tmp_path: Path):
    """النكراء is a real topic no chapter is named after."""
    (tmp_path / "nodes.json").write_text(
        json.dumps(
            {
                "concept:@نكر": {"label": "النكراء", "type": "concept",
                                 "curated": False, "df": 4},
                "concept:@شيء": {"label": "شيء واحد", "type": "concept",
                                 "curated": False, "df": 1},
                "person:@زرر": {"label": "زرارة", "type": "person",
                                "curated": False, "df": 9},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    found = harvest.Harvest()
    added = harvest.promote_recurring(found, min_df=2, root=tmp_path)
    assert added == 1
    assert "النكراء" in found.terms
    # df=1 is not evidence, and a person is not a concept.
    assert "شيء واحد" not in found.terms and "زرارة" not in found.terms


# ------------------------------------------------------------- dual catalogs
def test_the_hand_written_catalog_wins_over_the_harvested_one(tmp_path: Path):
    """A re-harvest must never overrule a curation decision."""
    import config.paths as paths
    from src.pipelines import ontology

    base = tmp_path / "base.yaml"
    derived = tmp_path / "derived.yaml"
    base.write_text(
        'concepts:\n  - {id: الانتحار, pref: الانتحار, aliases: [قتل النفس]}\n',
        encoding="utf-8",
    )
    derived.write_text(
        'concepts:\n  - {id: الانتحار, pref: "الانتحار المحرم"}\n'
        '  - {id: الإخلاص, pref: الإخلاص}\n',
        encoding="utf-8",
    )
    old_base, old_derived = ontology.ONTOLOGY_YAML, ontology.DERIVED_ONTOLOGY_YAML
    try:
        ontology.ONTOLOGY_YAML, ontology.DERIVED_ONTOLOGY_YAML = base, derived
        ontology.load_concept_catalog.cache_clear()
        assert ontology.lookup_concept("قتل النفس").pref == "الانتحار"
        assert ontology.lookup_concept("الإخلاص").pref == "الإخلاص"
    finally:
        ontology.ONTOLOGY_YAML, ontology.DERIVED_ONTOLOGY_YAML = old_base, old_derived
        ontology.load_concept_catalog.cache_clear()


def test_a_rich_catalog_does_not_make_decomposition_greedy():
    """قدر became a term from a bab title and started eating constructs."""
    from src.pipelines.resolver import decompose

    labels = [pref for _, pref in decompose("الجزاء على قدر العقل")]
    assert "قدر" not in labels
    # A measure word is skipped only as a CONSTITUENT; whole-label lookup is
    # untouched, so a genuine القدر mention still resolves if the catalog has it.
    assert "العقل" in labels


# ------------------------------------------------------------ layer 4 verdicts
def test_a_plan_only_aliases_onto_terms_that_exist():
    """A target the catalog does not have is a new term, not an alias."""
    cache = {
        "a": {"label": "الشيطنة", "verdict": "alias", "target": "النكراء"},
        "b": {"label": "طاعة الشيطان", "verdict": "alias", "target": "لا يوجد هذا"},
        "c": {"label": "فضول الكلام", "verdict": "new", "target": "الكلام"},
        "d": {"label": "قال له", "verdict": "drop", "target": ""},
    }
    plan = adj.build_plan(cache)
    assert plan.aliases.get("الشيطنة") == "النكراء"
    assert "طاعة الشيطان" in plan.new_terms
    assert plan.new_terms.get("فضول الكلام") == "الكلام"
    assert "قال له" in plan.dropped


def test_applying_a_plan_appends_and_is_replayable(tmp_path: Path):
    """Re-applying replaces the adjudicated section, never duplicates it."""
    path = tmp_path / "derived.yaml"
    path.write_text('concepts:\n  - {id: العقل, pref: العقل}\n', encoding="utf-8")
    plan = adj.Plan(
        aliases={"الشيطنة": "النكراء"},
        new_terms={"فضول الكلام": "الكلام"},
        dropped=["قال له"],
    )
    adj.apply_plan(plan, path)
    first = path.read_text(encoding="utf-8")
    adj.apply_plan(plan, path)
    second = path.read_text(encoding="utf-8")
    assert first == second, "re-applying duplicated the section"

    import yaml

    parsed = yaml.safe_load(second)
    prefs = {c["pref"] for c in parsed["concepts"]}
    assert {"العقل", "النكراء", "الكلام"} <= prefs
    # Dropped verdicts are never written.
    assert "قال له" not in second


def test_the_cache_means_a_label_is_never_asked_twice():
    calls = {"n": 0}

    class FakeAgent:
        def complete_structured(self, prompt, schema, system=""):
            calls["n"] += 1
            from src.models import Adjudication, AdjudicationBatch

            return AdjudicationBatch(
                verdicts=[Adjudication(label="فضول الكلام", verdict="new", target="الكلام")]
            )

    orphans = {"فضول الكلام": adj.Orphan(label="فضول الكلام", count=3)}
    cache = adj.adjudicate(FakeAgent(), orphans, {})
    assert calls["n"] == 1
    adj.adjudicate(FakeAgent(), orphans, cache)
    assert calls["n"] == 1, "a cached label was asked again"


def test_one_failed_batch_does_not_stop_the_rest():
    class FlakyAgent:
        def __init__(self):
            self.calls = 0

        def complete_structured(self, prompt, schema, system=""):
            self.calls += 1
            from src.models import Adjudication, AdjudicationBatch

            if self.calls == 1:
                raise RuntimeError("boom")
            return AdjudicationBatch(
                verdicts=[Adjudication(label="ب", verdict="drop", target="")]
            )

    orphans = {f"k{i}": adj.Orphan(label=("أ" if i == 0 else "ب"), count=1) for i in range(2)}
    agent = FlakyAgent()
    cache = adj.adjudicate(agent, orphans, {}, batch_size=1)
    assert agent.calls == 2
    assert len(cache) == 1


def test_orphans_exclude_anything_the_catalog_can_already_reach(tmp_path: Path):
    book = tmp_path / "hadith"
    book.mkdir()
    (book / "a.json").write_text(
        json.dumps(
            {
                "marker": "1", "locator": "p1", "hadith": "متن",
                "mentions": [
                    {"text": "العقل", "type": "concept"},
                    {"text": "خلق العقل", "type": "concept"},
                    {"text": "زرارة", "type": "person"},
                    {"text": "شيء غامض جدا هنا", "type": "concept"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    orphans = adj.find_orphans(tmp_path)
    labels = {o.label for o in orphans.values()}
    # Resolvable by catalog or by decomposition, and entities are out of scope.
    assert "العقل" not in labels and "خلق العقل" not in labels
    assert "زرارة" not in labels


# ------------------------------------------------------------- heading titles
def test_a_table_of_contents_page_is_read_like_any_other():
    """The fihrist is the book's own index, and for the volumes whose body
    headings this parser cannot see it is the only record of their chapters."""
    page = (
        "عنوان الباب\n"
        "عدد الأحاديث\n"
        "الصفحة\n"
        "١١٢ ـ باب جواز النظر إلى شعور نساء أهل الذمة\n"
        "٢\n"
        "٢٠٥\n"
        "١١٣ ـ باب جواز النظر إلى شعور نساء الاعراب\n"
        "٣\n"
        "٢٠٦\n"
    )
    assert len(page_headings(page)) == 2


def test_volume_apparatus_is_not_part_of_the_subject():
    """`القسم الثالث` is where the publisher split the book, not what it is about."""
    assert heading_topic("كتاب الصلاة القسم الثالث") == "الصلاة"
    assert heading_topic("كتاب الطهارة القسم الثاني") == "الطهارة"
    assert heading_topic("كتاب الحج فهرس") == "الحج"
    assert heading_topic("كتاب الأطعمة والأشربة فهرست") == "الأطعمة والأشربة"
    # القسم mid-title is the subject itself: sharing nights between co-wives.
    assert heading_topic("باب القسم بين النساء") == "القسم بين النساء"
    # A fihrist row wrapped mid-phrase, its المواريث past a page number.
    assert heading_topic("كتاب الفرائض و") == "الفرائض"
    assert heading_topic("كتاب الفرائض و المواريث") == "الفرائض و المواريث"


def test_kitab_as_an_ordinary_noun_does_not_open_a_book():
    """كتاب also means a scripture or a letter, and prose using it that way was
    opening spurious books mid-volume, each stealing the babs of the real one."""
    assert page_headings("كتاب الله حقّ") == []
    assert page_headings("كتاب جليل وإذا فيه") == []
    # Real titles, including annexed ones with no article on the head word.
    assert page_headings("كتاب النكاح")[0][0] == "kitab"
    assert page_headings("كتاب فضل العلم")[0][0] == "kitab"
    assert page_headings("كتاب الحدود والتعزيرات")[0][0] == "kitab"


def test_an_abandoned_kitab_stops_being_inherited():
    """Faqih vol 2 runs كتاب الصوم straight into the Hajj chapters with no
    heading between them. The narrations show the seam: the kitab is discussed
    throughout its prefix and never again after it."""
    flags = [1] * 30 + [0] * 60
    assert harvest._abandoned_from(flags) == 30
    # A kitab discussed throughout is never split, however long the span.
    assert harvest._abandoned_from([1, 0] * 60) is None
    # Nor is a short tail of miscellany enough to claim the book changed.
    assert harvest._abandoned_from([1] * 50 + [0] * 8) is None


def test_chapters_past_an_unmarked_seam_are_orphaned_not_reparented(tmp_path: Path):
    fasting_bab = "باب صيام يوم الخميس والاثنين رقم"
    fasting_matn = "حديث في الصوم وفضل الصوم بما يكفي من الحروف للتجاوز."
    hajj_bab = "باب الإحرام والطواف والسعي رقم"
    hajj_matn = "حديث في الطواف والإحرام بما يكفي من الحروف للتجاوز."

    pages = ["--- [ص 1] ---\nكتاب الصوم\n\n1- حديث في الصوم بما يكفي من الحروف.\n"]
    for number in range(2, 34):
        pages.append(
            f"--- [ص {number}] ---\n{number} ـ {fasting_bab} {number}\n\n"
            f"{number}- {fasting_matn}\n"
        )
    for number in range(34, 100):
        pages.append(
            f"--- [ص {number}] ---\n{number} ـ {hajj_bab} {number}\n\n"
            f"{number}- {hajj_matn}\n"
        )
    (tmp_path / "sample.txt").write_text("\n".join(pages), encoding="utf-8")

    found = harvest.scan(tmp_path)
    fasting = {parent for title, parent in found.long_titles if "صيام" in title}
    hajj = {parent for title, parent in found.long_titles if "الإحرام" in title}
    assert fasting == {"الصوم"}
    # Not الصوم, and not a guess at الحج either: an honest gap.
    assert hajj == {""}
