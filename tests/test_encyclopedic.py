from src.pipelines.llm_processor import looks_encyclopedic


def _inventory(*labels: str) -> str:
    return "قال " + " و ".join(labels)


def test_dense_short_list_is_encyclopedic_without_named_hadith():
    labels = [
        "الصدق",
        "الكذب",
        "الحياء",
        "القحة",
        "الرحمة",
        "القسوة",
        "العلم",
        "الجهل",
        "الزهد",
        "الرغبة",
        "الرفق",
        "الخرق",
        "الفهم",
        "الحمق",
    ]
    assert looks_encyclopedic({"hadith": _inventory(*labels)})


def test_sermon_with_many_wa_but_long_clauses_is_not_encyclopedic():
    clause = (
        "من عمل بما علم ورثه الله علم ما لم يعلم "
        "و من لم يعمل بما علم كان حجة عليه يوم القيامة"
    )
    matn = "قال " + " و ".join([clause] * 20)
    assert not looks_encyclopedic({"hadith": matn})


def test_short_hadith_is_not_encyclopedic():
    assert not looks_encyclopedic(
        {"hadith": "عن ابي عبد الله قال العقل ما عبد به الرحمن"}
    )


def test_llm_flag_still_wins():
    assert looks_encyclopedic(
        {"hadith": "قال العقل ما عبد به الرحمن", "is_encyclopedic": True}
    )


def test_spanned_page_needs_fewer_short_items():
    labels = [
        "الصدق",
        "الكذب",
        "الحياء",
        "القحة",
        "الرحمة",
        "القسوة",
        "العلم",
        "الجهل",
    ]
    assert looks_encyclopedic(
        {
            "hadith": _inventory(*labels),
            "page_start": "20",
            "page_end": "21",
        }
    )
    assert not looks_encyclopedic({"hadith": _inventory(*labels)})
