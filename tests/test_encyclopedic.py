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


def test_quranic_coordination_inside_sermon_is_not_encyclopedic():
    """Hisham-style: short و-lists exist inside ayat, but not as one inventory run."""
    verse = (
        "ان في خلق السموات و الارض و اختلاف الليل و النهار و الفلك "
        "التي تجري في البحر بما ينفع الناس لايات لقوم يعقلون"
    )
    matn = "قال يا هشام " + (" ثم قال " + verse) * 8
    assert not looks_encyclopedic(
        {
            "hadith": matn,
            "page_start": "13",
            "page_end": "19",
            "arabic_pages": ["a"] * 8,
            "is_encyclopedic": True,  # LLM latch must not force exhaustive
        }
    )


def test_short_hadith_is_not_encyclopedic():
    assert not looks_encyclopedic(
        {"hadith": "عن ابي عبد الله قال العقل ما عبد به الرحمن"}
    )


def test_llm_flag_alone_does_not_trigger():
    assert not looks_encyclopedic(
        {"hadith": "قال العقل ما عبد به الرحمن", "is_encyclopedic": True}
    )


def test_spanned_page_needs_shorter_streak():
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
            "arabic_pages": ["p20", "p21"],
        }
    )
    assert not looks_encyclopedic({"hadith": _inventory(*labels)})
