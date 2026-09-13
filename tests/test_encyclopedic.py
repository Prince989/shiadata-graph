from src.pipelines.llm_processor import looks_encyclopedic


def _inventory(*labels: str) -> str:
    return "قال " + " و ".join(labels)


_HOSTS_LABELS = [
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


def test_dense_short_list_is_encyclopedic_without_named_hadith():
    assert looks_encyclopedic({"hadith": _inventory(*_HOSTS_LABELS)})


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


def test_cosmic_list_in_doctrinal_sermon_is_not_encyclopedic():
    """Hadith 35 shape: سمائه و ارضه و شمسه… is not جنود العقل."""
    matn = (
        "عن ابي عبد الله ع في حديث طويل ان اول الامور و مبداها و قوتها "
        "و عمارتها التي لا ينتفع بشيء الا به العقل الذي جعله الله زينه "
        "لخلقه و نورا لهم فبالعقل عرف العباد خالقهم و انهم مخلوقون و انه "
        "المدبر لهم و انهم المدبرون و انه الباقي و هم الفانون و استدلوا "
        "بعقولهم على ما راوا من خلقه من سمائه و ارضه و شمسه و قمره و ليله "
        "و نهاره و بان له و لهم خالقا و مدبرا لم يزل و لا يزول و عرفوا به "
        "الحسن من القبيح و ان الظلمه في الجهل و ان النور في العلم فهذا ما "
        "دلهم عليه العقل قيل له فهل يكتفي العباد بالعقل دون غيره قال ان "
        "العاقل لدلاله عقله علم ان الله هو الحق و انه هو ربه و علم ان "
        "لخالقه محبه و ان له كراهيه و ان له طاعه و ان له معصيه فوجب على "
        "العاقل طلب العلم و الادب"
    )
    assert not looks_encyclopedic(
        {
            "hadith": matn,
            "page_start": "جلد 1 - صفحه 28",
            "page_end": "جلد 1 - صفحه 29",
            "arabic_pages": ["p28", "p29"],
            "is_encyclopedic": True,
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


def test_spanned_page_still_needs_full_inventory_streak():
    eight = _HOSTS_LABELS[:8]
    assert not looks_encyclopedic(
        {
            "hadith": _inventory(*eight),
            "page_start": "20",
            "page_end": "21",
            "arabic_pages": ["p20", "p21"],
        }
    )
    assert looks_encyclopedic(
        {
            "hadith": _inventory(*_HOSTS_LABELS),
            "page_start": "20",
            "page_end": "21",
            "arabic_pages": ["p20", "p21"],
        }
    )
