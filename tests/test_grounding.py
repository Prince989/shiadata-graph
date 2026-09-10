"""Evidence must come from the speech after قال, not the isnad."""

from src.pipelines.grounding import (
    check_mention,
    evidence_supports_mention,
    evidence_usable,
    ground_mentions,
    prefer_evidence_span,
)


_HOSTS = (
    "عِدَّةٌ مِنْ أَصْحَابِنَا عَنْ أَحْمَدَ بْنِ مُحَمَّدٍ عَنْ عَلِيِّ بْنِ "
    "حَدِيدٍ عَنْ سَمَاعَةَ بْنِ مِهْرَانَ قَالَ قُلْتُ لَهُ أَصْلَحَكَ اللَّهُ "
    "الْخَيْرُ وَ الشَّرُّ وَ الْإِيمَانُ وَ الْكُفْرُ وَ الْعَدْلُ وَ الْجَوْرُ"
)


def test_isnad_evidence_is_unusable_even_when_the_span_is_in_the_file():
    mention = {
        "text": "الخير",
        "type": "concept",
        "evidence": "أَحْمَدَ بْنِ مُحَمَّدٍ",
    }
    assert not evidence_usable(mention, _HOSTS)
    assert check_mention(mention, _HOSTS) is None  # term is in the body


def test_isnad_evidence_stripped_but_real_topic_kept():
    kept, rejected = ground_mentions(
        [
            {
                "text": "الخير",
                "type": "concept",
                "salience": 0.9,
                "evidence": "أَحْمَدَ بْنِ مُحَمَّدٍ",
            },
            {
                "text": "العدل",
                "type": "concept",
                "evidence": "عِدَّةٌ مِنْ أَصْحَابِنَا",
            },
        ],
        _HOSTS,
    )
    assert [m["text"] for m in kept] == ["الخير", "العدل"]
    assert all(not m.get("evidence") for m in kept)
    assert rejected == []


def test_isnad_evidence_drops_when_term_absent():
    kept, rejected = ground_mentions(
        [
            {
                "text": "الوسواس",
                "type": "concept",
                "evidence": "أَحْمَدَ بْنِ مُحَمَّدٍ",
            }
        ],
        _HOSTS,
    )
    assert kept == []
    assert rejected == [("الوسواس", "no evidence and term not in matn")]


def test_valid_pair_evidence_in_speech_is_kept():
    mention = {
        "text": "الخير",
        "type": "concept",
        "evidence": "الْخَيْرُ وَ الشَّرُّ",
    }
    assert evidence_usable(mention, _HOSTS)
    kept, rejected = ground_mentions([mention], _HOSTS)
    assert rejected == []
    assert kept[0]["evidence"] == "الْخَيْرُ وَ الشَّرُّ"


def test_inferred_evidence_after_qala_still_counts():
    matn = "عَنْ أَبِي عَبْدِ اللَّهِ ع قَالَ مَا عُبِدَ بِهِ الرَّحْمَنُ"
    mention = {
        "text": "العقل",
        "type": "concept",
        "evidence": "ما عبد به الرحمن",
    }
    assert evidence_usable(mention, matn)
    # Label is not in the speech; inferred evidence in the body is enough.
    assert check_mention(mention, matn) is None


def test_khashya_root_overlap_supports_mention():
    assert evidence_supports_mention(
        "خشية الله",
        "لَمْ يَخَفِ اللَّهَ مَنْ لَمْ يَعْقِلْ عَنِ اللَّهِ",
    )


def test_prefer_supporting_span_over_isnad():
    chosen = prefer_evidence_span(
        "الخير",
        "أَحْمَدَ بْنِ مُحَمَّدٍ",
        "الْخَيْرُ وَ الشَّرُّ",
        matn=_HOSTS,
    )
    assert "خير" in chosen.replace(" ", "") or "الْخَيْرُ" in chosen
