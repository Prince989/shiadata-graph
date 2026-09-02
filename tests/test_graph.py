from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.agents.errors import AllKeysExhausted, RateLimited
from src.agents.gemini import GeminiAgent
from src.agents.key_pool import FailureKind, KeyPool, LlmKey
from src.core.edge_classifier import build_tag_buckets
from src.core.phase2 import remap_existing_hadith_tags
from src.core.vector_engine import concepts_for_chunk, cosine_similarity, pairs_above_threshold
from src.extractors.chunkers import hadith_units, split_hadith_page
from src.extractors.epub_parser import ParsedUnit, parse_epub, strip_html
from src.extractors.txt_parser import parse_txt
from src.models import HadithExtraction, HadithPageExtraction, HistoryExtraction, TafsirExtraction
from src.pipelines.llm_processor import phase1_filename, process_unit, system_prompt
from src.pipelines.ontology import canonicalize_tag, is_grouping_label, remap_hadith_payload
from src.state_manager import ChunkRecord, ChunkStatus, StateManager

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def state(tmp_path: Path) -> StateManager:
    return StateManager(tmp_path / "state.db")


def test_txt_parser_preserves_arabic_and_banners():
    units = parse_txt(FIXTURES / "sample_hadith.txt")
    assert len(units) == 2
    assert units[0].locator == "جلد 1 - صفحه 1"
    assert "بِسْمِ" in units[0].text
    assert "كظم الغيظ" in units[1].text


def test_txt_parser_splits_mizan_ayah_headers():
    units = parse_txt(FIXTURES / "sample_mizan.txt")
    assert units[0].locator == "سوره 1 - آیات 1-5"
    assert "الحمد" in units[0].text


def test_hadith_split_kafi_does_not_glue_bab_title():
    page = """كِتَابُ الْعَقْلِ وَ الْجَهْلِ

1-
أَخْبَرَنَا أَبُو جَعْفَرٍ قَالَ لَمَّا خَلَقَ اللَّهُ الْعَقْلَ.

2- عَلِيُّ بْنُ مُحَمَّدٍ عَنْ سَهْلٍ قَالَ هَبَطَ جَبْرَئِيلُ.
"""
    pieces = split_hadith_page(page)
    assert len(pieces) == 2
    assert pieces[0][0].startswith("1")
    assert "كِتَابُ الْعَقْلِ" not in pieces[0][1]
    assert "أَخْبَرَنَا" in pieces[0][1]
    assert "هَبَطَ جَبْرَئِيلُ" in pieces[1][1]


def test_hadith_split_wasail_bracket_headers():
    page = """أبواب احكام العشرة
١ ـ باب وجوب عشرة الناس

[ ١٥٤٩٥ ] ١ ـ محمد بن يعقوب قال تؤدون الأمانة إليهم.

[ ١٥٤٩٦ ] ٢ ـ وبالإسناد عن صفوان قال اوصيكم بتقوى الله.

١ ـ الكافي ٢ : ٤٦٤
"""
    pieces = split_hadith_page(page)
    assert len(pieces) == 2
    assert "١٥٤٩٥" in pieces[0][0] or "15495" in pieces[0][0]
    assert "تؤدون الأمانة" in pieces[0][1]
    assert "اوصيكم بتقوى" in pieces[1][1]
    assert not any("باب وجوب" in body and "محمد بن يعقوب" not in body for _, body in pieces)


KAFI1 = Path(__file__).resolve().parents[1] / "data" / "raw_epubs" / "hadith" / "al-kafi-1.txt"


def test_real_kafi_pages_11_12_have_multiple_hadith_starts():
    if not KAFI1.exists():
        pytest.skip("al-kafi-1.txt not in raw_epubs")
    pages = parse_txt(KAFI1)
    page11 = next(u for u in pages if u.locator == "جلد 1 - صفحه 11")
    page12 = next(u for u in pages if u.locator == "جلد 1 - صفحه 12")
    starts11 = [token for token, _ in split_hadith_page(page11.text)]
    starts12 = [token for token, _ in split_hadith_page(page12.text)]
    assert starts11 == ["3 -", "4 -", "5 -", "6 -", "7 -", "8-"]
    assert starts12 == ["9 -", "10 -", "11 -"]


def test_hadith_units_keeps_multi_hadith_pages_separate():
    if not KAFI1.exists():
        pytest.skip("al-kafi-1.txt not in raw_epubs")
    pages = parse_txt(KAFI1)[:25]
    units = hadith_units(pages)
    by_loc = {u.locator: u for u in units}
    assert "جلد 1 - صفحه 11" in by_loc
    assert "جلد 1 - صفحه 12" in by_loc
    assert len(split_hadith_page(by_loc["جلد 1 - صفحه 11"].text)) == 6
    assert len(split_hadith_page(by_loc["جلد 1 - صفحه 12"].text)) == 3
    assert "…" not in by_loc["جلد 1 - صفحه 11"].locator


def test_hadith_units_skips_intro_and_keeps_continuation_page():
    pages = [
        ParsedUnit("جلد 1 - صفحه 1", "مقدمة المؤلف بدون رقم حديث هنا " * 3, "al-kafi-1.txt"),
        ParsedUnit(
            "جلد 1 - صفحه 10",
            "كِتَابُ الْعَقْلِ\n\n1- أَخْبَرَنَا أَبُو جَعْفَرٍ قَالَ لَمَّا خَلَقَ اللَّهُ الْعَقْلَ.\n",
            "al-kafi-1.txt",
        ),
        ParsedUnit(
            "جلد 1 - صفحه 11",
            "تتمة المتن من الصفحة السابقة دون رقم جديد في أول السطر.",
            "al-kafi-1.txt",
        ),
    ]
    units = hadith_units(pages)
    assert len(units) == 2
    assert units[0].locator == "جلد 1 - صفحه 10"
    assert units[1].locator == "جلد 1 - صفحه 11"
    assert "أَخْبَرَنَا" in units[0].text
    assert "تتمة المتن" in units[1].text


def test_html_strip_drops_tags_keeps_persian():
    html = "<html><body><h1>باب</h1><p>صبر و <b>توکل</b></p><script>x</script></body></html>"
    text = strip_html(html)
    assert "باب" in text
    assert "توکل" in text
    assert "<p>" not in text
    assert "script" not in text.lower() or "x" not in text


def test_epub_parser_reads_fixture(tmp_path: Path):
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("id1")
    book.set_title("Test")
    book.set_language("ar")
    chapter = epub.EpubHtml(title="باب الرفق", file_name="c1.xhtml", lang="ar")
    chapter.content = "<h1>باب الرفق</h1><p>الرفق يمن والخرق شؤم</p>"
    book.add_item(chapter)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", chapter]
    path = tmp_path / "sample.epub"
    epub.write_epub(str(path), book)
    units = parse_epub(path)
    assert units
    assert "الرفق" in units[0].text
    assert "<p>" not in units[0].text


def test_key_rotation_on_429(state: StateManager):
    pool = KeyPool(state, keys=["key-a", "key-b"])
    calls: list[str] = []

    def fake_generate(*, key: LlmKey, prompt: str, system, model, schema):
        calls.append(key.secret)
        if key.secret == "key-a":
            raise RateLimited("429 rate limit")
        return json.dumps(
            {
                "hadith": "x",
                "hadith_fa": "ی",
                "hadith_en": "x",
                "tags": [],
                "ravis": [],
            },
            ensure_ascii=False,
        )

    agent = GeminiAgent(state, key_pool=pool, generate_fn=fake_generate)
    result = agent.complete_structured("hi", HadithExtraction)
    assert result.hadith == "x"
    assert calls[0] == "key-a"
    assert "key-b" in calls


def test_all_keys_exhausted_leaves_chunks_pending(state: StateManager):
    pool = KeyPool(state, keys=["only"])
    k = pool.acquire()
    pool.report_failure(k, FailureKind.QUOTA_EXHAUSTED)
    with pytest.raises(AllKeysExhausted):
        pool.acquire()
    state.upsert_chunks(
        [
            {
                "id": "abc",
                "book_id": "al-kafi",
                "pipeline": "hadith",
                "locator": "p1",
                "source_path": "x.txt",
                "text": "hadith text long enough",
            }
        ]
    )
    assert state.get_chunk("abc").status == ChunkStatus.PENDING


def test_pydantic_hadith_schema():
    item = HadithExtraction.model_validate(
        {
            "hadith": "الرفق يمن",
            "hadith_fa": "مدارا مبارک است",
            "hadith_en": "Gentleness is blessed",
            "tags": ["الرفق"],
            "ravis": ["زرارة"],
        }
    )
    assert item.tags == ["الرفق"]
    page = HadithPageExtraction.model_validate(
        {
            "page": "جلد 1 - صفحه 11",
            "hadiths": [
                {
                    "marker": "3 -",
                    "hadith": "أ",
                    "hadith_fa": "آ",
                    "hadith_en": "a",
                    "tags": [],
                    "ravis": [],
                },
                {
                    "marker": "4 -",
                    "hadith": "ب",
                    "hadith_fa": "ب",
                    "hadith_en": "b",
                    "tags": ["العقل"],
                    "ravis": ["زرارة"],
                },
            ],
        }
    )
    assert len(page.hadiths) == 2
    with pytest.raises(ValidationError):
        HadithExtraction.model_validate({"hadith": "only"})


def test_process_unit_writes_page_hadiths_array(tmp_path: Path, state: StateManager):
    calls: list[type] = []

    def fake_generate(*, key, prompt, system, model, schema):
        calls.append(schema)
        return json.dumps(
            {
                "page": "جلد 1 - صفحه 11",
                "hadiths": [
                    {
                        "marker": "3 -",
                        "hadith": "حديث ثلاثة",
                        "hadith_fa": "سه",
                        "hadith_en": "three",
                        "tags": ["العقل"],
                        "ravis": ["علي بن إبراهيم"],
                    },
                    {
                        "marker": "4 -",
                        "hadith": "حديث أربعة",
                        "hadith_fa": "چهار",
                        "hadith_en": "four",
                        "tags": ["العلم"],
                        "ravis": ["محمد بن يحيى"],
                    },
                ],
            },
            ensure_ascii=False,
        )

    agent = GeminiAgent(state, key_pool=KeyPool(state, keys=["test-key"]), generate_fn=fake_generate)
    unit = ParsedUnit(
        "جلد 1 - صفحه 11",
        "3 - حديث ثلاثة بما يكفي من الحروف.\n4 - حديث أربعة بما يكفي من الحروف.",
        str(KAFI1 if KAFI1.exists() else tmp_path / "al-kafi-1.txt"),
    )
    status = process_unit(
        agent,
        state,
        book_id="hadith",
        pipeline="hadith",
        unit=unit,
        output_dir=tmp_path / "phase1",
        min_chars=10,
    )
    assert status == ChunkStatus.PROCESSED_PHASE1
    assert calls[0] is HadithPageExtraction
    written = tmp_path / "phase1" / "hadith" / phase1_filename(unit.source_path, unit.locator, "x")
    data = json.loads(written.read_text(encoding="utf-8"))
    assert isinstance(data["hadiths"], list)
    assert len(data["hadiths"]) == 2
    assert data["hadiths"][0]["marker"] == "3 -"


def test_pydantic_tafsir_and_history_schemas():
    tafsir = TafsirExtraction.model_validate(
        {
            "ayah_anchor": "سوره 1 - آیات 1-5",
            "core_concepts": ["الحمد"],
            "referenced_hadith": "",
            "summary_fa": "خط یک\nخط دو",
            "tafsir_chunk": "متن تفسیر",
        }
    )
    history = HistoryExtraction.model_validate(
        {
            "events": [
                {
                    "event_title": "قدوم علي",
                    "characters_involved": ["علي"],
                    "historical_concepts": ["الكوفة"],
                    "historical_chunk": "paragraphs",
                }
            ]
        }
    )
    assert tafsir.ayah_anchor.startswith("سوره")
    assert history.events[0].event_title


def test_cosine_keeps_pairs_above_half():
    ids = ["a", "b", "c"]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    pairs = pairs_above_threshold(ids, vectors, 0.5)
    pair_ids = {(left, right) for left, right, _ in pairs}
    assert ("a", "b") in pair_ids
    assert ("a", "c") not in pair_ids
    assert cosine_similarity(vectors[0], vectors[1]) > 0.5
    assert cosine_similarity(vectors[0], vectors[2]) < 0.5


def test_canonicalize_alias_and_parent_not_grouping():
    assert canonicalize_tag("قتل النفس") == "الانتحار"
    assert canonicalize_tag("عذاب القبر") == "عذاب البرزخ"
    assert is_grouping_label("الانتحار") is True
    assert is_grouping_label("المعاد") is False
    assert is_grouping_label("الإيمان") is False
    old = remap_hadith_payload({"hadiths": [{"hadith": "x", "tags": ["الإيمان"]}]})
    assert old["hadiths"][0]["tags"] == ["الإيمان"]


def test_hadith_tag_prompt_is_claim_grain():
    text = system_prompt("hadith")
    assert "at most one extra tag" not in text
    assert "Base Ontology" not in text
    assert "الانتحار" in text
    assert "عذاب البرزخ" in text
    assert "kitāb" in text.lower() or "kitab" in text.lower() or "bāb" in text or "باب" in text


def _hadith_chunk(cid: str, book: str, tags: list[str]) -> ChunkRecord:
    payload = {
        "page": "p",
        "hadiths": [
            {
                "marker": "1-",
                "hadith": "متن",
                "hadith_fa": "متن",
                "hadith_en": "text",
                "tags": tags,
                "ravis": [],
            }
        ],
    }
    return ChunkRecord(
        id=cid,
        book_id=book,
        pipeline="hadith",
        locator="p",
        source_path="x.txt",
        text="متن طويل بما يكفي",
        status=ChunkStatus.EMBEDDED,
        payload_json=json.dumps(payload, ensure_ascii=False),
    )


def test_tag_buckets_cross_book_and_keep_singleton():
    chunks = [
        _hadith_chunk("a", "hadith", ["الانتحار", "عذاب البرزخ"]),
        _hadith_chunk("b", "vasael-o-shia", ["قتل النفس"]),
        _hadith_chunk("c", "hadith", ["موضوع فريد"]),
        _hadith_chunk("d", "hadith", ["الإيمان", "المعاد"]),
    ]
    buckets = build_tag_buckets(chunks)
    assert set(buckets["الانتحار"]) == {"a", "b"}
    assert "c" in buckets["موضوع فريد"]
    assert "الإيمان" not in buckets
    assert "المعاد" not in buckets
    assert concepts_for_chunk(chunks[3]) == []


def test_extreme_df_skipped_on_large_set_only():
    popular = [_hadith_chunk(f"p{i}", "hadith", ["الانتحار"]) for i in range(8)]
    rare = [_hadith_chunk("r0", "hadith", ["موضوع فريد"])]
    hesab = [_hadith_chunk(f"h{i}", "hadith", ["الحساب"]) for i in range(2)]
    other = [_hadith_chunk(f"z{i}", "hadith", [f"موضوع-{i}"]) for i in range(10)]
    buckets = build_tag_buckets(
        popular + rare + hesab + other, max_df_ratio=0.15, min_chunks_for_df=20
    )
    assert "الانتحار" not in buckets
    assert "موضوع فريد" in buckets
    assert "الحساب" in buckets


def test_remap_existing_tags_without_gemini(tmp_path: Path, state: StateManager):
    payload = {
        "page": "جلد 1 - صفحه 1",
        "hadiths": [{"hadith": "x", "hadith_fa": "x", "hadith_en": "x", "tags": ["قتل النفس"], "ravis": []}],
    }
    cid = "chunk-remap-1"
    state.upsert_chunks(
        [
            {
                "id": cid,
                "book_id": "hadith",
                "pipeline": "hadith",
                "locator": "جلد 1 - صفحه 1",
                "source_path": str(tmp_path / "al-kafi-1.txt"),
                "text": "hadith text long enough",
            }
        ]
    )
    state.mark(cid, ChunkStatus.PROCESSED_PHASE1, payload=payload)
    out = tmp_path / "phase1"
    dest = out / "hadith"
    dest.mkdir(parents=True)
    path = dest / phase1_filename(str(tmp_path / "al-kafi-1.txt"), "جلد 1 - صفحه 1", cid)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    n = remap_existing_hadith_tags(state, output_dir=out)
    assert n == 1
    stored = state.get_chunk(cid).payload()
    assert stored["hadiths"][0]["tags"] == ["الانتحار"]
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["hadiths"][0]["tags"] == ["الانتحار"]


def test_process_unit_canonicalizes_alias_tags(tmp_path: Path, state: StateManager):
    def fake_generate(*, key, prompt, system, model, schema):
        return json.dumps(
            {
                "page": "جلد 1 - صفحه 11",
                "hadiths": [
                    {
                        "marker": "3 -",
                        "hadith": "حديث ثلاثة بما يكفي",
                        "hadith_fa": "سه",
                        "hadith_en": "three",
                        "tags": ["قتل النفس", "عذاب القبر"],
                        "ravis": [],
                    }
                ],
            },
            ensure_ascii=False,
        )

    agent = GeminiAgent(state, key_pool=KeyPool(state, keys=["test-key"]), generate_fn=fake_generate)
    unit = ParsedUnit(
        "جلد 1 - صفحه 11",
        "3 - حديث ثلاثة بما يكفي من الحروف للتجاوز.",
        str(tmp_path / "al-kafi-1.txt"),
    )
    process_unit(
        agent,
        state,
        book_id="hadith",
        pipeline="hadith",
        unit=unit,
        output_dir=tmp_path / "phase1",
        min_chars=10,
    )
    written = tmp_path / "phase1" / "hadith" / phase1_filename(unit.source_path, unit.locator, "x")
    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["hadiths"][0]["tags"] == ["الانتحار", "عذاب البرزخ"]

