from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from config.settings import Settings
from src.agents.errors import AllKeysExhausted, RateLimited, StructuredOutputError
from src.agents.gemini import GeminiAgent
from src.agents.key_pool import FailureKind, KeyPool, LlmKey
from src.core.edge_classifier import build_tag_buckets
from src.core.phase2 import remap_existing_hadith_tags
from src.core.vector_engine import concepts_for_chunk, cosine_similarity, pairs_above_threshold
from src.extractors.chunkers import hadith_units, page_prefix_and_starts, split_hadith_page, strip_folklib_footnotes
from src.extractors.epub_parser import ParsedUnit, parse_epub, strip_html
from src.extractors.txt_parser import parse_txt
from src.models import HadithExtraction, HadithPageExtraction, HadithUnify, HistoryExtraction, TafsirExtraction
from src.pipelines.hadith_accumulator import OpenHadith, consume_page
from src.pipelines.llm_processor import phase1_filename, process_unit, system_prompt, unify_assembled_hadith
from src.pipelines.ontology import canonicalize_tag, is_grouping_label, remap_hadith_payload
from src.pipelines.reset import reset_catalog_book
from src.state_manager import ChunkRecord, ChunkStatus, StateManager

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def state(tmp_path: Path) -> StateManager:
    return StateManager(tmp_path / "state.db")


def test_reset_catalog_book_wipes_chunks_buffer_and_json(tmp_path: Path, state: StateManager):
    book_id = "hadith"
    out = tmp_path / "output" / "phase1" / book_id
    out.mkdir(parents=True)
    (out / "al-kafi-1__1.json").write_text("{}", encoding="utf-8")
    state.upsert_chunks(
        [
            {
                "id": "h1",
                "book_id": book_id,
                "pipeline": "hadith",
                "locator": "جلد 1 - صفحه 10",
                "source_path": "al-kafi-1.txt",
                "text": "متن",
            }
        ]
    )
    state.set_hadith_progress(book_id, "al-kafi-1.txt", "جلد 1 - صفحه 10")
    state.set_hadith_buffer(book_id, "al-kafi-1.txt", {"marker": "11 -"})
    stats = reset_catalog_book(state, book_id, output_dir=tmp_path / "output")
    assert stats["chunks_removed"] == 1
    assert stats["json_removed"] == 1
    assert state.counts(book_id) == {}
    assert state.get_hadith_progress(book_id, "al-kafi-1.txt") is None
    assert state.get_hadith_buffer(book_id, "al-kafi-1.txt") is None
    assert not (out / "al-kafi-1__1.json").exists()


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


def test_collect_google_keys_keeps_three_unique_secrets():
    from config.settings import collect_google_keys

    keys = collect_google_keys(
        {
            "GOOGLE_API_KEY": "alpha",
            "GOOGLE_API_KEY1": "alpha",
            "GOOGLE_API_KEY2": "beta",
            "GOOGLE_API_KEY3": "gamma",
            "GOOGLE_API_KEY_4": "delta",
        }
    )
    assert keys == ["alpha", "beta", "gamma", "delta"]


def test_round_robin_cycles_every_configured_key(state: StateManager):
    pool = KeyPool(state, keys=["a", "b", "c"])
    order = [pool.acquire().secret for _ in range(6)]
    assert order == ["a", "b", "c", "a", "b", "c"]
    first = pool.acquire()
    pool.report_failure(first, FailureKind.RATE_LIMITED, 60_000)
    skipped = [pool.acquire().secret for _ in range(4)]
    assert skipped == ["b", "c", "b", "c"]
    assert "a" not in skipped


def test_classify_per_day_429_is_daily_quota():
    from src.agents.gemini import classify_provider_error

    kind, ms = classify_provider_error(
        Exception(
            "429 RESOURCE_EXHAUSTED. You exceeded your current quota. "
            "Quota exceeded for metric: generate_content_free_tier_requests, "
            "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier. "
            "Please retry in 3.295465684s. retryDelay': '3s'"
        )
    )
    assert kind == FailureKind.QUOTA_EXHAUSTED
    assert ms is None


def test_classify_rpm_429_uses_retry_delay():
    from src.agents.gemini import classify_provider_error

    kind, ms = classify_provider_error(
        Exception(
            "429 RESOURCE_EXHAUSTED. Please retry in 12.5s. "
            "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier."
        )
    )
    assert kind == FailureKind.RATE_LIMITED
    assert ms is not None
    assert 12_000 <= ms <= 14_000


def test_acquire_waits_out_short_rate_limit(state: StateManager):
    settings = Settings(key_acquire_wait_max_ms=2_000, google_api_keys=["k"])
    pool = KeyPool(state, settings=settings, keys=["k"])
    key = pool.acquire()
    pool.report_failure(key, FailureKind.RATE_LIMITED, 80)
    again = pool.acquire()
    assert again.secret == "k"


def test_acquire_does_not_wait_on_daily_quota(state: StateManager, monkeypatch):
    monkeypatch.setattr(
        "src.agents.key_pool.time.sleep",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("slept")),
    )
    pool = KeyPool(state, keys=["k"])
    key = pool.acquire()
    pool.report_failure(key, FailureKind.QUOTA_EXHAUSTED)
    with pytest.raises(AllKeysExhausted, match="tomorrow"):
        pool.acquire()


def test_gemini_exits_on_daily_quota_without_retry_loop(state: StateManager):
    def fail(**_kwargs):
        raise RuntimeError(
            "429 RESOURCE_EXHAUSTED. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier. "
            "Please retry in 55s."
        )

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_max_attempts=20),
        key_pool=KeyPool(state, keys=["a", "b"]),
        generate_fn=fail,
    )
    with pytest.raises(AllKeysExhausted, match="tomorrow"):
        agent.complete("x")


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


def test_structured_output_retries_truncated_json(state: StateManager):
    calls = {"n": 0}

    def fake_generate(*, key, prompt, system, model, schema):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"hadith": "unterminated'
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

    settings = Settings(gemini_max_attempts=3, google_api_keys=["k"])
    agent = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    result = agent.complete_structured("hi", HadithExtraction)
    assert result.hadith == "x"
    assert calls["n"] == 2


def test_run_phase1_records_truncated_page_error(
    tmp_path: Path, state: StateManager, monkeypatch: pytest.MonkeyPatch
):
    from src.pipelines import runner as runner_mod
    from src.pipelines.catalog import BookSpec

    raw = tmp_path / "hadith"
    raw.mkdir()
    book = raw / "al-kafi-1.txt"
    long = "متن حديث طويل بما يكفي. " * 8
    book.write_text(
        f"--- [جلد 1 - صفحه 10] ---\n\n1- {long}\n\n--- [جلد 1 - صفحه 11] ---\n\n2- {long}\n",
        encoding="utf-8",
    )

    def fake_generate(*, key, prompt, system, model, schema):
        return '{"page": "p", "hadiths": [{"hadith": "cut'

    settings = Settings(gemini_max_attempts=2, google_api_keys=["k"], skip_min_chars=10)
    agent = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    monkeypatch.setattr(runner_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        runner_mod,
        "resolve_book",
        lambda *a, **k: BookSpec("hadith", "hadith", "", [book]),
    )
    stats = runner_mod.run_phase1("hadith", state, agent, settings=settings, limit=2)
    assert stats["errors"] == 1
    assert stats["processed"] == 0


def test_hadith_limit_and_json_error_do_not_open_next_volume(
    tmp_path: Path, state: StateManager, monkeypatch: pytest.MonkeyPatch
):
    from src.pipelines import runner as runner_mod
    from src.pipelines.catalog import BookSpec

    raw = tmp_path / "hadith"
    raw.mkdir()
    long = "متن حديث طويل بما يكفي. " * 8

    def volume(name: str, vol: int) -> Path:
        path = raw / name
        pages = "\n".join(
            f"--- [جلد {vol} - صفحه {n}] ---\n\n{n}- {long}\n" for n in range(10, 16)
        )
        path.write_text(pages, encoding="utf-8")
        return path

    v1 = volume("al-kafi-1.txt", 1)
    v2 = volume("al-kafi-2.txt", 2)
    v3 = volume("al-kafi-3.txt", 3)
    page_calls: list[str] = []

    def fake_generate(*, key, prompt, system, model, schema):
        if schema is HadithUnify:
            return json.dumps({"tags": [], "ravis": []})
        page_calls.append(prompt.split("Locator:", 1)[-1][:80])
        marker = "10 -"
        for n in range(10, 16):
            if f"صفحه {n}" in prompt:
                marker = f"{n} -"
                break
        return json.dumps(
            {
                "page": "p",
                "hadiths": [
                    {
                        "marker": marker,
                        "hadith": long,
                        "hadith_fa": "ی",
                        "hadith_en": "x",
                        "tags": [],
                        "ravis": [],
                    }
                ],
            },
            ensure_ascii=False,
        )

    settings = Settings(gemini_max_attempts=2, google_api_keys=["k"], skip_min_chars=10)
    agent = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    monkeypatch.setattr(runner_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        runner_mod,
        "resolve_book",
        lambda *a, **k: BookSpec("hadith", "hadith", "", [v1, v2, v3]),
    )
    stats = runner_mod.run_phase1("hadith", state, agent, settings=settings, limit=2)
    assert stats["pages"] == 2
    assert stats["errors"] == 0
    assert len(page_calls) == 2
    assert state.get_hadith_progress("hadith", str(v2)) is None
    assert state.get_hadith_progress("hadith", str(v3)) is None
    assert state.get_hadith_progress("hadith", str(v1)) == "جلد 1 - صفحه 11"

    def boom(*, key, prompt, system, model, schema):
        return '{"page": "p", "hadiths": [{"hadith": "cut'

    agent2 = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=boom,
    )
    stats2 = runner_mod.run_phase1("hadith", state, agent2, settings=settings, limit=20)
    assert stats2["errors"] == 1
    assert state.get_hadith_progress("hadith", str(v2)) is None
    assert state.get_hadith_progress("hadith", str(v3)) is None


def test_page_prefix_keeps_continuation_before_new_marker():
    page = """تتمة الحديث السابق من دون رقم.

13 - علي بن محمد قال العقل غطاء.

14 - عدة من اصحابنا عن احمد قال الفضل جمال.
"""
    leading, starts = page_prefix_and_starts(page)
    assert "تتمة الحديث" in leading
    assert [t for t, _ in starts] == ["13 -", "14 -"]
    assert "تتمة" not in starts[0][1]


def test_strip_folklib_footnotes_drops_editor_notes_keeps_matn():
    page = """11 - عِدَّةٌ مِنْ أَصْحَابِنَا عَنْ أَحْمَدَ قَالَ قَالَ رَسُولُ اللَّهِ ص‌ مَا قَسَمَ اللَّهُ لِلْعِبَادِ شَيْئاً أَفْضَلَ مِنَ الْعَقْلِ فَنَوْمُ الْعَاقِلِ‌
[1] أي: يجازى على اعماله بقدر عقله فكل من كان عقله أكمل كان
ثوابه أجزل( آت)
[2] أي بالوسواس في نيتها أو أفعالهما.
أَفْضَلُ مِنْ سَهَرِ الْجَاهِلِ وَ إِقَامَةُ الْعَاقِلِ أَفْضَلُ مِنْ شُخُوصِ الْجَاهِلِ‌ [1]
وَ مَا يَتَذَكَّرُ إِلَّا أُولُوا الْأَلْبابِ‌ [2] .
"""
    clean = strip_folklib_footnotes(page)
    assert "يجازى" not in clean
    assert "بالوسواس" not in clean
    assert "[1]" not in clean
    assert "أَفْضَلُ مِنْ سَهَرِ الْجَاهِلِ" in clean
    assert "أُولُوا الْأَلْبابِ" in clean
    flushed, buf = consume_page("جلد 1 - صفحه 12", clean, [], None, None)
    assert buf is None
    assert len(flushed) == 1
    assert "يجازى" not in flushed[0]["hadith"]
    assert "سَهَرِ الْجَاهِلِ" in flushed[0]["hadith"]


def test_accumulator_merges_cross_page_hadiths_and_holds_buffer(state: StateManager):
    p1 = """كتاب العقل

1- الحديث الاول كامل في هذه الصفحة بما يكفي من الحروف.

2- الحديث الثاني يبدا هنا
"""
    p2 = """تتمة الثاني على الصفحة التالية بما يكفي.

3- الحديث الثالث يبدا
"""
    p3 = """تتمة الثالث حتى نهاية المجلد بما يكفي من الحروف للنص.
"""
    flushed, buf = consume_page("جلد 1 - صفحه 10", p1, [], None, p2)
    assert [r["marker"] for r in flushed] == ["1-"]
    assert buf is not None and buf.marker == "2-"
    state.set_hadith_buffer("hadith", "al-kafi-1.txt", buf.to_dict())
    held = OpenHadith.from_dict(state.get_hadith_buffer("hadith", "al-kafi-1.txt"))
    flushed2, buf = consume_page("جلد 1 - صفحه 11", p2, [], held, p3)
    assert [r["marker"] for r in flushed2] == ["2-"]
    assert "تتمة الثاني" in flushed2[0]["hadith"]
    assert flushed2[0]["page_start"] != flushed2[0]["page_end"]
    assert buf is not None and buf.marker == "3-"
    flushed3, buf = consume_page("جلد 1 - صفحه 12", p3, [], buf, None)
    assert [r["marker"] for r in flushed3] == ["3-"]
    assert buf is None
    assert "تتمة الثالث" in flushed3[0]["hadith"]


def test_unify_only_when_multipage(state: StateManager):
    calls: list = []

    def fake_generate(*, key, prompt, system, model, schema):
        calls.append(schema)
        return json.dumps({"tags": ["العقل"], "ravis": ["هشام بن الحكم"]}, ensure_ascii=False)

    agent = GeminiAgent(
        state,
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    single = {
        "marker": "1-",
        "locator": "جلد 1 - صفحه 10",
        "page_start": "جلد 1 - صفحه 10",
        "page_end": "جلد 1 - صفحه 10",
        "hadith": "متن",
        "hadith_fa": "متن",
        "hadith_en": "text",
        "tags": ["الجهل"],
        "ravis": ["زرارة"],
    }
    out = unify_assembled_hadith(agent, single)
    assert calls == []
    assert out["ravis"] == ["زرارة"]
    multi = dict(single)
    multi["page_end"] = "جلد 1 - صفحه 12"
    multi["locator"] = "جلد 1 - صفحه 10 تا 12"
    out = unify_assembled_hadith(agent, multi)
    assert calls == [HadithUnify]
    assert out["ravis"] == ["هشام بن الحكم"]
    assert "العقل" in out["tags"]

