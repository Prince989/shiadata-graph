from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from config.settings import Settings
from src.agents.errors import AllKeysExhausted, RateLimited, StructuredOutputError
from src.agents.gemini import GeminiAgent
from src.agents.key_pool import FailureKind, KeyPool, LlmKey, key_id_for
from src.core.edge_classifier import build_concept_buckets
from src.core.phase2 import remap_existing_semantic_nodes
from src.core.vector_engine import (
    bucket_keys_for_chunk,
    cosine_similarity,
    pairs_above_threshold,
    section_nodes_for_chunk,
)
from src.extractors.chunkers import hadith_units, page_prefix_and_starts, split_hadith_page, strip_folklib_footnotes
from src.extractors.classification import attach_sections, page_headings, section_nodes
from src.extractors.epub_parser import ParsedUnit, parse_epub, strip_html
from src.extractors.quran_refs import match_quran_phrases, page_quran_refs, parse_footnote_ref
from src.extractors.txt_parser import parse_txt
from src.models import (
    HadithExtraction,
    HadithPageExtraction,
    HadithUnify,
    HadithUnifyRequireTopics,
    HistoryExtraction,
    MentionsFill,
    TafsirExtraction,
)
from src.pipelines.hadith_accumulator import OpenHadith, consume_page
from src.pipelines.llm_processor import (
    hadith_system_extra,
    phase1_filename,
    process_unit,
    system_prompt,
    unify_assembled_hadith,
)
from src.pipelines.ontology import (
    broader_chain,
    bucket_eligible,
    canonicalize_concept,
    enforce_node_policy,
    normalize_ar,
    remap_hadith_payload,
    resolve_node,
)
from src.pipelines.proposals import apply_promotions, build_plan, collect, repair_map
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
                "concept_nodes": [],
                "ravis": [],
            },
            ensure_ascii=False,
        )

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_min_interval_ms=0),
        key_pool=pool,
        generate_fn=fake_generate,
    )
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


def test_classify_retired_model_404_is_server_error():
    from src.agents.gemini import classify_provider_error

    kind, ms = classify_provider_error(
        Exception(
            "404 NOT_FOUND. This model models/gemini-2.5-flash is no longer available "
            "to new users. Please update your code to use models/gemini-3.6-flash."
        )
    )
    assert kind == FailureKind.SERVER_ERROR
    assert ms is None


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
    with pytest.raises(AllKeysExhausted, match="free tier"):
        pool.acquire()


def test_quota_unlocks_after_pacific_midnight(state: StateManager, monkeypatch):
    pool = KeyPool(state, keys=["k"])
    key = pool.acquire()
    today_mid = 1_800_000_000_000
    lock = today_mid - 3_600_000
    retry = lock + 86_400_000
    now = today_mid + 60_000
    state.set_cooldown(key.id, FailureKind.QUOTA_EXHAUSTED.value, 3, retry)
    monkeypatch.setattr("src.agents.key_pool.time.time", lambda: now / 1000.0)
    monkeypatch.setattr("src.agents.key_pool.pacific_day_start_ms", lambda _now: today_mid)
    monkeypatch.setattr(
        "src.agents.key_pool.next_pacific_midnight_ms",
        lambda _now: today_mid + 86_400_000,
    )
    again = pool.acquire()
    assert again.secret == "k"
    assert state.get_cooldown(key.id) is None


def test_quota_stays_locked_same_pacific_day(state: StateManager, monkeypatch):
    monkeypatch.setattr(
        "src.agents.key_pool.time.sleep",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("slept")),
    )
    pool = KeyPool(state, keys=["k"])
    key = pool.acquire()
    today_mid = 1_800_000_000_000
    lock = today_mid + 3_600_000
    retry = lock + 86_400_000
    now = lock + 5_000
    state.set_cooldown(key.id, FailureKind.QUOTA_EXHAUSTED.value, 1, retry)
    monkeypatch.setattr("src.agents.key_pool.time.time", lambda: now / 1000.0)
    monkeypatch.setattr("src.agents.key_pool.pacific_day_start_ms", lambda _now: today_mid)
    monkeypatch.setattr(
        "src.agents.key_pool.next_pacific_midnight_ms",
        lambda _now: today_mid + 86_400_000,
    )
    with pytest.raises(AllKeysExhausted, match="free tier"):
        pool.acquire()


def test_server_error_strikes_reset_after_cooldown_expires(state: StateManager):
    import time as time_mod

    pool = KeyPool(state, keys=["k"])
    key = pool.acquire()
    now = int(time_mod.time() * 1000)
    state.set_cooldown(key.id, FailureKind.SERVER_ERROR.value, 5, now - 1_000)
    pool.report_failure(key, FailureKind.SERVER_ERROR)
    rec = state.get_cooldown(key.id)
    assert rec["strikes"] == 1
    def fail(**_kwargs):
        raise RuntimeError(
            "429 RESOURCE_EXHAUSTED. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier. "
            "Please retry in 55s."
        )

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_max_attempts=20, gemini_min_interval_ms=0),
        key_pool=KeyPool(state, keys=["a", "b"]),
        generate_fn=fail,
    )
    with pytest.raises(AllKeysExhausted, match="free tier"):
        agent.complete("x")


def test_gemini_waits_20s_between_calls(state: StateManager, monkeypatch: pytest.MonkeyPatch):
    slept: list[float] = []
    monkeypatch.setattr("src.agents.gemini.time.sleep", lambda s: slept.append(s))

    def fake_generate(**_kwargs):
        return "ok"

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_min_interval_ms=20_000, gemini_max_attempts=1),
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    assert agent.complete("a") == "ok"
    assert slept == []
    assert agent.complete("b") == "ok"
    assert len(slept) == 1
    assert slept[0] == pytest.approx(20.0, abs=0.05)


def test_gemini_min_interval_actually_sleeps(state: StateManager):
    import time as time_mod

    def fake_generate(**_kwargs):
        return "ok"

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_min_interval_ms=400, gemini_max_attempts=1),
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    t0 = time_mod.monotonic()
    agent.complete("a")
    agent.complete("b")
    elapsed = time_mod.monotonic() - t0
    assert elapsed >= 0.35


def test_gemini_503_falls_back_to_next_model(state: StateManager, monkeypatch: pytest.MonkeyPatch):
    slept: list[float] = []
    monkeypatch.setattr("src.agents.gemini.time.sleep", lambda s: slept.append(s))
    seen: list[tuple[str, str]] = []

    def fake_generate(*, key, prompt, system, model, schema):
        seen.append((key.secret, model))
        if model == "hot-model":
            raise RuntimeError("503 UNAVAILABLE. high demand")
        return "ok"

    agent = GeminiAgent(
        state,
        settings=Settings(
            gemini_min_interval_ms=20_000,
            gemini_max_attempts=3,
            gemini_models=["hot-model", "cool-model"],
            gemini_model="hot-model",
        ),
        key_pool=KeyPool(state, keys=["a", "b"]),
        generate_fn=fake_generate,
    )
    assert agent.complete("x") == "ok"
    assert seen == [("a", "hot-model"), ("a", "cool-model")]
    assert slept == [5.0]
    assert state.get_cooldown(key_id_for("a", 0)) is None


def test_gemini_503_retries_same_model_when_alone(
    state: StateManager, monkeypatch: pytest.MonkeyPatch
):
    slept: list[float] = []
    monkeypatch.setattr("src.agents.gemini.time.sleep", lambda s: slept.append(s))
    seen: list[str] = []

    def fake_generate(*, key, prompt, system, model, schema):
        seen.append(model)
        if len(seen) == 1:
            raise RuntimeError("503 UNAVAILABLE. high demand")
        return "ok"

    agent = GeminiAgent(
        state,
        settings=Settings(
            gemini_min_interval_ms=0,
            gemini_max_attempts=3,
            gemini_models=["gemini-3.6-flash"],
        ),
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    assert agent.complete("x") == "ok"
    assert seen == ["gemini-3.6-flash", "gemini-3.6-flash"]
    assert slept == [5.0]


def test_gemini_404_retired_model_falls_back(state: StateManager):
    seen: list[str] = []

    def fake_generate(*, key, prompt, system, model, schema):
        seen.append(model)
        if model == "gemini-2.5-flash":
            raise RuntimeError(
                "404 NOT_FOUND. This model models/gemini-2.5-flash is no longer available "
                "to new users. Please update your code to use models/gemini-3.6-flash."
            )
        return "ok"

    agent = GeminiAgent(
        state,
        settings=Settings(
            gemini_min_interval_ms=0,
            gemini_max_attempts=3,
            gemini_models=["gemini-2.5-flash", "gemini-3.6-flash"],
        ),
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    assert agent.complete("x") == "ok"
    assert seen == ["gemini-2.5-flash", "gemini-3.6-flash"]


def test_all_keys_exhausted_leaves_chunks_pending(state: StateManager):
    pool = KeyPool(state, keys=["only"])
    k = pool.acquire()
    pool.report_failure(k, FailureKind.QUOTA_EXHAUSTED)
    with pytest.raises(AllKeysExhausted, match="free tier"):
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


def _concept_nodes(*labels: str, role: str = "primary") -> list[dict]:
    return [{"node": label, "type": "concept", "role": role} for label in labels]


def test_pydantic_hadith_schema():
    item = HadithExtraction.model_validate(
        {
            "hadith": "الرفق يمن",
            "hadith_fa": "مدارا مبارک است",
            "hadith_en": "Gentleness is blessed",
            "semantic_nodes": _concept_nodes("الرفق"),
            "ravis": ["زرارة"],
        }
    )
    assert item.semantic_nodes[0].node == "الرفق"
    page = HadithPageExtraction.model_validate(
        {
            "page": "جلد 1 - صفحه 11",
            "hadiths": [
                {
                    "marker": "3 -",
                    "hadith": "أ",
                    "hadith_fa": "آ",
                    "hadith_en": "a",
                    "semantic_nodes": [],
                    "ravis": [],
                },
                {
                    "marker": "4 -",
                    "hadith": "ب",
                    "hadith_fa": "ب",
                    "hadith_en": "b",
                    "semantic_nodes": _concept_nodes("العقل"),
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
                        "semantic_nodes": _concept_nodes("العقل", "العبادة"),
                        "ravis": ["علي بن إبراهيم"],
                    },
                    {
                        "marker": "4 -",
                        "hadith": "حديث أربعة",
                        "hadith_fa": "چهار",
                        "hadith_en": "four",
                        "semantic_nodes": _concept_nodes("الجهل", "العبادة"),
                        "ravis": ["محمد بن يحيى"],
                    },
                ],
            },
            ensure_ascii=False,
        )

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_min_interval_ms=0),
        key_pool=KeyPool(state, keys=["test-key"]),
        generate_fn=fake_generate,
    )
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


def test_gate_rejects_real_bad_nodes():
    # Whole clauses and two-idea labels are dropped outright.
    assert resolve_node("العقل مناط التكليف والجزاء", "concept") is None
    assert resolve_node("فضيلة العقل على العبادة", "concept") is None
    assert resolve_node("الشيطنة والنكراء", "concept") is None
    # A short variant of a known topic is repaired onto it rather than dropped:
    # the output is always a vocabulary term, so it cannot fragment the graph.
    assert resolve_node("حقيقة العقل", "concept") == "العقل"
    assert resolve_node("ارتباط الثواب بالعقل", "concept") == "الثواب"


def test_gate_keeps_and_canonicalizes_good_nodes():
    assert resolve_node("الشيطنة", "concept") == "النكراء"
    assert resolve_node("قتل النفس", "concept") == "الانتحار"
    assert resolve_node("العقل", "concept") == "العقل"
    assert resolve_node("أبو عبد الله", "person") == "الإمام الصادق"
    assert resolve_node("أبو الحسن", "person") == "أبو الحسن"
    assert resolve_node("59:2", "ayah") == "59:2"
    assert resolve_node("أولي الأبصار", "ayah") is None


def test_parent_concepts_are_bucket_eligible_again():
    assert bucket_eligible("concept", "secondary") is True
    for label in ("الصبر", "التقية", "الإيمان", "الموت"):
        assert resolve_node(label, "concept") == label


def test_person_secondary_excluded_from_buckets_but_kept_in_graph():
    assert bucket_eligible("person", "secondary") is False
    assert bucket_eligible("person", "primary") is True
    assert bucket_eligible("ayah", "primary") is False


def test_normalization_folds_harakat_honorifics_and_kunya_case():
    # All four print the same man; only the last matches the gazetteer verbatim.
    forms = ["أَبُو عَبْدِ اللَّهِ (ع)", "أبي عبد الله", "أبا عبد الله", "أبو عبد الله عليه السلام"]
    assert {normalize_ar(f) for f in forms} == {normalize_ar("أبو عبد الله")}
    for form in forms:
        assert resolve_node(form, "person") == "الإمام الصادق"
    # Superscript alef is a letter, not a mark: deleting it breaks Qur'anic spellings.
    assert normalize_ar("ٱلْأَبْصَٰرِ") == normalize_ar("الأبصار")


def test_narrator_is_dropped_but_the_matn_subject_survives():
    """Verbatim hadith 3: the Imam asked is a narrator, معاوية is the topic."""
    nodes = [
        {"node": "العقل", "type": "concept", "role": "primary"},
        {"node": "معاوية", "type": "person", "role": "primary"},
        {"node": "أبي عبد الله", "type": "person", "role": "secondary"},
        {"node": "النكراء", "type": "concept", "role": "secondary"},
    ]
    ravis = ["أَحْمَدُ بْنُ إِدْرِيسَ", "بَعْضُ أَصْحَابِنَا", "أَبُو عَبْدِ اللَّهِ (ع)"]
    out = enforce_node_policy(nodes, ravis)
    labels = {n["node"] for n in out}
    assert "الإمام الصادق" not in labels and "أبي عبد الله" not in labels
    assert "معاوية" in labels
    # النكراء is what the matn defines, so it takes the primary slot.
    roles = {n["node"]: n["role"] for n in out}
    assert roles["النكراء"] == "primary"
    assert roles["معاوية"] == "secondary"


def test_participants_absent_from_the_isnad_are_kept_as_nodes():
    """Hadith 2: آدم and جبرئيل are in the story, not in the chain.

    They stay in the graph. They do not stay primary: primary slots decide
    bucketing, and a person node buckets nothing useful while الحياء groups
    every hadith on modesty across the corpus.
    """
    nodes = [
        {"node": "العقل", "type": "concept", "role": "primary"},
        {"node": "الحياء", "type": "concept", "role": "secondary"},
        {"node": "آدم", "type": "person", "role": "primary"},
        {"node": "جبرئيل", "type": "person", "role": "secondary"},
    ]
    out = enforce_node_policy(nodes, ["عَلِيُّ بْنُ مُحَمَّدٍ", "سَهْلُ بْنُ زِيَادٍ", "عَلِيٌّ"])
    roles = {n["node"]: n["role"] for n in out}
    assert set(roles) == {"العقل", "الحياء", "آدم", "جبرئيل"}
    assert roles["العقل"] == "primary"
    assert roles["الحياء"] == "primary"
    assert roles["آدم"] == "secondary"


def test_book_classification_is_inherited_by_every_page():
    """Kulayni's own kitab/bab headings, read straight off the page."""
    pages = [
        ParsedUnit("ص 9", "خطبة المؤلف بلا عنوان هنا " * 3, "al-kafi-1.txt"),
        ParsedUnit("ص 10", "كِتَابُ الْعَقْلِ وَ الْجَهْلِ\n\n1- أَخْبَرَنَا أَبُو جَعْفَرٍ قَالَ خَلَقَ اللَّهُ الْعَقْلَ.", "al-kafi-1.txt"),
        ParsedUnit("ص 11", "2- عَلِيُّ بْنُ مُحَمَّدٍ عَنْ سَهْلٍ قَالَ هَبَطَ جَبْرَئِيلُ عَلَى آدَمَ.", "al-kafi-1.txt"),
        ParsedUnit("ص 12", "بَابُ طِينَةِ\n\nالْمُؤْمِنِ وَ الْكَافِرِ\n\n3 - مُحَمَّدُ بْنُ يَحْيَى عَنْ أَحْمَدَ قَالَ.", "al-kafi-1.txt"),
    ]
    tagged = attach_sections(pages)
    assert tagged[0].kitab == ""
    assert tagged[1].kitab == "كِتَابُ الْعَقْلِ وَ الْجَهْلِ"
    # A heading governs every later page until the next one, so hadith 2
    # inherits it without the heading being reprinted.
    assert tagged[2].kitab == "كِتَابُ الْعَقْلِ وَ الْجَهْلِ"
    # Wrapped title reassembled from the line the typesetter split it onto.
    assert tagged[3].bab == "بَابُ طِينَةِ الْمُؤْمِنِ وَ الْكَافِرِ"


def test_prose_beginning_with_kitab_is_not_a_heading():
    page = "كتاب الحجّة و إن لم نكمّله على استحقاقه، لأنّا كرهنا أن نبخس حظوظه كلّها."
    assert page_headings(page) == []


def test_miscellany_bab_is_not_a_bucket_key():
    """Every volume has a بَابُ النَّوَادِرِ and they share no subject."""
    assert section_nodes("كِتَابُ الْعَقْلِ", "بَابُ النَّوَادِرِ") == [
        {"node": "كِتَابُ الْعَقْلِ", "type": "kitab"}
    ]


def test_vocabulary_is_closed_for_concepts_and_groups():
    """The model can only choose; it cannot invent a concept or a group."""
    assert resolve_node("اجتهاد المجتهدين", "concept") is None
    assert resolve_node("عتاب الله", "concept") is None
    assert resolve_node("أولي الأبصار", "group") is None
    assert resolve_node("العقل", "concept") == "العقل"
    assert resolve_node("بني إسرائيل", "group") == "بنو إسرائيل"
    # person stays open: named individuals cannot be enumerated up front.
    assert resolve_node("سلمان الفارسي", "person") == "سلمان الفارسي"
    # Non-hadith pipelines keep their free-text concepts until they have a
    # proposals channel of their own.
    assert resolve_node("الحمد", "concept", strict=False) == "الحمد"


def test_frequency_decides_which_proposals_become_terms():
    """The whole point: measured df, not a hand-written rule.

    خلق العقل and عقل المرء are indistinguishable as strings -- both are a noun
    plus a genitive. Only the corpus separates them.
    """
    payloads = [
        {"marker": "1", "locator": "p10", "proposed_nodes": ["خلق العقل", "عقل المرء"]},
        {"marker": "14", "locator": "p14", "proposed_nodes": ["خلق العقل"]},
        {"marker": "20", "locator": "p16", "proposed_nodes": ["اجتهاد المجتهدين"]},
    ]
    plan = build_plan(collect(payloads))
    assert [p.term for p in plan.promote] == ["خلق العقل"]
    assert repair_map(plan) == {"عقل المرء": "العقل"}
    assert [p.term for p in plan.drop] == ["اجتهاد المجتهدين"]


def test_promotion_appends_parseable_yaml_and_keeps_comments(tmp_path: Path):
    catalog = tmp_path / "base_ontology.yaml"
    catalog.write_text(
        "concepts:\n  # a comment worth keeping\n  - {id: العقل, pref: العقل}\n",
        encoding="utf-8",
    )
    payloads = [
        {"marker": str(i), "locator": "p", "proposed_nodes": ["خلق العقل"]}
        for i in range(3)
    ]
    plan = build_plan(collect(payloads))
    assert apply_promotions(plan, catalog_path=catalog) == 1
    text = catalog.read_text(encoding="utf-8")
    assert "a comment worth keeping" in text
    import yaml as _yaml

    parsed = _yaml.safe_load(text)
    assert {c["pref"] for c in parsed["concepts"]} == {"العقل", "خلق العقل"}


def test_already_catalogued_terms_are_not_proposals():
    payloads = [{"marker": "1", "locator": "p", "proposed_nodes": ["العقل", "الصبر"]}]
    assert collect(payloads) == {}


def test_qualified_compounds_reduce_to_the_shared_concept():
    """Verbatim labels from a real run. Each was its own df=1 node.

    Repair, not invention: every output here is already a vocabulary term.
    """
    assert resolve_node("عقل المرء", "concept") == "العقل"
    assert resolve_node("جهل المرء", "concept") == "الجهل"
    assert resolve_node("حساب العباد", "concept") == "الحساب"
    assert resolve_node("كمال العقل", "concept") == "العقل"
    assert resolve_node("قدر العقول", "concept") == "العقل"


def test_unreducible_compounds_are_dropped_not_stored():
    """Nothing else in the corpus will ever carry these, so they are noise."""
    for label in ("اجتهاد المجتهدين", "عتاب الله", "التكليف الإلهي"):
        assert resolve_node(label, "concept") is None
    # فرائض الله now reaches الفرائض, which the harvest found as a chapter title.
    # That is the enrichment working, not a leak: it resolves to a real term
    # rather than being stored as a singleton nobody can reach.
    assert resolve_node("فرائض الله", "concept") == "الفرائض"
    # A Qur'anic phrase the model retyped as a group to dodge the ayah ban.
    assert resolve_node("أولي الأبصار", "group") is None
    assert resolve_node("أولو الألباب", "concept") is None
    # A gazetteer group still passes.
    assert resolve_node("بني إسرائيل", "group") == "بنو إسرائيل"


def test_reduction_never_rescues_a_two_idea_label():
    # النكراء is a catalog concept, so reduction would happily keep it and
    # silently discard الشيطنة. The و-check has to win first.
    assert resolve_node("الشيطنة والنكراء", "concept") is None


def test_a_two_idea_label_the_catalog_itself_carries_is_matched_whole():
    """كتاب العقل و الجهل is one of al-Kafi's books, so it is one concept.

    The rule above is about reduction discarding half a label. Matching the
    whole of it against a catalog entry that happens to name two ideas is not
    that -- nothing is dropped. Only the typesetter's spacing around the
    conjunction differs, which `normalize_ar` folds.
    """
    assert resolve_node("العقل والجهل", "concept") == "العقل و الجهل"
    assert resolve_node("العقل و الجهل", "concept") == "العقل و الجهل"


def test_named_being_emitted_as_a_concept_is_retyped_and_demoted():
    """Hadith 10: الشيطان held a primary concept slot; وسواس is the topic."""
    out = enforce_node_policy(
        [
            {"node": "العقل", "type": "concept", "role": "primary"},
            {"node": "الشيطان", "type": "concept", "role": "primary"},
            {"node": "الوسواس", "type": "concept", "role": "secondary"},
        ]
    )
    by_node = {n["node"]: n for n in out}
    assert by_node["الشيطان"]["type"] == "person"
    assert by_node["الشيطان"]["role"] == "secondary"
    assert by_node["الوسواس"]["role"] == "primary"
    # person/secondary is not a bucket key, so Satan stops joining every
    # hadith that mentions him into one cluster.
    assert bucket_eligible("person", "secondary") is False


def test_over_marked_primaries_do_not_demote_the_curated_concept():
    """The model marking everything primary must not hand the slot to list order.

    Promote-then-truncate had this hole: with no secondaries to promote from,
    the overflow cut kept the first two in emission order and demoted النكراء.
    """
    out = enforce_node_policy(
        [
            {"node": "معاوية", "type": "person", "role": "primary"},
            {"node": "العقل", "type": "concept", "role": "primary"},
            {"node": "النكراء", "type": "concept", "role": "primary"},
        ],
        ["أَحْمَدُ بْنُ إِدْرِيسَ", "أَبُو عَبْدِ اللَّهِ (ع)"],
    )
    roles = {n["node"]: n["role"] for n in out}
    assert roles == {"معاوية": "secondary", "العقل": "primary", "النكراء": "primary"}


def test_entity_holds_primary_only_when_no_concept_survives():
    out = enforce_node_policy(
        [
            {"node": "معاوية", "type": "person", "role": "primary"},
            {"node": "بني إسرائيل", "type": "group", "role": "secondary"},
        ]
    )
    roles = {n["node"]: n["role"] for n in out}
    assert roles["معاوية"] == "primary"
    # A secondary entity is never promoted into a primary slot.
    assert roles["بنو إسرائيل"] == "secondary"


def test_repaired_and_dropped_terms_reach_the_proposals_channel():
    """Promotion must not depend on the model choosing the right field.

    خلق العقل written into semantic_nodes still repairs to العقل so the graph
    stays connected, but the original has to survive as a proposal or it can
    never accumulate the df that would promote it.
    """
    payload = remap_hadith_payload(
        {
            "hadiths": [
                {
                    "hadith": "x",
                    "ravis": [],
                    "semantic_nodes": [
                        {"node": "خلق العقل", "type": "concept", "role": "primary"},
                        {"node": "اجتهاد المجتهدين", "type": "concept", "role": "secondary"},
                        {"node": "العقل", "type": "concept", "role": "secondary"},
                    ],
                }
            ]
        }
    )
    row = payload["hadiths"][0]
    assert [n["node"] for n in row["semantic_nodes"]] == ["العقل"]
    assert set(row["proposed_nodes"]) == {"خلق العقل", "اجتهاد المجتهدين"}


def test_open_types_still_reject_descriptive_heads():
    """The closed vocabulary covers concepts; person/place stay open and need this."""
    assert resolve_node("فضيلة محمد", "person") is None
    assert resolve_node("أهمية كربلاء", "place") is None
    assert resolve_node("سلمان الفارسي", "person") == "سلمان الفارسي"


def test_kitab_is_a_graph_edge_not_a_comparison_bucket():
    """A kitab spans hundreds of narrations; pairwise comparison there is noise."""
    base = _hadith_chunk("h1", "hadith", ["العقل"])
    payload = json.loads(base.payload_json)
    payload["hadiths"][0]["kitab"] = "كِتَابُ الْعَقْلِ وَ الْجَهْلِ"
    payload["hadiths"][0]["bab"] = "بَابُ صِفَةِ الْعِلْمِ"
    chunk = replace(base, payload_json=json.dumps(payload, ensure_ascii=False))
    keys = bucket_keys_for_chunk(chunk)
    assert "بَابُ صِفَةِ الْعِلْمِ" in keys
    assert "كِتَابُ الْعَقْلِ وَ الْجَهْلِ" not in keys
    # Both still reach the exported graph.
    assert {n["type"] for n in section_nodes_for_chunk(chunk)} == {"kitab", "bab"}


def test_curated_concept_takes_a_free_primary_slot():
    """Hadith 3 re-run: معاوية arrived secondary, so there was nothing to trade."""
    out = enforce_node_policy(
        [
            {"node": "العقل", "type": "concept", "role": "primary"},
            {"node": "معاوية", "type": "person", "role": "secondary"},
            {"node": "النكراء", "type": "concept", "role": "secondary"},
        ],
        ["أَحْمَدُ بْنُ إِدْرِيسَ", "أَبُو عَبْدِ اللَّهِ (ع)"],
    )
    roles = {n["node"]: n["role"] for n in out}
    assert roles["النكراء"] == "primary"
    assert roles["معاوية"] == "secondary"


def test_generic_node_is_rejected_but_its_qualified_form_is_not():
    assert resolve_node("محبة", "concept") is None
    assert resolve_node("الأمر", "concept") is None
    assert resolve_node("محبة أهل البيت", "concept") == "محبة أهل البيت"
    assert resolve_node("الأمر بالمعروف", "concept") == "الأمر بالمعروف"


def test_catalog_curated_label_is_retyped_to_concept():
    """place:الجنة and event:يوم القيامة could never merge with their twins."""
    out = enforce_node_policy(
        [
            {"node": "الجنة", "type": "place", "role": "secondary"},
            {"node": "يوم القيامة", "type": "event", "role": "secondary"},
        ]
    )
    assert [(n["node"], n["type"]) for n in out] == [
        ("الجنة", "concept"),
        ("القيامة", "concept"),
    ]


def test_policy_caps_primary_and_total():
    vocab = ["الصبر", "التقوى", "الإيمان", "الكفر", "الموت", "القبر", "الدعاء", "التوبة"]
    nodes = [{"node": label, "type": "concept", "role": "primary"} for label in vocab]
    out = enforce_node_policy(nodes)
    assert len(out) <= 6
    assert sum(1 for n in out if n["role"] == "primary") == 2


def test_canonicalize_alias_and_parent_not_grouping():
    assert canonicalize_concept("قتل النفس") == "الانتحار"
    assert canonicalize_concept("عذاب القبر") == "عذاب البرزخ"
    old = remap_hadith_payload({"hadiths": [{"hadith": "x", "concept_nodes": ["الإيمان"]}]})
    assert old["hadiths"][0]["semantic_nodes"][0]["node"] == "الإيمان"
    assert "concept_nodes" not in old["hadiths"][0]


def test_hadith_prompt_asks_for_mentions_not_nodes():
    text = system_prompt("hadith")
    assert "at most one extra tag" not in text
    assert "Base Ontology" not in text
    # The contract is observations, resolved corpus-wide afterwards.
    assert "WHAT A MENTION IS" in text
    assert "salience" in text
    assert "evidence" in text
    # No vocabulary is imposed and none is listed.
    assert "VOCABULARY" not in text
    assert "CHOOSE, DO NOT INVENT" not in text
    # Citations are still resolved against the mushaf, never asked for.
    assert "verse number" in text
    assert "59:2" not in text
    assert "39:9" not in text
    # The rules that kept being broken are stated as principles.
    assert "معاوية" in text
    assert "NARRATORS ARE NOT MENTIONS" in text
    assert "WHAT IS ASSERTED" in text
    extra = hadith_system_extra("3 - متن\n\n4 - متن آخر طويل بما يكفي للعد.")
    assert "SEPARATE claim" in extra


def test_schema_forbids_the_model_from_inventing_an_ayah_node():
    with pytest.raises(ValidationError):
        HadithExtraction.model_validate(
            {
                "hadith": "متن",
                "hadith_fa": "فا",
                "hadith_en": "en",
                "semantic_nodes": [{"node": "39:9", "type": "ayah", "role": "secondary"}],
            }
        )


def _hadith_chunk(cid: str, book: str, nodes: list[str]) -> ChunkRecord:
    payload = {
        "page": "p",
        "hadiths": [
            {
                "marker": "1-",
                "hadith": "متن",
                "hadith_fa": "متن",
                "hadith_en": "text",
                "semantic_nodes": _concept_nodes(*nodes),
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


def test_concept_buckets_cross_book_and_keep_singleton():
    chunks = [
        _hadith_chunk("a", "hadith", ["الانتحار", "عذاب البرزخ"]),
        _hadith_chunk("b", "vasael-o-shia", ["قتل النفس"]),
        _hadith_chunk("c", "hadith", ["الزهد"]),
        _hadith_chunk("d", "hadith", ["الإيمان", "المعاد"]),
    ]
    buckets = build_concept_buckets(chunks)
    assert set(buckets["الانتحار"]) == {"a", "b"}
    assert "c" in buckets["الزهد"]
    assert set(buckets["الإيمان"]) == {"d"}
    assert "الإيمان" in bucket_keys_for_chunk(chunks[3])
    # "a" carries عذاب البرزخ, whose parent is المعاد, so it joins d's bucket
    # without either chunk having emitted المعاد as a node.
    assert set(buckets["المعاد"]) == {"a", "d"}
    assert "المعاد" in bucket_keys_for_chunk(chunks[0])
    assert "عذاب البرزخ" in bucket_keys_for_chunk(chunks[0])


def test_recompense_siblings_share_one_bucket_but_stay_distinct_nodes():
    """Hadiths 7, 8 and 9 name الحساب / الثواب / الجزاء and share no node."""
    chunks = [
        _hadith_chunk("h7", "hadith", ["الحساب"]),
        _hadith_chunk("h8", "hadith", ["الثواب"]),
        _hadith_chunk("h9", "hadith", ["الجزاء"]),
    ]
    buckets = build_concept_buckets(chunks)
    assert set(buckets["الجزاء الأخروي"]) == {"h7", "h8", "h9"}
    # Still three separate nodes -- reckoning is not reward.
    assert set(buckets["الحساب"]) == {"h7"}
    assert set(buckets["الثواب"]) == {"h8"}
    assert set(buckets["الجزاء"]) == {"h9"}
    # الحساب keeps its second parent as well as the new one.
    assert set(broader_chain("الحساب")) >= {"الجزاء الأخروي", "القيامة", "المعاد"}


def test_extreme_df_skipped_on_large_set_only():
    popular = [_hadith_chunk(f"p{i}", "hadith", ["الانتحار"]) for i in range(8)]
    rare = [_hadith_chunk("r0", "hadith", ["الزهد"])]
    hesab = [_hadith_chunk(f"h{i}", "hadith", ["الحساب"]) for i in range(2)]
    other = [_hadith_chunk(f"z{i}", "hadith", [f"موضوع-{i}"]) for i in range(10)]
    buckets = build_concept_buckets(
        popular + rare + hesab + other, max_df_ratio=0.15, min_chunks_for_df=20
    )
    assert "الانتحار" not in buckets
    assert "الزهد" in buckets
    assert "الحساب" in buckets


def test_remap_existing_concepts_without_gemini(tmp_path: Path, state: StateManager):
    payload = {
        "page": "جلد 1 - صفحه 1",
        "hadiths": [{"hadith": "x", "hadith_fa": "x", "hadith_en": "x", "concept_nodes": ["قتل النفس"], "ravis": []}],
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
    n = remap_existing_semantic_nodes(state, output_dir=out)
    assert n == 1
    stored = state.get_chunk(cid).payload()
    assert stored["hadiths"][0]["semantic_nodes"][0]["node"] == "الانتحار"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["hadiths"][0]["semantic_nodes"][0]["node"] == "الانتحار"


def test_process_unit_keeps_mentions_verbatim_for_the_resolver(tmp_path: Path, state: StateManager):
    def fake_generate(*, key, prompt, system, model, schema):
        return json.dumps(
            {
                "page": "جلد 1 - صفحه 11",
                "hadiths": [
                    {
                        "marker": "3 -",
                        "hadith_fa": "سه",
                        "hadith_en": "three",
                        "mentions": [
                            {"text": "قتل النفس", "type": "concept", "salience": 0.9},
                            {"text": "عذاب القبر", "type": "concept", "salience": 0.5},
                        ],
                        "ravis": [],
                    }
                ],
            },
            ensure_ascii=False,
        )

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_min_interval_ms=0),
        key_pool=KeyPool(state, keys=["test-key"]),
        generate_fn=fake_generate,
    )
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
    # Aliases are resolved corpus-wide now, so the page extract keeps the matn's
    # own wording; resolve-nodes maps قتل النفس onto الانتحار.
    assert [m["text"] for m in data["hadiths"][0]["mentions"]] == [
        "قتل النفس",
        "عذاب القبر",
    ]


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
                "concept_nodes": [],
                "ravis": [],
            },
            ensure_ascii=False,
        )

    settings = Settings(gemini_max_attempts=3, google_api_keys=["k"], gemini_min_interval_ms=0)
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

    settings = Settings(
        gemini_max_attempts=2, google_api_keys=["k"], skip_min_chars=10, gemini_min_interval_ms=0
    )
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
        if schema is MentionsFill:
            return json.dumps(
                {
                    "mentions": [
                        {
                            "text": "العقل",
                            "type": "concept",
                            "salience": 0.9,
                            "evidence": "متن حديث",
                        },
                        {
                            "text": "الجنة",
                            "type": "concept",
                            "salience": 0.5,
                            "evidence": "متن حديث",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        if isinstance(schema, type) and issubclass(schema, HadithUnify):
            return json.dumps(
                {
                    "semantic_nodes": _concept_nodes("العقل", "الجنة"),
                    "ravis": ["هشام"],
                }
            )
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
                        "semantic_nodes": _concept_nodes("العقل", "الجنة"),
                        "ravis": [],
                    }
                ],
            },
            ensure_ascii=False,
        )

    settings = Settings(
        gemini_max_attempts=2, google_api_keys=["k"], skip_min_chars=10, gemini_min_interval_ms=0
    )
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


def test_locator_matches_page_number_and_volume():
    from src.pipelines.runner import locator_matches_page

    loc = "جلد 1 - صفحه 30"
    assert locator_matches_page(loc, "30")
    assert locator_matches_page(loc, "30", volume=1)
    assert locator_matches_page(loc, "جلد 1 - صفحه 30")
    assert not locator_matches_page(loc, "300")
    assert not locator_matches_page(loc, "30", volume=2)
    assert not locator_matches_page("جلد 1 - صفحه 3", "30")


def test_hadith_phase1_page_filter_only_that_page(
    tmp_path: Path, state: StateManager, monkeypatch: pytest.MonkeyPatch
):
    from src.pipelines import runner as runner_mod
    from src.pipelines.catalog import BookSpec

    raw = tmp_path / "hadith"
    raw.mkdir()
    long = "متن حديث طويل بما يكفي. " * 8
    path = raw / "al-kafi-1.txt"
    pages = "\n".join(
        f"--- [جلد 1 - صفحه {n}] ---\n\n{n}- {long}\n" for n in range(29, 32)
    )
    path.write_text(pages, encoding="utf-8")
    page_calls: list[str] = []

    def fake_generate(*, key, prompt, system, model, schema):
        if schema is MentionsFill:
            return json.dumps(
                {
                    "mentions": [
                        {
                            "text": "العقل",
                            "type": "concept",
                            "salience": 0.9,
                            "evidence": "متن حديث",
                        },
                        {
                            "text": "الجنة",
                            "type": "concept",
                            "salience": 0.5,
                            "evidence": "متن حديث",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        if isinstance(schema, type) and issubclass(schema, HadithUnify):
            return json.dumps(
                {
                    "semantic_nodes": _concept_nodes("العقل", "الجنة"),
                    "ravis": ["هشام"],
                    "hadith_fa": "ف",
                    "hadith_en": "e",
                }
            )
        page_calls.append(prompt)
        return json.dumps(
            {
                "page": "جلد 1 - صفحه 30",
                "hadiths": [
                    {
                        "marker": "30 -",
                        "mentions": [
                            {
                                "text": "العقل",
                                "type": "concept",
                                "salience": 0.9,
                                "evidence": "متن حديث",
                            }
                        ],
                        "ravis": ["هشام"],
                        "quotes": [],
                        "hadith_fa": "ف",
                        "hadith_en": "e",
                    }
                ],
            },
            ensure_ascii=False,
        )

    settings = Settings(
        gemini_max_attempts=2,
        google_api_keys=["k"],
        skip_min_chars=10,
        gemini_min_interval_ms=0,
    )
    agent = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    # Pretend a full run already progressed past page 30 — page filter must
    # still hit 30 and must not rewrite that resume cursor.
    state.set_hadith_progress("hadith", str(path), "جلد 1 - صفحه 100")
    monkeypatch.setattr(runner_mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(
        runner_mod,
        "resolve_book",
        lambda *a, **k: BookSpec("hadith", "hadith", "", [path]),
    )
    stats = runner_mod.run_phase1(
        "hadith", state, agent, settings=settings, page="30", volume=1
    )
    assert stats["pages"] == 1
    assert stats["errors"] == 0
    assert len(page_calls) == 1
    assert "صفحه 30" in page_calls[0]
    assert state.get_hadith_progress("hadith", str(path)) == "جلد 1 - صفحه 100"


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


def test_strip_folklib_bottom_notes_do_not_leak_into_hadith_eight():
    page11 = """8- عَلِيُّ
بْنُ مُحَمَّدِ بْنِ عَبْدِ اللَّهِ‌ [6] عَنْ
إِبْرَاهِيمَ بْنِ إِسْحَاقَ الْأَحْمَرِ عَنْ مُحَمَّدِ بْنِ‌

[1] الشأن بالهمزة: الامر و الحال أي الزما شأنكما أو شأنكما
معكما و يحتمل أن يكون الإشارة تمثيلية و ان اللّه تعالى خلق صورة مناسبة لكل واحد
منها و بعثها مع جبرئيل عليه السلام( آت)

[6] الظاهر أنّه ابن بندار او عليّ بن محمّد بن عبد اللّه
القمّيّ كما أن الظاهر اتّحاد الرجلين. و قال الفيض- رحمه اللّه- كانه ابن أذينة
الذي هو من مشايخ الكليني و يحتمل ابن عمران البرقي.
"""
    page12 = """سُلَيْمَانَ الدَّيْلَمِيِّ عَنْ أَبِيهِ
قَالَ: قُلْتُ لِأَبِي عَبْدِ اللَّهِ ع فُلَانٌ مِنْ عِبَادَتِهِ.

9 - عَلِيُّ بْنُ إِبْرَاهِيمَ عَنْ أَبِيهِ قَالَ قَالَ رَسُولُ اللَّهِ ص‌ إِذَا بَلَغَكُمْ.
"""
    clean11 = strip_folklib_footnotes(page11)
    assert "يحتمل" not in clean11
    assert "تمثيلية" not in clean11
    assert "القمّيّ" not in clean11
    assert "( آت)" not in clean11
    assert "إِبْرَاهِيمَ" in clean11
    flushed, buf = consume_page("جلد 1 - صفحه 11", clean11, [], None, page12)
    assert [r["marker"] for r in flushed] == []
    assert buf is not None and buf.marker.startswith("8")
    flushed2, buf = consume_page("جلد 1 - صفحه 12", page12, [], buf, None)
    assert any(r["marker"].startswith("8") for r in flushed2)
    eight = next(r for r in flushed2 if r["marker"].startswith("8"))
    assert "يحتمل" not in eight["hadith"]
    assert "سُلَيْمَانَ" in eight["hadith"] or "سليمان" in eight["hadith"]


def test_footnote_refs_parse_every_format_this_edition_prints():
    assert parse_footnote_ref("البقرة: 269 و فيها « وَ ما يَذَّكَّرُ") == "2:269"
    assert parse_footnote_ref("الرعد 41.") == "13:41"          # no colon
    assert parse_footnote_ref("يونس، 39.") == "10:39"          # Arabic comma
    assert parse_footnote_ref("الحجّ: 12.") == "22:12"          # shadda in the name
    assert parse_footnote_ref("الصافّات: 138.") == "37:138"
    assert parse_footnote_ref("ص: 28.") == "38:28"             # one-letter sura
    assert parse_footnote_ref("آل عمران: 7.") == "3:7"         # multi-word
    assert parse_footnote_ref("المؤمن: 70.") == "40:70"        # classical name for غافر
    assert parse_footnote_ref("سورة البقرة: 1") == "2:1"


def test_footnote_refs_reject_commentary_and_cross_references():
    # "ص 321" would read as Sad:321 without the verse-count check.
    assert parse_footnote_ref("قد مر الحديث ص 321 فراجعه") is None
    assert parse_footnote_ref("و يأتي في ج 5 ص 87 و فيه: عن ابن أبي عمير") is None
    assert parse_footnote_ref("أي: يجازى على اعماله بقدر عقله") is None
    assert parse_footnote_ref("") is None
    # Real sura, impossible ayah.
    assert parse_footnote_ref("الفاتحة: 99") is None


def test_phrase_matcher_finds_unfootnoted_quotations():
    """Hadith 5 quotes 59:2 with no [n] marker anywhere on the page."""
    assert match_quran_phrases("إِنَّمَا قَالَ اللَّهُ‌ فَاعْتَبِرُوا يا أُولِي الْأَبْصارِ .") == ["59:2"]
    # أولوا الألباب closes both 2:269 and 3:7; both loci are real.
    assert set(match_quran_phrases("وَ مَا يَتَذَكَّرُ إِلَّا أُولُوا الْأَلْبابِ‌")) == {"2:269", "3:7"}
    # A hadith with no scripture in it must stay empty.
    assert match_quran_phrases("صَدِيقُ كُلِّ امْرِئٍ عَقْلُهُ وَ عَدُوُّهُ جَهْلُهُ.") == []


def test_page_refs_ignore_scripture_quoted_by_the_editor():
    """Page 12's note [3] quotes Surat al-Nas as commentary, not as a citation.

    The hadith never quotes it, so no CITES edge may be produced.
    """
    page = """11 - عِدَّةٌ مِنْ أَصْحَابِنَا عَنْ أَحْمَدَ بْنِ مُحَمَّدٍ رَفَعَهُ قَالَ قَالَ رَسُولُ اللَّهِ ص‌ مَا قَسَمَ اللَّهُ لِلْعِبَادِ شَيْئاً أَفْضَلَ مِنَ الْعَقْلِ.

[3] فهو يعلم ان الوسوسة من عمل الشيطان لما في قوله تعالى‌ « مِنْ

شَرِّ الْوَسْواسِ الْخَنَّاسِ الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ» و

لكنه لا يتمكن من طرده حين العمل.
"""
    assert page_quran_refs(page) == {}


def test_page_refs_attribute_a_footnote_to_the_hadith_that_carries_its_marker():
    page = """5 - قَالَ: قُلْتُ لِأَبِي الْحَسَنِ ع إِنَّ عِنْدَنَا قَوْماً لَهُمْ مَحَبَّةٌ [4] .

6 - أَحْمَدُ بْنُ إِدْرِيسَ قَالَ قَالَ أَبُو عَبْدِ اللَّهِ ع‌ مَنْ كَانَ عَاقِلًا كَانَ لَهُ دِينٌ.

[4] البقرة: 269.
"""
    refs = page_quran_refs(page)
    assert refs.get("5 -") == ["2:269"]
    assert "6 -" not in refs


def test_footnote_quoting_vocalized_quran_does_not_reopen_the_matn():
    """Verbatim page 12 of al-Kafi 1: note [3] quotes Surat al-Nas.

    The quoted verse carries 27 harakat, which used to satisfy the
    "matn resumed" heuristic and let both the rest of the verse and the
    editor's next sentence leak into hadith 11.
    """
    page = """11 - عِدَّةٌ
مِنْ أَصْحَابِنَا عَنْ أَحْمَدَ بْنِ مُحَمَّدِ بْنِ خَالِدٍ عَنْ بَعْضِ أَصْحَابِهِ
رَفَعَهُ قَالَ قَالَ رَسُولُ اللَّهِ ص‌ مَا قَسَمَ اللَّهُ لِلْعِبَادِ
شَيْئاً أَفْضَلَ مِنَ الْعَقْلِ فَنَوْمُ الْعَاقِلِ‌

[3] فهو يعلم ان الوسوسة من عمل الشيطان لما في قوله تعالى‌ « مِنْ
شَرِّ الْوَسْواسِ الْخَنَّاسِ الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ» و
لكنه لا يتمكن من طرده حين العمل.
"""
    clean = strip_folklib_footnotes(page)
    assert "الْخَنَّاسِ" not in clean
    assert "طرده" not in clean
    assert "الوسوسة من عمل الشيطان" not in clean
    # The hadith itself must survive intact.
    assert "مَا قَسَمَ اللَّهُ لِلْعِبَادِ" in clean
    assert "فَنَوْمُ الْعَاقِلِ" in clean


def test_unclosed_guillemet_cannot_swallow_the_next_hadith():
    page = """[2] قال تعالى « إن الله
لا يظلم الناس شيئا

3 - أَحْمَدُ بْنُ إِدْرِيسَ عَنْ مُحَمَّدِ بْنِ عَبْدِ الْجَبَّارِ قَالَ: قُلْتُ لَهُ مَا الْعَقْلُ.
"""
    clean = strip_folklib_footnotes(page)
    assert "لا يظلم الناس" not in clean
    assert clean.lstrip().startswith("3 -")


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
        return json.dumps(
            {
                "semantic_nodes": _concept_nodes("العقل", "الجنة"),
                "ravis": ["هشام بن الحكم"],
            },
            ensure_ascii=False,
        )

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_min_interval_ms=0),
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
        "semantic_nodes": _concept_nodes("الجهل", "العبادة"),
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
    assert "الجهل" in [n["node"] for n in out["semantic_nodes"]]


def test_unify_fills_hollow_single_page(state: StateManager):
    calls: list[type] = []

    def fake_generate(*, key, prompt, system, model, schema):
        calls.append(schema)
        if schema is MentionsFill:
            return json.dumps(
                {
                    "mentions": [
                        {
                            "text": "العقل",
                            "type": "concept",
                            "salience": 0.9,
                            "evidence": "ما العقل",
                        },
                        {
                            "text": "الجنة",
                            "type": "concept",
                            "salience": 0.5,
                            "evidence": "الجنان",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "hadith_fa": "فارسی",
                "hadith_en": "English",
                "mentions": [],
                "ravis": ["أحمد بن إدريس"],
            },
            ensure_ascii=False,
        )

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_min_interval_ms=0),
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    hollow = {
        "marker": "3 -",
        "locator": "جلد 1 - صفحه 11",
        "page_start": "جلد 1 - صفحه 11",
        "page_end": "جلد 1 - صفحه 11",
        "hadith": "قُلْتُ مَا الْعَقْلُ قَالَ مَا عُبِدَ بِهِ الرَّحْمَنُ وَ اكْتُسِبَ بِهِ الْجِنَانُ",
        "hadith_fa": "",
        "hadith_en": "",
        "mentions": [],
        "semantic_nodes": [],
        "ravis": [],
    }
    out = unify_assembled_hadith(agent, hollow)
    assert calls == [MentionsFill, HadithUnify]
    assert out["hadith_fa"] == "فارسی"
    assert out["hadith_en"] == "English"
    assert out["ravis"] == ["أحمد بن إدريس"]
    assert {m["text"] for m in out["mentions"]} >= {"العقل", "الجنة"}


def test_unify_accepts_translation_only_reply(state: StateManager):
    """Empty mentions must not reject unify -- that crashed live Phase 1 runs."""
    n = {"i": 0}

    def fake_generate(*, key, prompt, system, model, schema):
        n["i"] += 1
        assert schema is HadithUnify
        return json.dumps(
            {
                "hadith_fa": "ف",
                "hadith_en": "e",
                "mentions": [],
                "semantic_nodes": [],
                "ravis": ["زرارة"],
            },
            ensure_ascii=False,
        )

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_min_interval_ms=0, gemini_max_attempts=4),
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    hollow = {
        "marker": "6 -",
        "locator": "جلد 1 - صفحه 11",
        "page_start": "جلد 1 - صفحه 11",
        "page_end": "جلد 1 - صفحه 11",
        "hadith": "متن عن العقل",
        "hadith_fa": "",
        "hadith_en": "",
        "mentions": [
            {"text": "العقل", "type": "concept", "salience": 0.9, "evidence": "العقل"}
        ],
        "semantic_nodes": [],
        "ravis": ["زرارة"],
    }
    out = unify_assembled_hadith(agent, hollow)
    assert n["i"] == 1
    assert out["hadith_fa"] == "ف"
    assert out["hadith_en"] == "e"
    assert [m["text"] for m in out["mentions"]] == ["العقل"]


def test_unify_overwrites_truncated_translations(state: StateManager):
    """Multi-page stubs ending in `...` must not block a full re-translation."""
    from src.pipelines.llm_processor import needs_enrichment, translation_incomplete

    assert translation_incomplete("به اهل دینی که عقل ندارند اعتنایی نمی‌شود...")
    assert translation_incomplete(
        "Imam said: no importance...\nContinuation of the previous narration: ..."
    )
    assert not translation_incomplete("امام رضا فرمود: عقل را آفرید و گفت اقبال کن.")

    n = {"i": 0}

    def fake_generate(*, key, prompt, system, model, schema):
        n["i"] += 1
        assert schema is HadithUnify
        return json.dumps(
            {
                "hadith_fa": (
                    "امام رضا (ع) فرمود: به اهل دینی که عقل ندارند اعتنایی نمی‌شود. "
                    "گفتم: فدایت شوم، از کسانی که این امر را توصیف می‌کنند قومی نزد ما "
                    "بی‌اشکال‌اند ولی آن عقول را ندارند. فرمود: اینان از کسانی نیستند "
                    "که خدا خطابشان کرده؛ خدا عقل را آفرید و گفت پیش آی، پیش آمد، "
                    "بازگرد، بازگشت، و فرمود چیزی نیکوتر یا محبوب‌تر از تو نیافریدم."
                ),
                "hadith_en": (
                    "Imam al-Rida (as) said: No importance is given to people of "
                    "religion who have no intellect. I said: May I be your ransom, "
                    "some who describe this matter seem fine to us yet lack those "
                    "intellects. He said: Those are not whom God addressed; God "
                    "created the intellect and said come forward, it came, go back, "
                    "it went back, and said I created nothing better or dearer than you."
                ),
                "mentions": [],
                "semantic_nodes": [],
                "ravis": ["الحسن بن الجهم", "أبو الحسن الرضا ع"],
            },
            ensure_ascii=False,
        )

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_min_interval_ms=0, gemini_max_attempts=4),
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    stub = {
        "marker": "32 -",
        "locator": "جلد 1 - صفحه 27 تا 28",
        "page_start": "جلد 1 - صفحه 27",
        "page_end": "جلد 1 - صفحه 28",
        "hadith": "ذُكِرَ عِنْدَهُ أَصْحَابُنَا وَ ذُكِرَ الْعَقْلُ ... بِكَ آخُذُ وَ بِكَ أُعْطِي.",
        "hadith_fa": "امام رضا (ع) فرمودند: به اهل دینی که عقل ندارند اعتنایی نمی‌شود...",
        "hadith_en": (
            "Imam al-Rida (as) said when our companions and the intellect were "
            "mentioned: 'No importance is given to the people of religion who "
            "have no intellect...'"
        ),
        "mentions": [
            {"text": "العقل", "type": "concept", "salience": 0.95, "evidence": "الْعَقْلُ"}
        ],
        "semantic_nodes": [],
        "ravis": ["الحسن بن الجهم"],
    }
    assert needs_enrichment(stub) is True
    out = unify_assembled_hadith(agent, stub)
    assert n["i"] == 1
    assert "پیش آی" in out["hadith_fa"]
    assert "come forward" in out["hadith_en"]
    assert "..." not in out["hadith_fa"]
    assert out["ravis"] == ["الحسن بن الجهم", "أبو الحسن الرضا ع"]


def test_unify_requires_topics_when_mentions_empty(state: StateManager):
    n = {"i": 0}

    def fake_generate(*, key, prompt, system, model, schema):
        n["i"] += 1
        if schema is MentionsFill:
            if n["i"] == 1:
                return json.dumps({"mentions": []}, ensure_ascii=False)
            return json.dumps(
                {
                    "mentions": [
                        {
                            "text": "العقل",
                            "type": "concept",
                            "salience": 0.9,
                            "evidence": "ما العقل",
                        },
                        {
                            "text": "معاوية",
                            "type": "person",
                            "salience": 0.4,
                            "evidence": "معاوية",
                        },
                    ]
                },
                ensure_ascii=False,
            )
        assert schema is HadithUnify
        return json.dumps(
            {
                "hadith_fa": "ف",
                "hadith_en": "e",
                "mentions": [],
                "ravis": ["ز"],
            },
            ensure_ascii=False,
        )

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_min_interval_ms=0, gemini_max_attempts=4),
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    hollow = {
        "marker": "3 -",
        "locator": "جلد 1 - صفحه 11",
        "page_start": "جلد 1 - صفحه 11",
        "page_end": "جلد 1 - صفحه 11",
        "hadith": "ما العقل قال ما عبد به الرحمن فالذي كان في معاوية",
        "hadith_fa": "",
        "hadith_en": "",
        "mentions": [],
        "ravis": [],
    }
    out = unify_assembled_hadith(agent, hollow)
    assert n["i"] == 3  # MentionsFill fail, MentionsFill ok, soft HadithUnify
    assert {m["text"] for m in out["mentions"]} >= {"العقل", "معاوية"}
    assert out["hadith_fa"] == "ف"


def test_unify_survives_structured_output_failure(state: StateManager):
    def fake_generate(*, key, prompt, system, model, schema):
        raise StructuredOutputError("forced")

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_min_interval_ms=0, gemini_max_attempts=1),
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    hollow = {
        "marker": "7 -",
        "locator": "جلد 1 - صفحه 11",
        "page_start": "جلد 1 - صفحه 11",
        "page_end": "جلد 1 - صفحه 11",
        "hadith": "ما العقل قال ما عبد به الرحمن",
        "hadith_fa": "",
        "hadith_en": "",
        "mentions": [
            {
                "text": "العقل",
                "type": "concept",
                "salience": 0.9,
                "evidence": "ما العقل",
            }
        ],
        "ravis": [],
    }
    out = unify_assembled_hadith(agent, hollow)
    assert out["mentions"][0]["text"] == "العقل"
    assert out["hadith_fa"] == ""


def test_page_extract_allows_empty_mentions_unify_fills_them(state: StateManager):
    """Page gate used to reject the whole extract; MentionsFill recovers."""
    page = HadithExtraction(
        marker="3 -",
        hadith="ما العقل",
        hadith_fa="ف",
        hadith_en="e",
        mentions=[],
        ravis=["ز"],
    )
    assert page.mentions == []

    n = {"i": 0}

    def fake_generate(*, key, prompt, system, model, schema):
        n["i"] += 1
        assert schema is MentionsFill
        return json.dumps(
            {
                "mentions": [
                    {
                        "text": "العقل",
                        "type": "concept",
                        "salience": 0.9,
                        "evidence": "ما العقل",
                    },
                    {
                        "text": "العبادة",
                        "type": "concept",
                        "salience": 0.5,
                        "evidence": "عبد به الرحمن",
                    },
                ],
            },
            ensure_ascii=False,
        )

    agent = GeminiAgent(
        state,
        settings=Settings(gemini_min_interval_ms=0, gemini_max_attempts=2),
        key_pool=KeyPool(state, keys=["k"]),
        generate_fn=fake_generate,
    )
    out = unify_assembled_hadith(
        agent,
        {
            "marker": "3 -",
            "locator": "p",
            "page_start": "p",
            "page_end": "p",
            "hadith": "ما العقل قال ما عبد به الرحمن",
            "hadith_fa": "ف",
            "hadith_en": "e",
            "mentions": [],
            "ravis": ["ز"],
        },
    )
    assert n["i"] == 1
    assert {m["text"] for m in out["mentions"]} >= {"العقل", "العبادة"}
