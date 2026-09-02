from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.agents.errors import AllKeysExhausted, RateLimited
from src.agents.gemini import GeminiAgent
from src.agents.key_pool import FailureKind, KeyPool, LlmKey
from src.core.vector_engine import cosine_similarity, pairs_above_threshold
from src.extractors.epub_parser import parse_epub, strip_html
from src.extractors.txt_parser import parse_txt
from src.models import HadithExtraction, HistoryExtraction, TafsirExtraction
from src.state_manager import ChunkStatus, StateManager

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
    with pytest.raises(ValidationError):
        HadithExtraction.model_validate({"hadith": "only"})


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
