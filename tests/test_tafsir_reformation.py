"""Tafsir reformation: source-language routing, prompt bans, Groq payload."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.pipelines.phase1_report import Phase1DayReport
from unittest.mock import MagicMock

import pytest

from src.agents.gemini import GeminiAgent
from src.agents.key_pool import KeyPool
from src.extractors.epub_parser import ParsedUnit
from src.pipelines.prompts import tafsir_reformation_prompt
from src.pipelines.tafsir_reformation import (
    assemble_payload,
    infer_source_lang,
    llm_schema,
    process_unit,
    seam_unique_suffix,
    split_source_windows,
    stitch_window_translations,
    targets_complete,
)
from src.state_manager import StateManager
from tests.test_groq import _groq_settings


@pytest.fixture
def state(tmp_path: Path) -> StateManager:
    return StateManager(tmp_path / "state.db")


def test_persian_unit_keeps_fa_and_asks_ar_en():
    unit = ParsedUnit(
        locator="سوره 2 - آیات 220-227 | بيان",
        text="این تفسیر فارسی است و از پچژگ نیز دارد.",
        source_path="tafsir/al-mizan.txt",
    )
    lang = infer_source_lang(unit.text, "fa")
    assert lang == "fa"
    fields = set(llm_schema(lang).model_fields)
    assert fields == {"tafsir_ar", "tafsir_en"}
    payload = assemble_payload(
        unit,
        lang,
        {"tafsir_ar": "عربي كلاسيكي", "tafsir_en": "Formal English.", "tafsir_fa": "IGNORE"},
    )
    assert payload["tafsir_fa"] == unit.text
    assert payload["tafsir_ar"] == "عربي كلاسيكي"
    assert payload["tafsir_en"] == "Formal English."
    assert "mentions" not in payload
    assert targets_complete(payload)


def test_arabic_unit_keeps_ar_and_asks_fa_en():
    unit = ParsedUnit(
        locator="سوره 2 - آیات 1-5",
        text="قال المؤلف في تفسير الآية وهذا كلام عربي.",
        source_path="tafsir/al-borhan-1.txt",
    )
    lang = infer_source_lang(unit.text, "ar")
    assert lang == "ar"
    fields = set(llm_schema(lang).model_fields)
    assert fields == {"tafsir_fa", "tafsir_en"}
    payload = assemble_payload(
        unit,
        lang,
        {"tafsir_fa": "فارسی روان", "tafsir_en": "Formal English.", "tafsir_ar": "IGNORE"},
    )
    assert payload["tafsir_ar"] == unit.text
    assert payload["tafsir_fa"] == "فارسی روان"


def test_reformation_prompt_forbids_mentions_and_ayah_numbers():
    for lang in ("fa", "ar"):
        prompt = tafsir_reformation_prompt(lang)
        lower = prompt.lower()
        assert "mention" in lower
        assert "do not extract mentions" in lower or "do not extract mentions" in prompt.lower()
        assert "ayah" in lower
        assert "do not invent" in lower
        assert "cited_hadiths" in lower
        assert "phase 1" in lower
        assert "summarize" in lower
        assert "the passage discusses" in lower


def test_process_unit_fake_generator_persian(tmp_path: Path, state: StateManager):
    seen: dict = {}

    def fake_generate(**kwargs):
        seen["schema"] = kwargs["schema"]
        seen["system"] = kwargs["system"]
        return json.dumps({"tafsir_ar": "تفسير عربي", "tafsir_en": "Commentary in English."})

    settings = _groq_settings()
    agent = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, settings, keys=["k"]),
        generate_fn=fake_generate,
    )
    unit = ParsedUnit(
        locator="سوره 1 - آیات 1-5 | بيان",
        text="متن فارسی تفسیر با گاف.",
        source_path="al-mizan.txt",
    )
    dest = tmp_path / "out.json"
    report = Phase1DayReport(
        day=date(2026, 9, 27),
        report_dir=tmp_path / "reports",
        live=False,
        kind="tafsir-reformation",
        provider="groq",
        model="qwen/qwen3.8-27b",
    )
    agent.day_report = report
    report.record_http(0, 200)
    status, calls = process_unit(agent, unit, source_lang="fa", dest=dest)
    assert status == "processed"
    assert calls == 1
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["source_lang"] == "fa"
    assert payload["tafsir_fa"] == unit.text
    assert payload["tafsir_ar"] == "تفسير عربي"
    report.flush()
    assert "سوره 1 - آیات 1-5 | بيان" in report.md_path.read_text(encoding="utf-8")
    assert set(seen["schema"].model_fields) == {"tafsir_ar", "tafsir_en"}
    assert "mentions" not in seen["system"].lower() or "do not extract mentions" in seen["system"].lower()

    status2, calls2 = process_unit(agent, unit, source_lang="fa", dest=dest)
    assert status2 == "skipped"
    assert calls2 == 0


def test_groq_body_none_effort_and_low_temp(state: StateManager, monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"tafsir_ar": "عربي", "tafsir_en": "English"}
                        )
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode("utf-8")
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    monkeypatch.setattr("src.agents.gemini.urllib.request.urlopen", fake_urlopen)
    settings = _groq_settings()
    agent = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, settings, keys=["gsk_test"]),
        groq_temperature=0.2,
        groq_reasoning_effort="none",
        groq_max_completion_tokens=1000,
    )
    schema = llm_schema("fa")
    result = agent.complete_structured("Locator: x\n\nText: فارسی", schema, system="translate")
    assert result.tafsir_en == "English"
    assert captured["body"]["reasoning_effort"] == "none"
    assert captured["body"]["temperature"] == 0.2
    assert captured["body"]["max_completion_tokens"] == 1000


def test_groq_accepts_fenced_or_reasoning_json(state: StateManager, monkeypatch: pytest.MonkeyPatch):
    from src.agents.gemini import coerce_structured_text

    assert coerce_structured_text('preamble\n```json\n{"tafsir_en": "x"}\n```') == '{"tafsir_en": "x"}'

    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": 'Here is JSON:\n{"tafsir_ar": "عربي", "tafsir_en": "English"}',
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode("utf-8")
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    monkeypatch.setattr("src.agents.gemini.urllib.request.urlopen", fake_urlopen)
    settings = _groq_settings()
    agent = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, settings, keys=["gsk_test"]),
        groq_temperature=0.2,
        groq_reasoning_effort="none",
    )
    schema = llm_schema("fa")
    result = agent.complete_structured("Locator: x", schema, system="translate")
    assert result.tafsir_en == "English"
    assert "response_format" not in captured["body"]


def test_windows_overlap_full_text_and_seam_drops_duplicate_head():
    paras = [f"Paragraph {i} with enough words to stay intact. " * 3 for i in range(8)]
    text = "\n\n".join(paras)
    windows = split_source_windows(text, window_chars=180, overlap_chars=60)
    assert len(windows) >= 2
    second = windows[1]
    assert second["send_start"] < second["unique_start"]
    assert text[second["send_start"] : second["unique_start"]]
    # Overlap is real source, not a recap.
    overlap = text[second["send_start"] : second["unique_start"]]
    assert "Paragraph" in overlap
    assert "passage discusses" not in overlap

    prev = "ONE unique start. Shared tail about mercy remains across this whole commentary unit."
    curr = "Shared tail about mercy remains across this whole commentary unit. TWO unique continuation here."
    unique = seam_unique_suffix(
        prev_translation=prev,
        curr_translation=curr,
        overlap_source="Shared tail about mercy remains.",
        window_source=curr,
    )
    assert unique.startswith("TWO unique")
    assert unique.count("Shared tail") == 0

    # Different wording for the overlap must not be sliced mid-sentence.
    kept = seam_unique_suffix(
        prev_translation="The argument closes with another supernatural event.",
        curr_translation=(
            "Overlap rendered in other words entirely. "
            "miracle, so that people may accept it."
        ),
        overlap_source="x" * 80,
        window_source=("x" * 80) + (" unique source words." * 8),
    )
    assert kept.startswith("Overlap rendered")
    assert "miracle, so that" in kept

    stitched = stitch_window_translations(
        "aaaa bbbb cccc dddd",
        [
            {
                "send_start": 0,
                "send_end": 10,
                "unique_start": 0,
                "tafsir_en": "AAAA BBBB",
            },
            {
                "send_start": 5,
                "send_end": 19,
                "unique_start": 10,
                "tafsir_en": "BBBB CCCC DDDD",
            },
        ],
        "tafsir_en",
    )
    assert "AAAA" in stitched
    assert "DDDD" in stitched
    # Four-character overlap is below the safe match, so the window stays whole.
    assert stitched.count("BBBB") == 2


def test_resume_stops_and_continues_windows(tmp_path: Path, state: StateManager):
    calls = {"n": 0}

    def fake_generate(**kwargs):
        calls["n"] += 1
        n = calls["n"]
        return json.dumps({"tafsir_ar": f"عربي-{n}", "tafsir_en": f"English-{n}."})

    settings = _groq_settings()
    agent = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, settings, keys=["k"]),
        generate_fn=fake_generate,
    )
    text = "\n\n".join(f"Block {i} " + ("متن فارسی گاف. " * 20) for i in range(6))
    unit = ParsedUnit(
        locator="سوره 1 - آیات 1-5 | بيان",
        text=text,
        source_path="al-mizan.txt",
    )
    dest = tmp_path / "out.json"
    status, used = process_unit(
        agent,
        unit,
        source_lang="fa",
        dest=dest,
        budget=1,
        window_chars=220,
        overlap_chars=80,
    )
    assert status == "partial"
    assert used == 1
    assert not dest.exists()
    draft = dest.with_suffix(".draft.json")
    assert draft.exists()
    saved = json.loads(draft.read_text(encoding="utf-8"))
    assert saved["locator"] == unit.locator
    done = sum(1 for w in saved["windows"] if w.get("tafsir_en"))
    assert done == 1

    status2, used2 = process_unit(
        agent,
        unit,
        source_lang="fa",
        dest=dest,
        window_chars=220,
        overlap_chars=80,
    )
    assert status2 == "processed"
    assert used2 >= 1
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["tafsir_fa"] == unit.text
    assert "English-" in payload["tafsir_en"]
    assert not draft.exists()


def test_json_mode_failure_splits_window(tmp_path: Path, state: StateManager):
    from src.agents.errors import RequestTooLarge

    hits = {"n": 0}

    def fake_generate(**kwargs):
        hits["n"] += 1
        if hits["n"] == 1:
            raise RequestTooLarge("input tokens too large")
        return json.dumps({"tafsir_ar": f"عربي-{hits['n']}", "tafsir_en": f"English-{hits['n']}."})

    settings = _groq_settings()
    agent = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, settings, keys=["k"]),
        generate_fn=fake_generate,
    )
    text = "\n\n".join(f"Block {i} " + ("متن فارسی گاف. " * 25) for i in range(4))
    unit = ParsedUnit(
        locator="سوره 1 - آیات 1-5 | بيان",
        text=text,
        source_path="al-mizan.txt",
    )
    dest = tmp_path / "out.json"
    status, used = process_unit(
        agent,
        unit,
        source_lang="fa",
        dest=dest,
        window_chars=400,
        overlap_chars=80,
    )
    assert status == "processed"
    assert hits["n"] >= 2
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["tafsir_fa"] == unit.text
