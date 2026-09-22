"""Groq/Qwen provider: keys, payload, 429 cooldown, interval, truncation."""

from __future__ import annotations

import json
import logging
from datetime import date
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import urllib.error
from pydantic import BaseModel

from config.settings import Settings, collect_groq_keys
from src.agents.errors import StructuredOutputError
from src.agents.gemini import GeminiAgent
from src.agents.key_pool import FailureKind, KeyPool, key_id_for
from src.models import Mention
from src.pipelines.phase1_report import Phase1DayReport
from src.state_manager import StateManager


@pytest.fixture
def state(tmp_path: Path) -> StateManager:
    return StateManager(tmp_path / "state.db")


def _groq_settings(**kwargs) -> Settings:
    values = dict(
        llm_provider="groq",
        groq_api_keys=["k"],
        groq_min_interval_ms=0,
        groq_model="qwen/qwen3.8-27b",
        groq_reasoning_effort="medium",
        groq_max_completion_tokens=16384,
        gemini_max_attempts=3,
        key_acquire_wait_max_ms=1_000,
    )
    values.update(kwargs)
    return Settings(**values)


def test_mention_salience_accepts_qwen_words():
    assert Mention(text="العقل", salience="high").salience == 0.9
    assert Mention(text="العقل", salience="medium").salience == 0.6
    assert Mention(text="العقل", salience="low").salience == 0.3
    assert Mention(text="العقل", salience="primary").salience == 0.9
    assert Mention(text="العقل", salience="0.85").salience == 0.85
    assert Mention(text="العقل", salience=0.7).salience == 0.7
    assert Settings.model_fields["llm_provider"].default == "gemini"


def test_collect_groq_keys_order_and_dedupe():
    keys = collect_groq_keys(
        {
            "GROQ_API_KEY": "alpha",
            "GROQ_API_KEY1": "alpha",
            "GROQ_API_KEY2": "beta",
            "GROQ_API_KEY3": "gamma",
            "GROQ_API_KEY_4": "delta",
        }
    )
    assert keys == ["alpha", "beta", "gamma", "delta"]


def test_groq_key_ids_use_groq_prefix(state: StateManager):
    pool = KeyPool(state, _groq_settings(), keys=["secret"])
    assert pool._keys[0].id.startswith("groq-0-")
    assert pool._keys[0].id == key_id_for("secret", 0, "groq")


def test_groq_posts_medium_reasoning_payload(state: StateManager, monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    class Tiny(BaseModel):
        page: str

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["auth"] = req.get_header("Authorization")
        captured["ua"] = req.get_header("User-agent")
        payload = {
            "choices": [
                {
                    "message": {"content": json.dumps({"page": "جلد 1 - صفحه 39"})},
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
    agent = GeminiAgent(
        state,
        settings=_groq_settings(),
        key_pool=KeyPool(state, _groq_settings(), keys=["gsk_test"]),
    )
    result = agent.complete_structured("Locator: x", Tiny, system="extract")
    assert result.page == "جلد 1 - صفحه 39"
    assert captured["body"]["model"] == "qwen/qwen3.8-27b"
    assert captured["body"]["reasoning_effort"] == "medium"
    assert captured["body"]["max_completion_tokens"] == 16384
    assert captured["body"]["temperature"] == 0.6
    assert captured["auth"] == "Bearer gsk_test"
    assert captured["ua"] == "shiadata-graph"
    system = captured["body"]["messages"][0]["content"]
    assert "Return JSON only" in system


def test_groq_429_stays_rate_limited_not_overnight(
    state: StateManager, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr("src.agents.gemini.time.sleep", lambda s: None)
    monkeypatch.setattr("src.agents.key_pool.time.sleep", lambda s: None)

    def fake_generate(**_kwargs):
        raise urllib.error.HTTPError(
            "https://api.groq.com/openai/v1/chat/completions",
            429,
            "Too Many Requests",
            hdrs={},
            fp=BytesIO(b'{"error":{"code":"rate_limit_exceeded"}}'),
        )

    settings = _groq_settings(
        gemini_max_attempts=4,
        groq_min_interval_ms=1,
        key_cooldown_base_ms=1,
        key_acquire_wait_max_ms=50,
    )
    pool = KeyPool(state, settings, keys=["k"])
    agent = GeminiAgent(state, settings=settings, key_pool=pool, generate_fn=fake_generate)
    with caplog.at_level(logging.WARNING, logger="src.agents.gemini"):
        with pytest.raises(Exception):
            agent.complete("hi")
    assert any("HTTP 429" in rec.getMessage() for rec in caplog.records)
    row = state.get_cooldown(pool._keys[0].id)
    assert row is not None
    assert row["reason"] == FailureKind.RATE_LIMITED.value


def test_groq_min_interval_sleeps(state: StateManager, monkeypatch: pytest.MonkeyPatch):
    slept: list[float] = []
    monkeypatch.setattr("src.agents.gemini.time.sleep", lambda s: slept.append(s))

    def fake_generate(**_kwargs):
        return "ok"

    settings = _groq_settings(groq_min_interval_ms=400, gemini_min_interval_ms=20_000)
    agent = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, settings, keys=["k"]),
        generate_fn=fake_generate,
    )
    agent.complete("a")
    agent.complete("b")
    assert slept
    assert slept[0] >= 0.3


def test_groq_does_not_wait_between_different_keys(
    state: StateManager, monkeypatch: pytest.MonkeyPatch
):
    slept: list[float] = []
    monkeypatch.setattr("src.agents.gemini.time.sleep", lambda s: slept.append(s))

    def fake_generate(**_kwargs):
        return "ok"

    settings = _groq_settings(groq_min_interval_ms=75_000, gemini_min_interval_ms=20_000)
    agent = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, settings, keys=["a", "b", "c"]),
        generate_fn=fake_generate,
    )
    agent.complete("one")
    agent.complete("two")
    agent.complete("three")
    assert slept == []


def test_groq_length_finish_is_structured_error(
    state: StateManager, monkeypatch: pytest.MonkeyPatch
):
    def fake_urlopen(req, timeout=None):
        payload = {
            "choices": [
                {"message": {"content": '{"page":'}, "finish_reason": "length"}
            ]
        }
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode("utf-8")
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        return resp

    monkeypatch.setattr("src.agents.gemini.urllib.request.urlopen", fake_urlopen)
    settings = _groq_settings(gemini_max_attempts=1)
    agent = GeminiAgent(
        state,
        settings=settings,
        key_pool=KeyPool(state, settings, keys=["k"]),
    )
    with pytest.raises(StructuredOutputError, match="truncated"):
        agent.complete("x")


def test_groq_day_report_names_qwen_and_skips_overnight_lock(
    tmp_path: Path, state: StateManager
):
    report = Phase1DayReport(
        day=date(2026, 9, 22),
        report_dir=tmp_path,
        live=False,
        provider="groq",
        model="qwen/qwen3.8-27b",
    )
    assert report.md_path.name == "2026-09-22-qwen-qwen3.8-27b.md"
    for _ in range(5):
        report.record_http(0, 429)
    pool = KeyPool(state, _groq_settings(), keys=["k"])
    report.bind_key_pool(pool)
    report.flush()
    row = state.get_cooldown(pool._keys[0].id)
    assert row is None
    md = report.md_path.read_text(encoding="utf-8")
    assert "qwen/qwen3.8-27b" in md
    assert "provider `groq`" in md
