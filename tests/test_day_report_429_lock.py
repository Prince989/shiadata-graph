"""Tests for day-report-driven 429 overnight locks."""
from datetime import date
from pathlib import Path

import pytest

from src.agents.gemini import GeminiAgent, classify_provider_error
from src.agents.key_pool import FailureKind, KeyPool
from src.pipelines.phase1_report import Phase1DayReport
from src.state_manager import StateManager


@pytest.fixture
def state(tmp_path: Path) -> StateManager:
    return StateManager(tmp_path / "state.db")


def test_day_report_429_over_three_locks_on_next_rate_limit(tmp_path: Path, state: StateManager):
    report = Phase1DayReport(day=date(2026, 9, 14), report_dir=tmp_path, live=False)
    for _ in range(3):
        report.record_http(0, 429)
    assert report.count_429(0) == 3

    agent = GeminiAgent(state, key_pool=KeyPool(state, keys=["k"]), day_report=report)

    class Exc(Exception):
        status_code = 429

    report.record_http(0, 429)
    assert report.count_429(0) == 4
    kind, _ = classify_provider_error(
        Exc(
            "429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier "
            "Please retry in 10s"
        )
    )
    assert kind == FailureKind.RATE_LIMITED
    if kind == FailureKind.RATE_LIMITED and agent._day_429_count(agent.pool._keys[0]) > 3:
        kind = FailureKind.QUOTA_EXHAUSTED
    agent.pool.report_failure(agent.pool._keys[0], kind)
    row = state.get_cooldown(agent.pool._keys[0].id)
    assert row is not None
    assert row["reason"] == FailureKind.QUOTA_EXHAUSTED.value


def test_startup_lock_indexes_from_day_report(tmp_path: Path, state: StateManager):
    report = Phase1DayReport(day=date(2026, 9, 14), report_dir=tmp_path, live=False)
    for _ in range(5):
        report.record_http(1, 429)
    pool = KeyPool(state, keys=["a", "b", "c"])
    locked = pool.lock_indexes_for_day(report.key_indexes_over_429(3))
    assert len(locked) == 1
    assert locked[0].secret == "b"
    row = state.get_cooldown(locked[0].id)
    assert row["reason"] == FailureKind.QUOTA_EXHAUSTED.value
