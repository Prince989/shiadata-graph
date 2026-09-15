from datetime import date
from pathlib import Path

from src.agents.key_pool import FailureKind, KeyPool
from src.pipelines.phase1_report import Phase1DayReport, hadith_run_id
from src.state_manager import StateManager


def test_hadith_run_id_format():
    assert (
        hadith_run_id(
            "al-kafi-1.txt",
            {"marker": "13 -", "locator": "جلد 1 - صفحه 20"},
        )
        == "al-kafi-1_1_20_13"
    )
    assert (
        hadith_run_id(
            "al-kafi-1.txt",
            {"marker": "14 -", "locator": "جلد 1 - صفحه 20 تا 23"},
        )
        == "al-kafi-1_1_20-23_14"
    )


def test_day_report_writes_live_after_each_event(tmp_path: Path):
    day = date(2026, 9, 12)
    report = Phase1DayReport(day=day, report_dir=tmp_path, live=True)
    assert report.md_path.exists()
    report.record_http(0, 503)
    text = report.md_path.read_text(encoding="utf-8")
    assert "|   1 |   1 |   0 |   0 |   0 | -    |       0 |" in text
    report.record_http(0, 200)
    report.record_hadith(
        "al-kafi-1.txt",
        {"marker": "13 -", "locator": "جلد 1 - صفحه 20"},
    )
    text = report.md_path.read_text(encoding="utf-8")
    assert "al-kafi-1_1_20_13" in text
    assert "### Key 1" in text
    assert "|   1 |   1 |   1 |   0 |   0 | -    |       1 |" in text


def test_day_report_merges_same_day_rerun(tmp_path: Path):
    day = date(2026, 9, 12)
    first = Phase1DayReport(day=day, report_dir=tmp_path, live=False)
    first.record_http(0, 200)
    first.record_http(0, 503)
    first.record_http(1, 429)
    first.record_http(1, 401)
    first.record_hadith(
        "al-kafi-1.txt",
        {"marker": "13 -", "locator": "جلد 1 - صفحه 20"},
        key_no=1,
    )
    path = first.flush()
    assert path.name == "2026-09-12.md"

    second = Phase1DayReport(day=day, report_dir=tmp_path, live=False)
    second.record_http(0, 200)
    second.record_hadith(
        "al-kafi-1.txt",
        {"marker": "14 -", "locator": "جلد 1 - صفحه 20 تا 23"},
        key_no=1,
    )
    second.flush()
    text = path.read_text(encoding="utf-8")
    assert "`al-kafi-1_1_20_13`" in text
    assert "`al-kafi-1_1_20-23_14`" in text
    assert "|   1 |   1 |   2 |   0 |   0 | -    |       2 |" in text
    assert "|   2 |   0 |   0 |   1 |   1 | -    |       0 |" in text


def test_day_report_lock_column_from_key_pool(tmp_path: Path):
    day = date(2026, 9, 15)
    state = StateManager(tmp_path / "state.db")
    pool = KeyPool(state, keys=["a", "b", "c"])
    pool.report_failure(pool._keys[1], FailureKind.QUOTA_EXHAUSTED)

    report = Phase1DayReport(day=day, report_dir=tmp_path, live=False)
    report.record_http(0, 200)
    report.record_http(1, 429)
    report.bind_key_pool(pool)
    report.flush()
    text = report.md_path.read_text(encoding="utf-8")
    assert "| lock" in text
    assert "|   1 |   0 |   1 |   0 |   0 | -     |       0 |" in text
    assert "|   2 |   0 |   0 |   1 |   0 | quota |       0 |" in text
    data = report.json_path.read_text(encoding="utf-8")
    assert '"lock": "quota"' in data


def test_day_report_archives_json_without_pacific_marker(tmp_path: Path):
    day = date(2026, 9, 15)
    stale = tmp_path / f"{day.isoformat()}.json"
    stale.write_text(
        '{"date":"2026-09-15","keys":{"1":{"503":0,"200":0,"429":14,"401":0,"hadiths":[]}}}\n',
        encoding="utf-8",
    )
    report = Phase1DayReport(day=day, report_dir=tmp_path, live=False)
    assert report.count_429(0) == 0
    assert list(tmp_path.glob("2026-09-15-stale-*.json"))
    report.record_http(0, 200)
    report.flush()
    assert "pacific_day_start_ms" in report.json_path.read_text(encoding="utf-8")
