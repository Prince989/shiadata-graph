from datetime import date
from pathlib import Path

from src.pipelines.phase1_report import Phase1DayReport, hadith_run_id


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
    assert "|   1 |   1 |   0 |   0 |       0 |" in text
    report.record_http(0, 200)
    report.record_hadith(
        "al-kafi-1.txt",
        {"marker": "13 -", "locator": "جلد 1 - صفحه 20"},
    )
    text = report.md_path.read_text(encoding="utf-8")
    assert "al-kafi-1_1_20_13" in text
    assert "### Key 1" in text
    assert "|   1 |   1 |   1 |   0 |       1 |" in text


def test_day_report_merges_same_day_rerun(tmp_path: Path):
    day = date(2026, 9, 12)
    first = Phase1DayReport(day=day, report_dir=tmp_path, live=False)
    first.record_http(0, 200)
    first.record_http(0, 503)
    first.record_http(1, 429)
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
    assert "|   1 |   1 |   2 |   0 |       2 |" in text
