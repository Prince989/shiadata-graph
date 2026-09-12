from src.pipelines.hadith_accumulator import OpenHadith
from src.pipelines.runner import _dump_resume_buffer, _split_resume_buffer


def test_legacy_open_buffer_still_loads():
    raw = OpenHadith(marker="12 -", page_start="p13", page_end="p15").to_dict()
    buf, pending = _split_resume_buffer(raw)
    assert buf is not None and buf.marker == "12 -"
    assert pending == []


def test_pending_unify_survives_round_trip():
    open_buf = OpenHadith(marker="14 -", page_start="p20", page_end="p20")
    pending = [
        {
            "marker": "12 -",
            "page_start": "جلد 1 - صفحه 13",
            "page_end": "جلد 1 - صفحه 19",
            "hadith": "يا هشام",
        }
    ]
    dumped = _dump_resume_buffer(open_buf, pending)
    buf, loaded = _split_resume_buffer(dumped)
    assert buf is not None and buf.marker == "14 -"
    assert loaded[0]["marker"] == "12 -"
    assert _dump_resume_buffer(None, []) is None
