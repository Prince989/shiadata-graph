"""SQLite stays the default. Postgres is selected only by DATABASE_URL."""

from src.state_manager import StateManager, adapt_sql, source_basename


def test_adapt_sql_rewrites_placeholders() -> None:
    sql = "SELECT * FROM chunks WHERE id = ? AND book_id = ?"
    assert adapt_sql(sql, postgres=False) == sql
    assert adapt_sql(sql, postgres=True) == (
        "SELECT * FROM chunks WHERE id = %s AND book_id = %s"
    )


def test_explicit_empty_url_stays_on_sqlite(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    state = StateManager(tmp_path / "state.db", database_url="")
    try:
        assert state._postgres is False
        assert state.db_path == tmp_path / "state.db"
        assert state.upsert_chunks(
            [
                {
                    "id": "c1",
                    "book_id": "hadith",
                    "pipeline": "hadith",
                    "locator": "جلد 1 - صفحه 1",
                    "source_path": "al-kafi-1.txt",
                    "text": "متن",
                }
            ]
        ) == 1
        assert state.get_chunk("c1") is not None
    finally:
        state.close()


def test_windows_resume_path_matches_container_path(tmp_path) -> None:
    windows = r"D:\shiadata.dev\shiadata-graph\data\raw_epubs\hadith\al-kafi-5.txt"
    linux = "/app/data/raw_epubs/hadith/al-kafi-5.txt"
    assert source_basename(windows) == "al-kafi-5.txt"
    state = StateManager(tmp_path / "state.db", database_url="")
    try:
        state.set_hadith_progress("hadith", windows, "جلد 5 - صفحه 95")
        state.set_hadith_buffer(
            "hadith",
            windows,
            {"open": {"marker": "2"}, "pending_unify": []},
        )
        assert state.get_hadith_progress("hadith", linux) == "جلد 5 - صفحه 95"
        assert state.get_hadith_buffer("hadith", linux)["open"]["marker"] == "2"
        assert state.get_hadith_progress("hadith", str(tmp_path / "al-kafi-1.txt")) is None
        state.set_hadith_progress("hadith", linux, "جلد 5 - صفحه 96")
        stored = [
            row["source_path"]
            for row in state._conn.execute("SELECT source_path FROM hadith_progress")
        ]
        assert stored == [linux]
    finally:
        state.close()
