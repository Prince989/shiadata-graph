"""SQLite stays the default. Postgres is selected only by DATABASE_URL."""

from src.state_manager import StateManager, adapt_sql


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
