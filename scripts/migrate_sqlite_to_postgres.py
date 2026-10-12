"""Copy data/state.db into Postgres. Safe to run again: tables are truncated first.

Al-Mizan JSON and drafts are not in this database. Copy the data/ directory
for those. Run this before the hadith and mizan containers start:

    docker compose --profile tools run --rm migrate
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.paths import STATE_DB_PATH
from src.state_manager import StateManager

_TABLES: dict[str, tuple[str, ...]] = {
    "chunks": (
        "id",
        "book_id",
        "pipeline",
        "locator",
        "source_path",
        "text",
        "status",
        "payload_json",
        "error",
        "attempts",
        "canonical_id",
        "created_at",
        "updated_at",
    ),
    "embeddings": ("chunk_id", "model", "vector_json"),
    "edge_pairs": ("pair_id", "left_id", "right_id", "relation", "cosine"),
    "hadith_buffer": ("book_id", "source_path", "payload_json", "updated_at"),
    "hadith_progress": ("book_id", "source_path", "last_locator", "updated_at"),
    "jobs": ("id", "phase", "book_id", "pause_reason", "started_at", "finished_at"),
    "key_cooldowns": (
        "key_id",
        "reason",
        "strikes",
        "retry_at_ms",
        "exhausted_day",
    ),
}

_BATCH = 200


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(row[1]) for row in rows]


def _copy_table(
    source: sqlite3.Connection,
    dest,
    table: str,
) -> int:
    wanted = _TABLES[table]
    present = _sqlite_columns(source, table)
    if not present:
        return 0
    columns = [name for name in wanted if name in present]
    if not columns:
        return 0
    select_list = ", ".join(columns)
    rows = source.execute(f"SELECT {select_list} FROM {table}").fetchall()
    if not rows:
        return 0
    placeholders = ", ".join(["%s"] * len(columns))
    col_sql = ", ".join(columns)
    sql = f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})"
    copied = 0
    batch: list[tuple] = []
    for row in rows:
        batch.append(tuple(row[name] for name in columns))
        if len(batch) >= _BATCH:
            dest.executemany(sql, batch)
            copied += len(batch)
            batch.clear()
    if batch:
        dest.executemany(sql, batch)
        copied += len(batch)
    return copied


def migrate(sqlite_path: Path, database_url: str, *, skip_cooldowns: bool = False) -> dict[str, int]:
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
    state = StateManager(database_url=database_url)
    state.close()

    import psycopg

    uri = sqlite_path.resolve().as_uri() + "?mode=ro"
    source = sqlite3.connect(uri, uri=True)
    source.row_factory = sqlite3.Row
    copy_tables = [
        name for name in _TABLES if not (skip_cooldowns and name == "key_cooldowns")
    ]
    counts: dict[str, int] = {}
    try:
        with psycopg.connect(database_url) as dest:
            with dest.transaction():
                dest.execute(
                    "TRUNCATE TABLE "
                    + ", ".join(_TABLES)
                    + " RESTART IDENTITY"
                )
                for table in copy_tables:
                    counts[table] = _copy_table(source, dest, table)
                max_id = None
                has_jobs = source.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
                ).fetchone()
                if has_jobs:
                    max_id = source.execute("SELECT MAX(id) FROM jobs").fetchone()[0]
                if max_id:
                    dest.execute(
                        "SELECT setval(pg_get_serial_sequence('jobs', 'id'), %s)",
                        (int(max_id),),
                    )
    finally:
        source.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, default=STATE_DB_PATH)
    parser.add_argument("--database-url", default="")
    parser.add_argument(
        "--skip-cooldowns",
        action="store_true",
        help="Leave key_cooldowns empty. Those rows expire on their own.",
    )
    args = parser.parse_args()
    url = (args.database_url or "").strip()
    if not url:
        import os

        url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit("DATABASE_URL is required")
    counts = migrate(args.sqlite, url, skip_cooldowns=args.skip_cooldowns)
    for table, count in counts.items():
        print(f"{table}: {count}")


if __name__ == "__main__":
    main()
