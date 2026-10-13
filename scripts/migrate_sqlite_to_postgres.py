"""Copy the merged SQLite state into Postgres.

The hadith resume rows were saved with Windows paths. This script rewrites
those paths to the files mounted in the container, then checks that
`get_hadith_progress` finds them. Safe to run again: tables are truncated first.

Al-Mizan JSON and drafts are not in this database. They stay in the data/
directory. Run this before the hadith and mizan containers start:

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

from config.paths import RAW_EPUBS_DIR, STATE_DB_PATH
from src.state_manager import StateManager, source_basename

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


def _raw_files_by_name(raw_root: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    if not raw_root.is_dir():
        raise SystemExit(f"raw epubs directory not found: {raw_root}")
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".epub"}:
            grouped.setdefault(path.name, []).append(str(path))
    return grouped


def _remap_source_path(stored: str, files_by_name: dict[str, list[str]]) -> str:
    matches = files_by_name.get(source_basename(stored), [])
    if len(matches) == 1:
        return matches[0]
    return stored


def _require_progress_files(
    source: sqlite3.Connection, files_by_name: dict[str, list[str]]
) -> None:
    problems: list[str] = []
    rows = source.execute("SELECT source_path FROM hadith_progress").fetchall()
    for row in rows:
        stored = str(row[0])
        name = source_basename(stored)
        matches = files_by_name.get(name, [])
        if len(matches) == 1:
            continue
        if not matches:
            problems.append(f"{stored} (file {name} is not in the raw epubs directory)")
        else:
            problems.append(f"{name} matches {len(matches)} files")
    if problems:
        raise SystemExit(
            "hadith resume files cannot be mapped onto the container paths, "
            "so the run would start from the beginning. "
            + "; ".join(problems)
        )


def _copy_table(
    source: sqlite3.Connection,
    dest,
    table: str,
    files_by_name: dict[str, list[str]],
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
    path_index = columns.index("source_path") if "source_path" in columns else None
    copied = 0
    batch: list[tuple] = []
    for row in rows:
        values = [row[name] for name in columns]
        if path_index is not None and values[path_index]:
            values[path_index] = _remap_source_path(str(values[path_index]), files_by_name)
        batch.append(tuple(values))
        if len(batch) >= _BATCH:
            with dest.cursor() as cur:
                cur.executemany(sql, batch)
            copied += len(batch)
            batch.clear()
    if batch:
        with dest.cursor() as cur:
            cur.executemany(sql, batch)
        copied += len(batch)
    return copied


def _verify_hadith_resume(database_url: str) -> None:
    from src.pipelines.catalog import resolve_book

    state = StateManager(database_url=database_url)
    try:
        spec = resolve_book("hadith")
        by_name = {path.name: path for path in spec.files}
        rows = state._conn.execute(
            "SELECT source_path, last_locator FROM hadith_progress WHERE book_id = ?",
            ("hadith",),
        ).fetchall()
        problems: list[str] = []
        seen: dict[str, str] = {}
        for row in rows:
            stored = str(row["source_path"])
            name = source_basename(stored)
            locator = str(row["last_locator"])
            seen[name] = locator
            live = by_name.get(name)
            print(f"resume {name} {locator}")
            if live is None:
                problems.append(f"no raw file named {name}")
                continue
            if stored != str(live):
                problems.append(f"{name} stored as {stored}")
            if state.get_hadith_progress("hadith", str(live)) != locator:
                problems.append(f"{name} is not visible to the hadith command")
        for number in range(1, 6):
            name = f"al-kafi-{number}.txt"
            if name not in seen:
                problems.append(f"missing progress for {name}")
        volume_5 = by_name.get("al-kafi-5.txt")
        if volume_5 is not None and state.get_hadith_buffer("hadith", str(volume_5)) is None:
            problems.append("volume 5 buffer is not visible to the hadith command")
        chunk_count = state._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        if int(chunk_count["n"]) <= 0:
            problems.append("chunks table is empty")
        if problems:
            raise SystemExit("hadith resume check failed: " + "; ".join(problems))
    finally:
        state.close()


def migrate(sqlite_path: Path, database_url: str, *, skip_cooldowns: bool = False) -> dict[str, int]:
    if not sqlite_path.is_file():
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
    files_by_name = _raw_files_by_name(RAW_EPUBS_DIR)
    state = StateManager(database_url=database_url)
    state.close()

    import psycopg

    uri = sqlite_path.resolve().as_uri() + "?mode=ro&immutable=1"
    source = sqlite3.connect(uri, uri=True)
    source.row_factory = sqlite3.Row
    _require_progress_files(source, files_by_name)
    copy_tables = [
        name for name in _TABLES if not (skip_cooldowns and name == "key_cooldowns")
    ]
    counts: dict[str, int] = {}
    try:
        with psycopg.connect(database_url) as dest:
            with dest.transaction():
                dest.execute(
                    "ALTER TABLE key_cooldowns "
                    "ALTER COLUMN retry_at_ms TYPE BIGINT"
                )
                dest.execute(
                    "TRUNCATE TABLE "
                    + ", ".join(_TABLES)
                    + " RESTART IDENTITY"
                )
                for table in copy_tables:
                    counts[table] = _copy_table(source, dest, table, files_by_name)
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
    _verify_hadith_resume(database_url)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    merged = ROOT / "merged-data" / "state.db"
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=merged if merged.is_file() else STATE_DB_PATH,
    )
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
