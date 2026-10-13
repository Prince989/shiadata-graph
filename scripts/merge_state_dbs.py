"""Build merged-data/state.db from the two reference state databases.

another-data supplies every hadith chunk, progress row, buffer row, and Gemini
cooldown. data supplies the al-Mizan chunks and the Groq cooldowns. Jobs,
embeddings, and edge pairs are left empty.

Both sources are opened with mode=ro&immutable=1. This script never inserts,
updates, deletes, or vacuums either reference database. The only database it
creates is merged-data/state.db.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.state_manager import StateManager

ANOTHER = ROOT / "another-data" / "state.db"
DATA = ROOT / "data" / "state.db"
MERGED = ROOT / "merged-data" / "state.db"

VOLUME_5_LOCATOR = "جلد 5 - صفحه 95"


def ro_connect(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _snapshot(paths: list[Path]) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for path in paths:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(path) + suffix) if suffix else path
            if candidate.exists():
                stat = candidate.stat()
                found[str(candidate)] = (stat.st_size, stat.st_mtime_ns)
    return found


def _assert_sources_unchanged(before: dict[str, tuple[int, int]]) -> None:
    after = _snapshot([ANOTHER, DATA])
    changed = [path for path, stat in before.items() if after.get(path) != stat]
    missing = [path for path in before if path not in after]
    if changed or missing:
        raise SystemExit(f"reference databases changed: {changed or missing}")


def _copy_query(
    source: sqlite3.Connection,
    dest: sqlite3.Connection,
    table: str,
    select_sql: str,
    params: tuple = (),
) -> int:
    cursor = source.execute(select_sql, params)
    columns = [item[0] for item in cursor.description]
    rows = cursor.fetchall()
    if not rows:
        return 0
    placeholders = ", ".join("?" for _ in columns)
    names = ", ".join(columns)
    dest.executemany(
        f"INSERT INTO {table} ({names}) VALUES ({placeholders})",
        [tuple(row) for row in rows],
    )
    return len(rows)


def merge() -> None:
    if MERGED.resolve() in {ANOTHER.resolve(), DATA.resolve()}:
        raise SystemExit("refusing to write a reference database")
    MERGED.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(MERGED) + suffix) if suffix else MERGED
        if candidate.exists():
            candidate.unlink()

    another = ro_connect(ANOTHER)
    data = ro_connect(DATA)
    state = StateManager(MERGED, database_url="")
    try:
        dest = state._conn
        hadith_chunks = _copy_query(
            another,
            dest,
            "chunks",
            "SELECT * FROM chunks WHERE book_id = 'hadith'",
        )
        mizan_chunks = _copy_query(
            data,
            dest,
            "chunks",
            "SELECT * FROM chunks WHERE book_id = 'al-mizan'",
        )
        progress = _copy_query(another, dest, "hadith_progress", "SELECT * FROM hadith_progress")
        buffer_rows = _copy_query(another, dest, "hadith_buffer", "SELECT * FROM hadith_buffer")
        gemini = _copy_query(
            another,
            dest,
            "key_cooldowns",
            "SELECT * FROM key_cooldowns WHERE key_id LIKE 'gemini-%'",
        )
        groq = _copy_query(
            data,
            dest,
            "key_cooldowns",
            "SELECT * FROM key_cooldowns WHERE key_id LIKE 'groq-%'",
        )
        dest.commit()
        dest.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        state.close()
        another.close()
        data.close()

    print(
        "merged "
        f"hadith_chunks={hadith_chunks} al_mizan_chunks={mizan_chunks} "
        f"progress={progress} buffer={buffer_rows} gemini={gemini} groq={groq}"
    )


def _count(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0])


def check() -> None:
    if not MERGED.exists():
        raise SystemExit(f"missing {MERGED}")

    before_sources = _snapshot([ANOTHER, DATA])
    another = ro_connect(ANOTHER)
    data = ro_connect(DATA)
    state = StateManager(MERGED, database_url="")
    try:
        conn = state._conn
        processed = _count(
            conn,
            "SELECT COUNT(*) FROM chunks WHERE book_id = 'hadith' AND status = 'PROCESSED_PHASE1'",
        )
        errors = _count(
            conn,
            "SELECT COUNT(*) FROM chunks WHERE book_id = 'hadith' AND status = 'ERROR'",
        )
        pending = _count(
            conn,
            "SELECT COUNT(*) FROM chunks WHERE book_id = 'al-mizan' AND status = 'PENDING'",
        )
        chunk_total = _count(conn, "SELECT COUNT(*) FROM chunks")
        distinct_ids = _count(conn, "SELECT COUNT(DISTINCT id) FROM chunks")
        progress_rows = conn.execute(
            "SELECT source_path, last_locator FROM hadith_progress WHERE book_id = 'hadith' ORDER BY source_path"
        ).fetchall()
        buffer_rows = conn.execute(
            "SELECT source_path FROM hadith_buffer WHERE book_id = 'hadith'"
        ).fetchall()

        expected_hadith = _count(another, "SELECT COUNT(*) FROM chunks WHERE book_id = 'hadith'")
        expected_mizan = _count(data, "SELECT COUNT(*) FROM chunks WHERE book_id = 'al-mizan'")
        expected_gemini = _count(
            another, "SELECT COUNT(*) FROM key_cooldowns WHERE key_id LIKE 'gemini-%'"
        )
        expected_groq = _count(
            data, "SELECT COUNT(*) FROM key_cooldowns WHERE key_id LIKE 'groq-%'"
        )
        gemini = _count(conn, "SELECT COUNT(*) FROM key_cooldowns WHERE key_id LIKE 'gemini-%'")
        groq = _count(conn, "SELECT COUNT(*) FROM key_cooldowns WHERE key_id LIKE 'groq-%'")
        cooldown_total = _count(conn, "SELECT COUNT(*) FROM key_cooldowns")
        jobs = _count(conn, "SELECT COUNT(*) FROM jobs")
        embeddings = _count(conn, "SELECT COUNT(*) FROM embeddings")
        edges = _count(conn, "SELECT COUNT(*) FROM edge_pairs")

        problems: list[str] = []
        if processed != 8417:
            problems.append(f"processed {processed} != 8417")
        if errors != 7:
            problems.append(f"errors {errors} != 7")
        if pending != 1133:
            problems.append(f"al-mizan pending {pending} != 1133")
        if chunk_total != expected_hadith + expected_mizan:
            problems.append(
                f"chunks {chunk_total} != {expected_hadith} + {expected_mizan}"
            )
        if distinct_ids != chunk_total:
            problems.append(f"duplicate chunk ids: {chunk_total - distinct_ids}")
        if len(progress_rows) != 5:
            problems.append(f"hadith_progress rows {len(progress_rows)} != 5")
        volume_paths = [str(row["source_path"]) for row in progress_rows]
        if [Path(path).name for path in volume_paths] != [f"al-kafi-{n}.txt" for n in range(1, 6)]:
            problems.append(f"progress volumes {volume_paths}")
        volume_5 = next(
            (row for row in progress_rows if str(row["last_locator"]) == VOLUME_5_LOCATOR),
            None,
        )
        if volume_5 is None:
            problems.append("volume 5 locator missing")
        elif state.get_hadith_progress("hadith", str(volume_5["source_path"])) != VOLUME_5_LOCATOR:
            problems.append("get_hadith_progress did not return the volume 5 locator")
        if len(buffer_rows) != 1:
            problems.append(f"hadith_buffer rows {len(buffer_rows)} != 1")
        elif volume_5 is not None and str(buffer_rows[0]["source_path"]) != str(volume_5["source_path"]):
            problems.append("buffer is not the volume 5 source")
        elif volume_5 is not None and state.get_hadith_buffer("hadith", str(volume_5["source_path"])) is None:
            problems.append("get_hadith_buffer returned nothing for volume 5")
        if gemini != expected_gemini or groq != expected_groq or cooldown_total != gemini + groq:
            problems.append(
                f"cooldowns gemini={gemini}/{expected_gemini} groq={groq}/{expected_groq}"
            )
        if jobs or embeddings or edges:
            problems.append(f"unexpected rows jobs={jobs} embeddings={embeddings} edges={edges}")

        if problems:
            raise SystemExit("check failed: " + "; ".join(problems))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        state.close()
        another.close()
        data.close()

    merged_stat = MERGED.stat()
    merged_stamp = (merged_stat.st_size, merged_stat.st_mtime_ns)
    _write_cooldown_on_copy()
    _assert_sources_unchanged(before_sources)
    after = MERGED.stat()
    if (after.st_size, after.st_mtime_ns) != merged_stamp:
        raise SystemExit("merged state.db changed during the cooldown write check")
    print(
        "ok processed=8417 errors=7 progress=5 buffer=volume-5 "
        "al_mizan_pending=1133 cooldowns=gemini+groq"
    )


def _write_cooldown_on_copy() -> None:
    with tempfile.TemporaryDirectory(prefix="merged-state-write-") as directory:
        copy_path = Path(directory) / "state.db"
        shutil.copy2(MERGED, copy_path)
        probe = StateManager(copy_path, database_url="")
        try:
            probe.set_cooldown("groq-merge-check", "rate_limited", 1, 1, None)
            row = probe._conn.execute(
                "SELECT reason FROM key_cooldowns WHERE key_id = ?",
                ("groq-merge-check",),
            ).fetchone()
            if row is None or row["reason"] != "rate_limited":
                raise SystemExit("cooldown write on the temp copy did not stick")
            probe.clear_cooldown("groq-merge-check")
        finally:
            probe.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge the two state databases.")
    parser.add_argument("--check", action="store_true", help="Validate merged-data/state.db only.")
    args = parser.parse_args()
    before = _snapshot([ANOTHER, DATA])
    if not args.check:
        merge()
        _assert_sources_unchanged(before)
    check()


if __name__ == "__main__":
    main()
