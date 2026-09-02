"""SQLite checkpointing for long-running ETL jobs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from config.paths import STATE_DB_PATH


class ChunkStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSED_PHASE1 = "PROCESSED_PHASE1"
    EMBEDDED = "EMBEDDED"
    PROCESSED_PHASE2 = "PROCESSED_PHASE2"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True)
class ChunkRecord:
    id: str
    book_id: str
    pipeline: str
    locator: str
    source_path: str
    text: str
    status: ChunkStatus
    payload_json: str | None = None
    error: str | None = None
    attempts: int = 0
    canonical_id: str | None = None

    def payload(self) -> dict[str, Any] | None:
        if not self.payload_json:
            return None
        return json.loads(self.payload_json)


class StateManager:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path or STATE_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init()

    def close(self) -> None:
        self._conn.close()

    def _init(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                book_id TEXT NOT NULL,
                pipeline TEXT NOT NULL,
                locator TEXT NOT NULL,
                source_path TEXT NOT NULL,
                text TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                canonical_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_book_status
                ON chunks(book_id, status);
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT NOT NULL,
                model TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                PRIMARY KEY (chunk_id, model)
            );
            CREATE TABLE IF NOT EXISTS key_cooldowns (
                key_id TEXT PRIMARY KEY,
                reason TEXT NOT NULL,
                strikes INTEGER NOT NULL,
                retry_at_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edge_pairs (
                pair_id TEXT PRIMARY KEY,
                left_id TEXT NOT NULL,
                right_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                cosine REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phase TEXT NOT NULL,
                book_id TEXT,
                pause_reason TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );
            """
        )
        self._conn.commit()

    def now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def upsert_chunks(self, rows: Iterable[dict[str, str]]) -> int:
        inserted = 0
        ts = self.now()
        for row in rows:
            cur = self._conn.execute(
                """
                INSERT INTO chunks (
                    id, book_id, pipeline, locator, source_path, text,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING
                """,
                (
                    row["id"],
                    row["book_id"],
                    row["pipeline"],
                    row["locator"],
                    row["source_path"],
                    row["text"],
                    ChunkStatus.PENDING.value,
                    ts,
                    ts,
                ),
            )
            inserted += cur.rowcount
        self._conn.commit()
        return inserted

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return self._to_chunk(row) if row else None

    def list_chunks(
        self,
        book_id: str | None = None,
        statuses: list[ChunkStatus] | None = None,
        pipeline: str | None = None,
        limit: int | None = None,
    ) -> list[ChunkRecord]:
        clauses: list[str] = []
        args: list[Any] = []
        if book_id:
            clauses.append("book_id = ?")
            args.append(book_id)
        if pipeline:
            clauses.append("pipeline = ?")
            args.append(pipeline)
        if statuses:
            placeholders = ",".join("?" * len(statuses))
            clauses.append(f"status IN ({placeholders})")
            args.extend(s.value for s in statuses)
        sql = "SELECT * FROM chunks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY source_path, locator"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return [self._to_chunk(r) for r in self._conn.execute(sql, args)]

    def mark(
        self,
        chunk_id: str,
        status: ChunkStatus,
        *,
        payload: dict[str, Any] | None = None,
        error: str | None = None,
        canonical_id: str | None = None,
        bump_attempts: bool = False,
    ) -> None:
        fields = ["status = ?", "updated_at = ?", "error = ?"]
        args: list[Any] = [status.value, self.now(), error]
        if payload is not None:
            fields.append("payload_json = ?")
            args.append(json.dumps(payload, ensure_ascii=False))
        if canonical_id is not None:
            fields.append("canonical_id = ?")
            args.append(canonical_id)
        if bump_attempts:
            fields.append("attempts = attempts + 1")
        args.append(chunk_id)
        self._conn.execute(
            f"UPDATE chunks SET {', '.join(fields)} WHERE id = ?",
            args,
        )
        self._conn.commit()

    def counts(self, book_id: str | None = None) -> dict[str, int]:
        if book_id:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM chunks WHERE book_id = ? GROUP BY status",
                (book_id,),
            )
        else:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM chunks GROUP BY status"
            )
        return {row["status"]: row["n"] for row in rows}

    def save_embedding(self, chunk_id: str, model: str, vector: list[float]) -> None:
        self._conn.execute(
            """
            INSERT INTO embeddings(chunk_id, model, vector_json)
            VALUES (?, ?, ?)
            ON CONFLICT(chunk_id, model) DO UPDATE SET vector_json = excluded.vector_json
            """,
            (chunk_id, model, json.dumps(vector)),
        )
        self._conn.commit()

    def get_embedding(self, chunk_id: str, model: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT vector_json FROM embeddings WHERE chunk_id = ? AND model = ?",
            (chunk_id, model),
        ).fetchone()
        return json.loads(row["vector_json"]) if row else None

    def load_embeddings(self, model: str, chunk_ids: list[str]) -> dict[str, list[float]]:
        if not chunk_ids:
            return {}
        out: dict[str, list[float]] = {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT chunk_id, vector_json FROM embeddings WHERE model = ? AND chunk_id IN ({placeholders})",
            [model, *chunk_ids],
        )
        for row in rows:
            out[row["chunk_id"]] = json.loads(row["vector_json"])
        return out

    def save_edge(self, pair_id: str, left_id: str, right_id: str, relation: str, cosine: float) -> None:
        self._conn.execute(
            """
            INSERT INTO edge_pairs(pair_id, left_id, right_id, relation, cosine)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(pair_id) DO UPDATE SET relation = excluded.relation, cosine = excluded.cosine
            """,
            (pair_id, left_id, right_id, relation, cosine),
        )
        self._conn.commit()

    def has_edge(self, pair_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM edge_pairs WHERE pair_id = ?", (pair_id,)
        ).fetchone()
        return row is not None

    def list_edges(self) -> list[sqlite3.Row]:
        return list(self._conn.execute("SELECT * FROM edge_pairs"))

    def get_cooldown(self, key_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM key_cooldowns WHERE key_id = ?", (key_id,)
        ).fetchone()
        return dict(row) if row else None

    def set_cooldown(self, key_id: str, reason: str, strikes: int, retry_at_ms: int) -> None:
        self._conn.execute(
            """
            INSERT INTO key_cooldowns(key_id, reason, strikes, retry_at_ms)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key_id) DO UPDATE SET
                reason = excluded.reason,
                strikes = excluded.strikes,
                retry_at_ms = excluded.retry_at_ms
            """,
            (key_id, reason, strikes, retry_at_ms),
        )
        self._conn.commit()

    def clear_cooldown(self, key_id: str) -> None:
        self._conn.execute("DELETE FROM key_cooldowns WHERE key_id = ?", (key_id,))
        self._conn.commit()

    def record_job(self, phase: str, book_id: str | None, pause_reason: str | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO jobs(phase, book_id, pause_reason, started_at) VALUES (?, ?, ?, ?)",
            (phase, book_id, pause_reason, self.now()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def finish_job(self, job_id: int, pause_reason: str | None = None) -> None:
        self._conn.execute(
            "UPDATE jobs SET finished_at = ?, pause_reason = ? WHERE id = ?",
            (self.now(), pause_reason, job_id),
        )
        self._conn.commit()

    @staticmethod
    def _to_chunk(row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            id=row["id"],
            book_id=row["book_id"],
            pipeline=row["pipeline"],
            locator=row["locator"],
            source_path=row["source_path"],
            text=row["text"],
            status=ChunkStatus(row["status"]),
            payload_json=row["payload_json"],
            error=row["error"],
            attempts=row["attempts"],
            canonical_id=row["canonical_id"],
        )
