"""Wipe Phase 1/2 SQLite resume state and optional JSON for one catalog book."""

from __future__ import annotations

from pathlib import Path

from src.pipelines.catalog import load_book_catalog
from src.state_manager import StateManager


def reset_catalog_book(
    state: StateManager,
    book_id: str,
    *,
    output_dir: Path,
    delete_outputs: bool = True,
    clear_cooldowns: bool = False,
) -> dict[str, int | str | bool]:
    catalog = load_book_catalog()
    if book_id not in catalog:
        known = ", ".join(sorted(catalog))
        raise KeyError(f"Unknown book '{book_id}'. Known: {known}")
    json_removed = 0
    if delete_outputs:
        folder = output_dir / "phase1" / book_id
        if folder.is_dir():
            for path in folder.glob("*.json"):
                path.unlink()
                json_removed += 1
    chunks_removed = sum(state.counts(book_id).values())
    state.reset_book(book_id)
    if clear_cooldowns:
        state.clear_all_cooldowns()
    return {
        "book_id": book_id,
        "chunks_removed": chunks_removed,
        "json_removed": json_removed,
        "cooldowns_cleared": clear_cooldowns,
    }
