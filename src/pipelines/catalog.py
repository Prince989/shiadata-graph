"""Resolve CLI book ids to files and pipeline kinds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from config.paths import BOOKS_YAML, RAW_EPUBS_DIR
from src.pipelines.ontology import load_ontology

__all__ = ["BookSpec", "load_book_catalog", "load_ontology", "resolve_book"]


@dataclass(frozen=True)
class BookSpec:
    book_id: str
    pipeline: str
    description: str
    files: list[Path]


def load_book_catalog() -> dict:
    return yaml.safe_load(BOOKS_YAML.read_text(encoding="utf-8"))


def resolve_book(book_id: str, raw_dir: Path | None = None) -> BookSpec:
    catalog = load_book_catalog()
    if book_id not in catalog:
        known = ", ".join(sorted(catalog))
        raise KeyError(f"Unknown book '{book_id}'. Known: {known}")
    entry = catalog[book_id]
    root = raw_dir or RAW_EPUBS_DIR
    files: list[Path] = []
    for pattern in entry.get("globs", []):
        files.extend(sorted(root.glob(pattern)))
    unique = []
    seen = set()
    for path in files:
        if path.suffix.lower() not in {".txt", ".epub"}:
            continue
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        unique.append(path)
    if not unique:
        raise FileNotFoundError(
            f"No source files for '{book_id}' under {root} (globs={entry.get('globs')})."
        )
    return BookSpec(
        book_id=book_id,
        pipeline=entry["pipeline"],
        description=entry.get("description", ""),
        files=unique,
    )
