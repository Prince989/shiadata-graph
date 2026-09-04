"""CLI for shiadata-graph ETL phases.

Agents (Gemini + embeddings) are constructed once and reused. Later phases
should import GeminiAgent / EmbeddingAgent from src.agents rather than
opening a new SDK client.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from config.paths import OUTPUT_DIR
from config.settings import get_settings
from src.agents.embeddings import EmbeddingAgent
from src.agents.errors import AllKeysExhausted
from src.agents.gemini import GeminiAgent
from src.core.neo4j_export import export_neo4j
from src.core.phase2 import run_phase2
from src.pipelines.catalog import load_book_catalog, resolve_book
from src.pipelines.reset import reset_catalog_book
from src.pipelines.runner import run_phase1
from src.state_manager import StateManager

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _setup_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _stack() -> tuple[StateManager, GeminiAgent, EmbeddingAgent]:
    settings = get_settings()
    state = StateManager(settings.state_db)
    gemini = GeminiAgent(state, settings)
    embeddings = EmbeddingAgent(settings, state)
    return state, gemini, embeddings


@app.command("run-phase1")
def phase1(
    book: str = typer.Option(..., "--book", help="Catalog id, e.g. al-kafi"),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Max pages/chunks this run across all volume files (not per file)",
    ),
) -> None:
    """Parse a book and run Gemini structured extraction."""
    _setup_logging()
    state, gemini, _ = _stack()
    try:
        stats = run_phase1(book, state, gemini, limit=limit)
    except AllKeysExhausted as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    finally:
        state.close()
    typer.echo(stats)


@app.command("run-phase2")
def phase2(
    book: str = typer.Option(..., "--book"),
) -> None:
    """Embed, canonicalise duplicates, and classify SUPPORTS/CONTRADICTS/EXCEPTS."""
    _setup_logging()
    state, gemini, embeddings = _stack()
    try:
        stats = run_phase2(book, state, gemini, embeddings)
    except AllKeysExhausted as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    finally:
        state.close()
    typer.echo(stats)


@app.command("export-neo4j")
def export_cmd(
    dest: Path | None = typer.Option(None, "--dest"),
) -> None:
    _setup_logging()
    settings = get_settings()
    state = StateManager(settings.state_db)
    path = export_neo4j(state, dest)
    state.close()
    typer.echo(f"Wrote {path}")


@app.command("status")
def status(
    book: str | None = typer.Option(None, "--book"),
) -> None:
    settings = get_settings()
    state = StateManager(settings.state_db)
    typer.echo(state.counts(book))
    typer.echo(f"gemini_keys={len(settings.google_api_keys)} model={settings.gemini_model}")
    typer.echo(f"embed_model={settings.embedding_model}")
    typer.echo(f"raw_data={settings.raw_data_dir}")
    state.close()


@app.command("reset-book")
def reset_book_cmd(
    book: str = typer.Option(..., "--book", help="Catalog id, e.g. hadith"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not prompt"),
    keep_outputs: bool = typer.Option(
        False, "--keep-outputs", help="Leave data/output/phase1/<book>/*.json"
    ),
    cooldowns: bool = typer.Option(
        False, "--cooldowns", help="Also clear Gemini key cooldowns (all books)"
    ),
) -> None:
    """Drop SQLite chunks/buffer/progress for a book so Phase 1 starts over."""
    _setup_logging()
    settings = get_settings()
    delete_outputs = not keep_outputs
    extra = " and Phase 1 JSON" if delete_outputs else ""
    extra += " and Gemini cooldowns" if cooldowns else ""
    if not yes and not typer.confirm(f"Reset '{book}' SQLite state{extra}?"):
        raise typer.Abort()
    state = StateManager(settings.state_db)
    try:
        stats = reset_catalog_book(
            state,
            book,
            output_dir=OUTPUT_DIR,
            delete_outputs=delete_outputs,
            clear_cooldowns=cooldowns,
        )
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    finally:
        state.close()
    typer.echo(stats)


@app.command("list-books")
def list_books() -> None:
    catalog = load_book_catalog()
    raw = get_settings().raw_data_dir
    for book_id, entry in catalog.items():
        try:
            spec = resolve_book(book_id, raw)
            typer.echo(f"{book_id}\t{entry['pipeline']}\t{len(spec.files)} files")
        except FileNotFoundError:
            typer.echo(f"{book_id}\t{entry['pipeline']}\tMISSING")


if __name__ == "__main__":
    app()
