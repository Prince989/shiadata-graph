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
from src.agents.errors import AllKeysExhausted, ProviderServerError
from src.agents.gemini import GeminiAgent
from src.core.neo4j_export import export_neo4j
from src.core.phase2 import run_phase2
from src.pipelines.catalog import load_book_catalog, resolve_book
from src.pipelines.proposals import (
    DEFAULT_THRESHOLD,
    apply_promotions,
    build_plan,
    collect,
    report,
    scan_output_dir,
)
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
    except ProviderServerError as exc:
        typer.echo(
            "Gemini temporarily unavailable (503/5xx). "
            "Progress saved — re-run the same command later.\n"
            f"{exc}",
            err=True,
        )
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
    except ProviderServerError as exc:
        typer.echo(
            "Gemini temporarily unavailable (503/5xx). "
            "Progress saved — re-run the same command later.\n"
            f"{exc}",
            err=True,
        )
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
    typer.echo(
        f"gemini_keys={len(settings.google_api_keys)} "
        f"models={','.join(settings.gemini_models)}"
    )
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


@app.command("resolve-nodes")
def resolve_nodes(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report without writing nodes back into payloads"
    ),
) -> None:
    """Resolve every mention in the corpus into canonical graph nodes.

    Runs between phase 1 and phase 2. Identity is a property of the whole
    corpus -- knowing that عقل المرء is العقل needs العقل to have been seen
    somewhere else -- so it cannot be settled while extracting one page.
    """
    _setup_logging()
    from src.pipelines import resolve_pass

    stats = resolve_pass.run(write=not dry_run)
    if not stats.get("mentions"):
        typer.echo("no mentions found; run run-phase1 first")
        raise typer.Exit(code=1)
    for name, value in stats.items():
        typer.echo(f"{name:<12} {value}")
    if dry_run:
        typer.echo("(dry run: nothing written)")


@app.command("proposals")
def proposals(
    threshold: int = typer.Option(
        DEFAULT_THRESHOLD,
        "--threshold",
        help="Distinct hadiths that must propose a term before it is promoted",
    ),
    promote: bool = typer.Option(
        False, "--promote", help="Write promoted terms into the concept catalog"
    ),
) -> None:
    """Review what the model asked for that the vocabulary could not express.

    Terms several narrations independently propose are real topics; terms only
    one narration wanted are that narration's phrasing. Run without --promote to
    see the plan, then again with it to apply.
    """
    _setup_logging()
    payloads = scan_output_dir()
    if not payloads:
        typer.echo("no phase-1 payloads found; run run-phase1 first")
        raise typer.Exit(code=1)
    plan = build_plan(collect(payloads), threshold=threshold)
    typer.echo(report(plan))
    if promote:
        added = apply_promotions(plan, threshold=threshold)
        typer.echo(f"\npromoted {added} terms into the catalog")
        typer.echo("re-run run-phase1 so extraction can select them")


if __name__ == "__main__":
    app()
