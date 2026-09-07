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
    page: str | None = typer.Option(
        None,
        "--page",
        help=(
            "Only process this print page (debug). "
            "Bare number like 30, or locator fragment like 'جلد 1 - صفحه 30'. "
            "Does not advance resume progress."
        ),
    ),
    volume: str | None = typer.Option(
        None,
        "--volume",
        help="With --page, restrict to جلد N (e.g. --volume 1)",
    ),
) -> None:
    """Parse a book and run Gemini structured extraction."""
    _setup_logging()
    state, gemini, _ = _stack()
    try:
        stats = run_phase1(book, state, gemini, limit=limit, page=page, volume=volume)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
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


@app.command("harvest-ontology")
def harvest_ontology(
    min_df: int = typer.Option(2, "--min-df", help="Corpus df before a term is promoted"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report without writing"),
) -> None:
    """Build the concept catalog from the books' own chapter headings.

    Layers 1-3 of the enrichment ladder, all free and deterministic:
    headings become terms, long titles are mined for the terms inside them, and
    labels the corpus keeps reaching for are promoted on the same evidence.

    Writes config/derived_ontology.yaml only. base_ontology.yaml -- the
    hand-written aliases and judgement calls -- is never touched.
    """
    _setup_logging()
    from src.pipelines import harvest as harvester

    # Build from the hand-written catalog only. Reading the derived file the
    # harvest is about to replace lets its own junk cite itself as evidence.
    if not dry_run:
        harvester.reset_derived()
    found = harvester.scan()
    typer.echo(f"layer 1 (headings)     {found.summary()}")
    if not dry_run:
        # Layer 2 decomposes against layer 1, so layer 1 has to be on disk first.
        harvester.write(found)
    mined = harvester.mine_long_titles(found)
    typer.echo(f"layer 2 (decomposed)   +{mined}")
    promoted = harvester.promote_recurring(found, min_df=min_df)
    typer.echo(f"layer 3 (recurring)    +{promoted}")
    # Only now has every layer voted, so only now can `broader` be settled.
    # Deciding it while scanning made it first-write-wins, which is filename
    # order: الميراث was printed 150 times, mostly under كتاب الفرائض, and got
    # النكاح because al-Kafi 5 sorts before Faqih 4.
    settled = harvester.settle_parents(found)
    typer.echo(f"parents (majority vote) {settled}")
    if dry_run:
        typer.echo(harvester.report(found))
        typer.echo("(dry run: nothing written)")
        return
    written = harvester.write(found)
    typer.echo(f"wrote {written} concepts to config/derived_ontology.yaml")


@app.command("adjudicate")
def adjudicate_cmd(
    apply: bool = typer.Option(False, "--apply", help="Write accepted verdicts to the catalog"),
    limit: int | None = typer.Option(None, "--limit", help="Max orphans to ask about"),
    ask: bool = typer.Option(True, "--ask/--no-ask", help="Call Gemini for uncached orphans"),
) -> None:
    """Layer 4: judge the labels no catalog or morphology could resolve.

    Asked once per distinct label, offline, and cached in config/adjudicated.json
    so the same string is never paid for twice. Run --no-ask to see the residue
    and replay the cache without spending anything.
    """
    _setup_logging()
    from src.pipelines import adjudicate as adj

    orphans = adj.find_orphans()
    if not orphans:
        typer.echo("no orphan labels; layers 1-3 resolved everything")
        return
    cache = adj.load_cache()
    if ask:
        _, gemini, _ = _stack()
        try:
            cache = adj.adjudicate(gemini, orphans, cache, limit=limit)
        except AllKeysExhausted as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc
        adj.save_cache(cache)
    plan = adj.build_plan(cache)
    typer.echo(adj.report(orphans, plan))
    if apply:
        rows = adj.apply_plan(plan)
        typer.echo(f"appended {rows} adjudicated concepts; re-run resolve-nodes")


@app.command("enrich-aliases")
def enrich_aliases(
    apply: bool = typer.Option(False, "--apply", help="Write config/derived_aliases.yaml"),
    limit: int | None = typer.Option(None, "--limit", help="Max concepts to ask about"),
    ask: bool = typer.Option(False, "--ask/--no-ask", help="Call Gemini for uncached concepts"),
) -> None:
    """Find the other wordings each concept is written in.

    The catalog knows a concept by its chapter title only, so a narration that
    phrases the topic any other way misses it. Synonymy is the one thing these
    books did not write down -- string similarity pairs الزاني with الزانية, and
    the corpus's own glosses cannot be parsed reliably -- so this asks a model.

    It does not trust the answer. A proposal is kept only where the corpus
    actually uses that wording, and only if no other concept already claims it.
    Cached per concept, so a re-run spends nothing. Default is --no-ask: replay
    the cache and see the plan for free.
    """
    _setup_logging()
    from src.pipelines import aliases as al

    concepts = al.catalog_concepts()
    cache = al.load_cache()
    typer.echo(f"catalog concepts {len(concepts)}, already asked {len(cache)}")
    if ask:
        _, gemini, _ = _stack()
        try:
            cache = al.propose(gemini, concepts, cache, limit=limit)
        except AllKeysExhausted as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc
        al.save_cache(cache)
    if not cache:
        typer.echo("nothing proposed yet; re-run with --ask")
        raise typer.Exit(code=1)
    kept, stats = al.ground(cache)
    typer.echo(al.report(kept, stats))
    if apply:
        written = al.write(kept)
        typer.echo(f"wrote {written} aliases across {len(kept)} concepts")


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
