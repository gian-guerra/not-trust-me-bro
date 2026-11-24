"""Command-line interface. Keep it thin: parse arguments, call the library, print."""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer
from weaviate.exceptions import WeaviateStartUpError

from not_trust_me_bro.answers import Answer
from not_trust_me_bro.config import get_settings
from not_trust_me_bro.rag import Rag, find_pdfs, open_rag
from not_trust_me_bro.retrieval import Mode

app = typer.Typer(
    help="A RAG system that answers only from your own sources. No 'trust me bro'.",
    no_args_is_help=True,
)


def fail(message: str) -> typer.Exit:
    typer.secho(message, fg="red", err=True)
    return typer.Exit(1)


@app.command()
def ingest(
    path: Annotated[
        Path, typer.Argument(help="A PDF, or a folder of PDFs (searched recursively).")
    ],
    force: Annotated[bool, typer.Option(help="Re-ingest even if nothing changed.")] = False,
) -> None:
    """Add documents to your library. Unchanged files are skipped."""
    try:
        pdfs = find_pdfs(path)
    except (FileNotFoundError, ValueError) as e:
        raise fail(f"Can't ingest {path}: {e}") from None
    if not pdfs:
        raise fail(f"No PDFs found in {path}")

    _show_progress_logs()
    failed = 0
    with _rag() as rag:
        for i, pdf in enumerate(pdfs, 1):
            typer.secho(f"[{i}/{len(pdfs)}] {pdf.name}", bold=True)
            start = time.perf_counter()
            try:
                result = rag.ingest(pdf, force=force)
            except Exception as e:  # one bad file must not stop the others
                failed += 1
                typer.secho(f"      ✗ failed: {e}", fg="red")
                continue
            took = time.perf_counter() - start
            if result.status == "unchanged":
                typer.echo(f"      = unchanged since last ingest ({result.chunks} chunks), skipped")
            else:
                typer.secho(f"      ✓ ingested: {result.chunks} chunks · {took:.1f}s", fg="green")

    typer.echo(f"\n{len(pdfs) - failed}/{len(pdfs)} documents ready.")
    if failed:
        raise typer.Exit(1)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question, in quotes.")],
    source: Annotated[
        list[str] | None,
        typer.Option("--source", "-s", help="Only search this document (repeatable)."),
    ] = None,
    k: Annotated[int | None, typer.Option(help="Chunks to retrieve.")] = None,
    mode: Annotated[Mode | None, typer.Option(help="semantic, bm25 or hybrid.")] = None,
    model: Annotated[str | None, typer.Option(help="e.g. openai/gpt-6-luna")] = None,
    think: Annotated[
        bool | None, typer.Option("--think/--no-think", help="Local thinking models only.")
    ] = None,
    show_sources: Annotated[
        bool, typer.Option("--show-sources", help="Print the text of every source.")
    ] = False,
) -> None:
    """Answer a question from your documents, with citations."""
    with _rag() as rag:
        if source:
            unknown = set(source) - set(rag.sources())
            if unknown:
                raise fail(
                    f"Not in your library: {', '.join(sorted(unknown))} (see `ntmb sources`)"
                )
        try:
            result = rag.ask(question, sources=source, k=k, mode=mode, model=model, think=think)
        except Exception as e:  # LLM unreachable, bad key, unknown model…
            raise fail(f"The LLM call failed: {e}") from None
    _print_answer(result, show_sources)


@app.command()
def sources() -> None:
    """List the documents in your library."""
    with _rag() as rag:
        docs = rag.sources()
    if not docs:
        typer.echo("Your library is empty. Add documents with `ntmb ingest <path>`.")
    for name, chunks in docs.items():
        typer.echo(f"  {name}  ({chunks} chunks)")


@app.command()
def config() -> None:
    """Show the current settings (secrets are masked)."""
    for name, value in get_settings().model_dump().items():
        typer.echo(f"{name}: {value}")


@contextmanager
def _rag() -> Iterator[Rag]:
    """Open the library, turning "Weaviate isn't running" into a readable message."""
    try:
        session = open_rag(get_settings())
        rag = session.__enter__()
    except WeaviateStartUpError:
        raise fail("Can't reach Weaviate. Start it with `docker compose up -d`.") from None
    try:
        yield rag
    finally:
        session.__exit__(None, None, None)


def _show_progress_logs() -> None:
    """Print our own INFO logs (one line per ingestion stage). Libraries stay at WARNING."""
    logger = logging.getLogger("not_trust_me_bro")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("      %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


def _print_answer(a: Answer, show_sources: bool) -> None:
    color = "yellow" if a.refused or a.uncited else None
    typer.secho(f"\n{a.text}\n", fg=color)
    if a.refused and a.chunks:
        # Shown anyway: they tell you if the refusal was right or the search missed.
        typer.secho("Closest passages searched (none answered it)", bold=True)
    elif a.chunks:
        typer.secho("Sources (★ = cited)", bold=True)
    for i, c in enumerate(a.chunks, 1):
        star = "★" if c in a.cited else " "
        section = f"  § {c.section}" if c.section else ""
        typer.echo(f" {star} [{i}] {c.citation}{section}")
        if show_sources:
            typer.secho("      " + c.text.replace("\n", "\n      ") + "\n", dim=True)

    if a.uncited:
        typer.secho(
            "\n⚠ No citations to your sources: treat this answer as unverified.", fg="yellow"
        )
    if a.invalid_citations:
        typer.secho(
            f"\n⚠ Cited sources that don't exist: {a.invalid_citations}. Check those claims.",
            fg="yellow",
        )


def main() -> None:
    app()
