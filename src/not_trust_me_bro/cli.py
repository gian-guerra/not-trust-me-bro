"""Command-line interface. Keep it thin: parse arguments, call the library, print."""

import typer

from not_trust_me_bro.config import get_settings

app = typer.Typer(
    help="A RAG system that answers only from your own sources. No 'trust me bro'.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    # A callback keeps Typer in multi-command mode even while there's only one command.
    pass


@app.command()
def config() -> None:
    """Show the current settings (secrets are masked)."""
    for name, value in get_settings().model_dump().items():
        typer.echo(f"{name}: {value}")


def main() -> None:
    app()
