"""Spectramotus command line: `spectramotus --help`."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from spectramotus import __version__

app = typer.Typer(no_args_is_help=True, help="Spectramotus v4 - spectramorphize photos into living-portrait packs.")
models_app = typer.Typer(no_args_is_help=True, help="Model files (MediaPipe tasks, wake word).")
app.add_typer(models_app, name="models")
console = Console()


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"spectramotus {__version__}")


@app.command()
def doctor() -> None:
    """Check the environment: single OpenCV, MPS, CoreML, ffmpeg codecs, weights, models, keys."""
    from spectramotus.doctor import FAIL, OK, run_all

    checks = run_all()
    table = Table(title="spectramotus doctor", show_lines=False)
    table.add_column("check", style="bold")
    table.add_column("status")
    table.add_column("detail")
    colors = {OK: "green", "WARN": "yellow", FAIL: "red"}
    for c in checks:
        table.add_row(c.name, f"[{colors[c.status]}]{c.status}[/]", c.detail)
    console.print(table)
    failed = [c for c in checks if c.status == FAIL]
    raise typer.Exit(code=1 if failed else 0)


@models_app.command("download")
def models_download(force: bool = typer.Option(False, help="Re-download even if present.")) -> None:
    """Download the MediaPipe task models into models/mediapipe/."""
    from spectramotus.sense.models import download_models

    console.print("Downloading MediaPipe models...")
    written = download_models(force=force, log=console.print)
    console.print(f"done ({len(written)} new file(s)).")


if __name__ == "__main__":
    app()
