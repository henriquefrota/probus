import sys

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

import typer

from probus import __version__
from probus.report import to_json, to_markdown
from probus.runner import Runner

app = typer.Typer(
    name="probus",
    help="Probus — model risk checks for quantitative research.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"probus {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show the Probus version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Probus — model risk checks for quantitative research."""


@app.command()
def audit(
    path: str = typer.Argument(..., help="Python file or directory to audit."),
    format: str = typer.Option(
        "markdown", "--format", "-f", help="Output format: markdown or json."
    ),
) -> None:
    """Run model risk checks on a Python file or directory."""
    runner = Runner()
    findings = runner.run(path)

    if format == "json":
        typer.echo(to_json(findings))
    else:
        typer.echo(to_markdown(findings, path))

    if findings:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
