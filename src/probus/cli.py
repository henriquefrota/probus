import typer

from probus.report import to_json, to_markdown
from probus.runner import Runner

app = typer.Typer(
    name="probus",
    help="Probus — model risk checks for quantitative research.",
    add_completion=False,
)


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
