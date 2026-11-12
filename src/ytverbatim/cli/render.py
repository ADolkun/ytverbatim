from pathlib import Path

import click

from ytverbatim import atomic
from ytverbatim.contract import load_document
from ytverbatim.errors import ContractError
from ytverbatim.render import render_markdown
from ytverbatim.ytdlp.runner import redact


@click.command(name="render")
@click.argument("context", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--stdout", "to_stdout", is_flag=True, help="Print the markdown instead of writing it.")
def render_command(context: Path, *, to_stdout: bool) -> None:
    """Render a context.json as markdown, written beside it with the .md suffix."""
    try:
        markdown = render_markdown(load_document(context))
    except ContractError as exc:
        raise click.ClickException(redact(str(exc))) from exc
    if to_stdout:
        click.echo(markdown, nl=False)
        return
    target = context.with_suffix(".md")
    atomic.write_text(target, markdown)
    click.echo(redact(str(target)))
