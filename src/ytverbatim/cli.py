import click

from ytverbatim import __version__


@click.group()
@click.version_option(__version__, prog_name="ytverbatim")
def main() -> None:
    """Produce verified, timestamped YouTube transcripts for LLM context."""
