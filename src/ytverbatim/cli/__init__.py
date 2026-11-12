from collections.abc import Mapping
from importlib import import_module
from typing import Any, override

import click

from ytverbatim import __version__

# Subcommand name -> "module:attribute". Modules are imported on first use so a heavy
# subcommand never slows down (or breaks) the others.
LAZY_COMMANDS: Mapping[str, str] = {
    "auth": "ytverbatim.cli.auth:auth",
    "fetch": "ytverbatim.cli.fetch:fetch_command",
    "llm": "ytverbatim.cli.llm:llm",
    "render": "ytverbatim.cli.render:render_command",
    "run": "ytverbatim.cli.run:run_command",
}


class LazyGroup(click.Group):
    def __init__(self, *, lazy_commands: Mapping[str, str], **attrs: Any) -> None:
        super().__init__(**attrs)
        self._lazy_commands = lazy_commands

    @override
    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*super().list_commands(ctx), *self._lazy_commands})

    @override
    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        target = self._lazy_commands.get(cmd_name)
        if target is None:
            return super().get_command(ctx, cmd_name)
        module_name, _, attribute = target.partition(":")
        command = getattr(import_module(module_name), attribute)
        if not isinstance(command, click.Command):
            raise TypeError(f"{target} is not a click command")
        return command


@click.group(cls=LazyGroup, lazy_commands=LAZY_COMMANDS)
@click.version_option(__version__, prog_name="ytverbatim")
def main() -> None:
    """Produce verified, timestamped YouTube transcripts for LLM context."""
