from typing import Literal

import click
from pydantic import BaseModel

from ytverbatim.config import LLMOptions, load_options
from ytverbatim.errors import YtverbatimError
from ytverbatim.llm import (
    LLMRequest,
    Readiness,
    UsageMeter,
    claude_cli,
    client_from_options,
    complete_json,
    openai_compat,
)


class _Probe(BaseModel):
    ok: Literal[True]


@click.group()
def llm() -> None:
    """Check the configured LLM transport."""


@llm.command()
@click.option("--provider", type=click.Choice(["claude-cli", "openai-compat"]), help="Override [llm] provider.")
@click.option("--model", help="Override [llm] model.")
@click.option("--effort", type=click.Choice(claude_cli.EFFORTS), help="Override [llm] effort (claude-cli only).")
@click.option("--probe", is_flag=True, help='Also send one tiny generation asking for {"ok": true}; this spends quota.')
def check(provider: str | None, model: str | None, effort: str | None, *, probe: bool) -> None:
    """Report whether the LLM transport is ready. Sends no generation unless --probe is given."""
    try:
        options = load_options(overrides={"llm": {"provider": provider, "model": model, "effort": effort}}).llm
        if options.provider == "none":
            raise click.ClickException("no LLM provider configured; set [llm] provider or pass --provider")
        status = _readiness(options)
        click.echo(f"{options.provider}: {'ready' if status.ready else 'not ready'} - {status.detail}")
        if not status.ready:
            raise SystemExit(1)
        if probe:
            _probe(options)
    except YtverbatimError as exc:
        raise click.ClickException(str(exc)) from exc


def _readiness(options: LLMOptions) -> Readiness:
    if options.provider == "claude-cli":
        return claude_cli.readiness()
    return openai_compat.readiness(options.base_url, options.api_key_env)


def _probe(options: LLMOptions) -> None:
    meter = UsageMeter()
    client = meter.metered("probe", client_from_options(options))
    request = LLMRequest(system="You are a connectivity probe.", user='Return {"ok": true}.', max_output_tokens=256)
    result = complete_json(client, request, _Probe)
    (usage,) = meter.summary()
    click.echo(
        f"probe {client.id}: {result.model_dump_json()} "
        f"({usage.calls} call(s), input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens})"
    )
