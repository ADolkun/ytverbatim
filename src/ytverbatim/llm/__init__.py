from dataclasses import replace
from typing import Literal, cast

from ytverbatim.config import LLMOptions
from ytverbatim.errors import ConfigError
from ytverbatim.llm import claude_cli, openai_compat
from ytverbatim.llm.claude_cli import ClaudeCLIClient
from ytverbatim.llm.jsonio import complete_json, extract_json_object
from ytverbatim.llm.openai_compat import OpenAICompatClient
from ytverbatim.llm.protocol import LLMClient, LLMRequest, LLMResponse, Readiness, Usage
from ytverbatim.llm.usage import UsageMeter

__all__ = [
    "ClaudeCLIClient",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "OpenAICompatClient",
    "Readiness",
    "Usage",
    "UsageMeter",
    "apply_spec",
    "check_ready",
    "client_from_options",
    "complete_json",
    "extract_json_object",
]

Provider = Literal["claude-cli", "openai-compat"]
_PROVIDERS = ("claude-cli", "openai-compat")


def client_from_options(options: LLMOptions) -> ClaudeCLIClient | OpenAICompatClient:
    """Build the configured transport; options the provider would ignore are rejected rather than dropped."""
    if options.provider == "none":
        raise ConfigError("no LLM provider configured; set [llm] provider to claude-cli or openai-compat")
    model = options.model or ""
    if options.provider == "claude-cli":
        stray = [name for name in ("base_url", "api_key_env") if getattr(options, name) is not None]
        if stray:
            raise ConfigError(f"[llm] {', '.join(stray)} do not apply to claude-cli")
        return ClaudeCLIClient(model=model, effort=options.effort, timeout_s=options.timeout_s)
    if options.effort is not None:
        raise ConfigError("[llm] effort does not apply to openai-compat")
    return OpenAICompatClient(
        base_url=options.base_url or "",
        model=model,
        api_key_env=options.api_key_env,
        timeout_s=options.timeout_s,
    )


def apply_spec(options: LLMOptions, spec: str) -> LLMOptions:
    """Apply a `--llm` spec: `none`, or `PROVIDER:MODEL` with a full model id (`claude-cli:claude-sonnet-5-5`).

    The same provider keeps its configured settings (effort, base URL, key). Another provider starts from its
    defaults: a configured `base_url` must not block `claude-cli`, nor a configured `effort` block `openai-compat`.
    """
    if spec == "none":
        return LLMOptions(provider="none", timeout_s=options.timeout_s)
    provider, _, model = spec.partition(":")
    if provider not in _PROVIDERS or not model:
        known = ", ".join(_PROVIDERS)
        raise ConfigError(f"--llm takes none or PROVIDER:MODEL with PROVIDER one of {known}; got {spec!r}")
    if provider == options.provider:
        return replace(options, model=model)
    return LLMOptions(provider=cast("Provider", provider), model=model, timeout_s=options.timeout_s)


def check_ready(options: LLMOptions) -> None:
    """Fail before any work when the configured transport cannot run: no `claude` CLI, or no base URL or key.

    Cheap and local: it sends nothing and spends nothing; `ytverbatim llm check --probe` makes a real call.
    """
    if options.provider == "claude-cli":
        claude_cli.executable()
    elif options.provider == "openai-compat":
        if not options.base_url:
            raise ConfigError("openai-compat needs [llm] base_url, e.g. http://127.0.0.1:8080/v1")
        openai_compat.check_key(options.api_key_env)
