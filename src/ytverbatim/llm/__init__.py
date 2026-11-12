from ytverbatim.config import LLMOptions
from ytverbatim.errors import ConfigError
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
    "client_from_options",
    "complete_json",
    "extract_json_object",
]


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
