"""The `claude` CLI as a pure-completion transport on the user's claude.ai subscription.

Every call is a fresh `claude -p` child with no tools, no MCP servers, no settings sources and an allow-listed
environment, and every call re-verifies from the stream that this isolation actually held.
"""

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ytverbatim import paths, proc
from ytverbatim.errors import ConfigError, DependencyError, LLMOutputError, LLMUnavailableError, ProcTimeout
from ytverbatim.llm.jsonio import as_object
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Readiness, Usage

PROVIDER = "claude-cli"
EFFORTS = ("low", "medium", "high", "xhigh", "max")

# Passed through only when set. Everything else, notably ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
# CLAUDE_CODE_OAUTH_TOKEN, ANTHROPIC_BASE_URL, CLAUDE_CODE_USE_BEDROCK/VERTEX and CLAUDE_CODE_EFFORT_LEVEL,
# outranks or replaces the subscription login or the argv, so it must never reach the child.
PASSTHROUGH_ENV = (
    "HOME",
    "PATH",
    "USER",
    "LOGNAME",
    "TMPDIR",
    "LANG",
    "CLAUDE_CONFIG_DIR",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "no_proxy",
    "NODE_EXTRA_CA_CERTS",
)
FIXED_ENV: Mapping[str, str] = {
    # Retry policy belongs to the caller; a silent CLI retry would spend quota outside it.
    "CLAUDE_CODE_MAX_RETRIES": "0",
    # Auto memory is read regardless of --setting-sources.
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    # No auto-update mid-batch and no telemetry.
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}
# A safety-classifier flag must end the run as an error rather than silently switch models.
_SETTINGS = '{"switchModelsOnFlag":false}'
_AUTH_STATUS_TIMEOUT_S = 10.0
_AUTH_ERRORS = frozenset({"authentication_failed", "oauth_org_not_allowed", "account_on_hold", "billing_error"})
_AUTH_HTTP = frozenset({401, 403})
_RATE_LIMIT_HTTP = 429
_TAIL = 500
_FULL_MODEL_PREFIX = "claude-"


def child_env(source: Mapping[str, str] | None = None, *, max_output_tokens: int | None = None) -> dict[str, str]:
    environ = os.environ if source is None else source
    env = {key: environ[key] for key in PASSTHROUGH_ENV if key in environ}
    env.update({key: value for key, value in environ.items() if key.startswith("LC_")})
    env.update(FIXED_ENV)
    if max_output_tokens is not None:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
    return env


def stdin_line(user: str) -> str:
    # client_composed delivers the text verbatim: no @path expansion, and a leading "/" is not a command.
    message = {
        "type": "user",
        "message": {"role": "user", "content": user},
        "parent_tool_use_id": None,
        "client_composed": True,
    }
    return json.dumps(message, ensure_ascii=False) + "\n"


class ClaudeCLIClient:
    def __init__(self, *, model: str, effort: str | None = None, timeout_s: float = 900.0) -> None:
        if not model:
            raise ConfigError("claude-cli needs a model (set [llm] model, e.g. sonnet)")
        if effort is not None and effort not in EFFORTS:
            raise ConfigError(f"claude-cli effort must be one of: {', '.join(EFFORTS)}")
        self._model = model
        self._effort = effort
        self._timeout_s = timeout_s

    @property
    def id(self) -> str:
        # Effort changes the replies, so it is part of the identity that checkpoints are keyed by.
        return f"{PROVIDER}:{self._model}" + (f"@{self._effort}" if self._effort is not None else "")

    def argv(self, executable: str, system_file: Path) -> list[str]:
        effort = ["--effort", self._effort] if self._effort is not None else []
        return [
            executable,
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self._model,
            *effort,
            # A file rather than argv keeps the prompt out of `ps` for every other local user. The equals form keeps
            # a path that starts with "-" from parsing as a flag.
            f"--system-prompt-file={system_file}",
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources=",
            "--settings",
            _SETTINGS,
            "--disable-slash-commands",
            "--permission-mode",
            "dontAsk",
            "--no-session-persistence",
            "--max-turns",
            "1",
        ]

    def complete(self, request: LLMRequest) -> LLMResponse:
        # `json_schema` is not sent: --json-schema works through a hidden tool call and extra turns, which
        # conflict with --tools "" and --max-turns 1; complete_json puts the schema in the prompt instead.
        # Claude Code has no temperature control, so `temperature` cannot be honoured.
        executable = _executable()
        env = child_env(max_output_tokens=request.max_output_tokens)
        # Created 0600 in the private cache, outside the child's empty cwd, and removed when the call ends.
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=paths.llm_dir("claude"), prefix="system-", suffix=".txt"
        ) as system_file:
            system_file.write(request.system)
            system_file.flush()
            try:
                result = proc.run(
                    self.argv(executable, Path(system_file.name)),
                    timeout_s=self._timeout_s,
                    env=env,
                    cwd=paths.llm_workdir("claude"),
                    stdin=stdin_line(request.user),
                )
            except ProcTimeout as exc:
                raise LLMUnavailableError(f"{self.id}: {exc}; usage unknown") from exc
        run = parse_stream(result.stdout)
        if run.init is None or run.result is None:
            raise LLMUnavailableError(
                f"{self.id}: unusable result (exit {result.returncode}); usage unknown{_stderr_tail(result.stderr)}"
            )
        usage = _usage(run.result)
        try:
            check_isolation(run, model=self._model)
        except LLMUnavailableError as exc:
            raise LLMUnavailableError(str(exc), responded=True, usage=usage) from None
        _raise_for_failure(self.id, run, result.returncode, usage)
        # Checked on success only: a failed run reports its own reason, and its turn count says nothing more.
        if run.result.get("num_turns") != 1:
            raise LLMUnavailableError(
                f"{self.id}: the run took {run.result.get('num_turns')!r} turns, not 1; a pure completion has one",
                responded=True,
                usage=usage,
            )
        text = run.result.get("result")
        if not isinstance(text, str) or not text.strip():
            raise LLMOutputError(f"{self.id}: empty reply", responded=True, usage=usage)
        return LLMResponse(text=text, usage=usage, elapsed_s=result.elapsed_s)


@dataclass(frozen=True)
class ClaudeRun:
    init: dict[str, object] | None
    result: dict[str, object] | None
    assistant_error: str | None
    rejected_limit: dict[str, object] | None


def parse_stream(stdout: str) -> ClaudeRun:
    init: dict[str, object] | None = None
    result: dict[str, object] | None = None
    assistant_error: str | None = None
    rejected_limit: dict[str, object] | None = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            decoded: object = json.loads(line)
        except ValueError:
            decoded = None
        if not isinstance(decoded, dict):
            raise LLMUnavailableError("claude CLI printed a malformed stream line; usage unknown")
        message = as_object(decoded)
        kind = message.get("type")
        if kind == "system" and message.get("subtype") == "init":
            if init is not None:
                raise LLMUnavailableError("claude CLI printed two init messages; usage unknown")
            init = message
        elif kind == "result":
            if result is not None:
                raise LLMUnavailableError("claude CLI printed two result messages; usage unknown")
            result = message
        elif kind == "assistant" and isinstance(error := message.get("error"), str):
            assistant_error = error
        elif kind == "rate_limit_event":
            info = as_object(message.get("rate_limit_info"))
            if info.get("status") == "rejected":
                rejected_limit = info
    return ClaudeRun(init, result, assistant_error, rejected_limit)


def check_isolation(run: ClaudeRun, *, model: str) -> None:
    """Fail unless the child ran on the subscription login with no tools, MCP servers or user plugins.

    A full model id (`claude-...`) must also be the model the child started with; an alias such as `sonnet`
    resolves inside the CLI, so there is nothing exact to compare it with.
    """
    init = run.init or {}
    result = run.result or {}
    violations: list[str] = []
    if init.get("apiKeySource") != "none":
        violations.append(f"apiKeySource is {init.get('apiKeySource')!r}, not 'none'")
    if model.startswith(_FULL_MODEL_PREFIX) and init.get("model") != model:
        violations.append(f"the model is {init.get('model')!r}, not the requested {model!r}")
    if init.get("tools") != []:
        violations.append("tools are enabled")
    if init.get("mcp_servers") != []:
        violations.append("MCP servers are connected")
    # Claude Code always loads its own builtin plugins; anything else could carry hooks or tools.
    plugins = init.get("plugins", [])
    if not isinstance(plugins, list) or any(as_object(p).get("path") != "builtin" for p in plugins):
        violations.append("non-builtin plugins are loaded")
    model_usage = as_object(result.get("modelUsage", {}))
    if any(as_object(entry).get("provider", "firstParty") != "firstParty" for entry in model_usage.values()):
        violations.append("a model was served by a provider other than firstParty")
    if result.get("permission_denials", []) != []:
        violations.append("the model attempted tool use")
    if violations:
        raise LLMUnavailableError(f"claude CLI isolation violation: {'; '.join(violations)}")


def readiness(*, timeout_s: float = _AUTH_STATUS_TIMEOUT_S) -> Readiness:
    """`claude auth status`: logged in through claude.ai on firstParty. Sends no generation."""
    try:
        executable = _executable()
        result = proc.run(
            [executable, "auth", "status"], timeout_s=timeout_s, env=child_env(), cwd=paths.llm_workdir("claude")
        )
    except (DependencyError, ProcTimeout) as exc:
        return Readiness(False, str(exc))
    try:
        status = as_object(json.loads(result.stdout))
    except ValueError:
        return Readiness(False, f"`claude auth status` printed unreadable output (exit {result.returncode})")
    # Only these three keys are read; email, org and the rest of the status never leave this function.
    method = status.get("authMethod")
    provider = status.get("apiProvider")
    if status.get("loggedIn") is not True:
        return Readiness(False, "not logged in; run `claude auth login` with your claude.ai account")
    if method != "claude.ai":
        return Readiness(False, f"logged in through {method!r}, not a claude.ai subscription")
    if provider != "firstParty":
        return Readiness(False, f"API provider is {provider!r}, not firstParty")
    return Readiness(True, "logged in through claude.ai (firstParty)")


def executable() -> str:
    """Path of the `claude` CLI, or `DependencyError`; a cheap check that sends nothing."""
    return _executable()


def _executable() -> str:
    found = shutil.which("claude")
    if found is None:
        raise DependencyError("claude CLI not found on PATH; install Claude Code, then run `claude auth login`")
    return found


def _raise_for_failure(client_id: str, run: ClaudeRun, returncode: int, usage: Usage | None) -> None:
    result = run.result or {}
    error = run.assistant_error
    status = result.get("api_error_status")
    if error == "max_output_tokens" or result.get("stop_reason") == "max_tokens":
        raise LLMOutputError(
            f"{client_id}: the reply hit max_output_tokens and was cut off", responded=True, usage=usage
        )
    if result.get("is_error") is not True and returncode == 0 and error is None:
        return
    category = error or result.get("terminal_reason") or f"exit {returncode}"
    if error in _AUTH_ERRORS or status in _AUTH_HTTP:
        raise LLMUnavailableError(
            f"{client_id}: the subscription login was rejected ({category}); run `claude auth status` and sign in "
            "with your claude.ai account. No other credential was tried."
        )
    if run.rejected_limit is not None or error == "rate_limit" or status == _RATE_LIMIT_HTTP:
        resets = (run.rejected_limit or {}).get("resetsAt")
        when = (
            f" until {datetime.fromtimestamp(resets, UTC).isoformat(timespec='minutes')}"
            if isinstance(resets, int)
            else ""
        )
        raise LLMUnavailableError(f"{client_id}: plan usage limit reached; wait{when}. No fallback was tried.")
    detail = result.get("result")
    message = f": {detail[:_TAIL]}" if isinstance(detail, str) and detail else ""
    http = f", HTTP {status}" if isinstance(status, int) else ""
    raise LLMUnavailableError(f"{client_id}: call failed ({category}{http}){message}. No retry was attempted.")


def _usage(result: Mapping[str, object]) -> Usage | None:
    # modelUsage covers every request of the run; `usage` covers the main loop only.
    model_usage = as_object(result.get("modelUsage"))
    if not model_usage:
        return None
    input_tokens = output_tokens = 0
    for entry in model_usage.values():
        counts = as_object(entry)
        fresh, cache_read, cache_write, output = (
            counts.get(key)
            for key in ("inputTokens", "cacheReadInputTokens", "cacheCreationInputTokens", "outputTokens")
        )
        if not (
            isinstance(fresh, int)
            and isinstance(cache_read, int)
            and isinstance(cache_write, int)
            and isinstance(output, int)
        ):
            return None
        input_tokens += fresh + cache_read + cache_write
        output_tokens += output
    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)


def _stderr_tail(stderr: str) -> str:
    text = stderr.strip().replace(str(Path.home()), "~")
    return f"\n{text[-_TAIL:]}" if text else ""
