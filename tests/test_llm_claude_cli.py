import json
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ytverbatim import paths, proc
from ytverbatim.errors import ConfigError, DependencyError, LLMOutputError, LLMUnavailableError, ProcTimeout
from ytverbatim.llm import claude_cli
from ytverbatim.llm.claude_cli import ClaudeCLIClient
from ytverbatim.llm.protocol import LLMRequest, Usage

FAKE_CLAUDE = "/opt/fake/bin/claude"
SYSTEM_FLAG = "--system-prompt-file="
DANGEROUS = {
    "ANTHROPIC_API_KEY": "sk-ant-fake",
    "ANTHROPIC_AUTH_TOKEN": "fake-token",
    "CLAUDE_CODE_OAUTH_TOKEN": "fake-oauth",
    "ANTHROPIC_BASE_URL": "https://proxy.example.com",
    "ANTHROPIC_MODEL": "claude-other",
    "CLAUDE_CODE_USE_BEDROCK": "1",
    "CLAUDE_CODE_USE_VERTEX": "1",
    "CLAUDE_CODE_EFFORT_LEVEL": "max",
    "CLAUDECODE": "1",
    "OPENAI_API_KEY": "sk-fake",
}


@dataclass
class FakeProc:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    raises: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list)

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        stdin: str | None = None,
    ) -> proc.ProcResult:
        system_files = [arg.removeprefix(SYSTEM_FLAG) for arg in argv if arg.startswith(SYSTEM_FLAG)]
        system_file = Path(system_files[0]) if system_files else None
        self.calls.append(
            {
                "argv": list(argv),
                "timeout_s": timeout_s,
                "env": dict(env or {}),
                "cwd": cwd,
                "stdin": stdin,
                "system_file": system_file,
                "system": None if system_file is None else system_file.read_text(encoding="utf-8"),
                "system_mode": None if system_file is None else system_file.stat().st_mode & 0o777,
            }
        )
        if self.raises is not None:
            raise self.raises
        return proc.ProcResult(self.returncode, self.stdout, self.stderr, 1.25)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: FAKE_CLAUDE if name == "claude" else None)
    return tmp_path


@pytest.fixture
def fake(home: Path, monkeypatch: pytest.MonkeyPatch) -> FakeProc:
    stub = FakeProc(stdout=stream())
    monkeypatch.setattr(proc, "run", stub.run)
    return stub


def init_message(**overrides: object) -> dict[str, object]:
    message: dict[str, object] = {
        "type": "system",
        "subtype": "init",
        "tools": [],
        "mcp_servers": [],
        "model": "claude-haiku-5-5",
        "apiKeySource": "none",
        "plugins": [{"name": "builtin-thing", "path": "builtin", "source": "builtin-thing@builtin"}],
    }
    message.update(overrides)
    return message


def result_message(**overrides: object) -> dict[str, object]:
    message: dict[str, object] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 1,
        "stop_reason": "end_turn",
        "terminal_reason": "completed",
        "api_error_status": None,
        "result": '{"ok": true}',
        "permission_denials": [],
        "modelUsage": {
            "claude-haiku-5-5": {
                "inputTokens": 2,
                "outputTokens": 9,
                "cacheReadInputTokens": 100,
                "cacheCreationInputTokens": 575,
                "provider": "firstParty",
            }
        },
    }
    message.update(overrides)
    return message


def stream(*messages: dict[str, object]) -> str:
    chosen = messages or (init_message(), assistant(), result_message())
    return "".join(json.dumps(m) + "\n" for m in chosen)


def assistant(**overrides: object) -> dict[str, object]:
    message: dict[str, object] = {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": '{"ok": true}'}]},
    }
    message.update(overrides)
    return message


REQUEST = LLMRequest(system="Be terse.", user="Say hi.", max_output_tokens=321)


def test_argv_is_the_isolated_pure_completion_command(fake: FakeProc) -> None:
    ClaudeCLIClient(model="sonnet", effort="low").complete(REQUEST)
    system_file = fake.calls[0]["system_file"]
    assert fake.calls[0]["argv"] == [
        FAKE_CLAUDE,
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        "sonnet",
        "--effort",
        "low",
        f"{SYSTEM_FLAG}{system_file}",
        "--tools",
        "",
        "--strict-mcp-config",
        "--setting-sources=",
        "--settings",
        '{"switchModelsOnFlag":false}',
        "--disable-slash-commands",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
        "--max-turns",
        "1",
    ]


def test_effort_is_omitted_when_unset() -> None:
    argv = ClaudeCLIClient(model="sonnet").argv(FAKE_CLAUDE, Path("/tmp/system.txt"))
    assert "--effort" not in argv


def test_the_system_prompt_travels_in_a_private_file_not_argv(fake: FakeProc) -> None:
    secret = "-Starts with a dash and stays out of ps."
    ClaudeCLIClient(model="sonnet").complete(LLMRequest(system=secret, user="Say hi."))
    call = fake.calls[0]
    argv = call["argv"]
    assert isinstance(argv, list)
    assert not any(secret in arg for arg in argv)
    assert call["system"] == secret
    assert call["system_mode"] == 0o600
    system_file = call["system_file"]
    assert isinstance(system_file, Path)
    assert system_file.parent != call["cwd"]
    assert not system_file.exists()


def test_the_system_file_is_removed_when_the_call_times_out(fake: FakeProc) -> None:
    fake.raises = ProcTimeout("claude", 900.0)
    with pytest.raises(LLMUnavailableError):
        ClaudeCLIClient(model="sonnet").complete(REQUEST)
    system_file = fake.calls[0]["system_file"]
    assert isinstance(system_file, Path)
    assert not system_file.exists()


@pytest.mark.parametrize("turns", [0, 2, None])
def test_a_run_that_did_not_take_exactly_one_turn_is_refused(fake: FakeProc, turns: int | None) -> None:
    fake.stdout = stream(init_message(), result_message(num_turns=turns))
    with pytest.raises(LLMUnavailableError, match="not 1"):
        ClaudeCLIClient(model="sonnet").complete(REQUEST)


def test_a_full_model_id_must_be_the_model_the_child_started_with(fake: FakeProc) -> None:
    ClaudeCLIClient(model="claude-haiku-5-5").complete(REQUEST)
    with pytest.raises(LLMUnavailableError, match="not the requested 'claude-sonnet-5-5'"):
        ClaudeCLIClient(model="claude-sonnet-5-5").complete(REQUEST)


def test_an_alias_is_not_compared_with_the_resolved_model(fake: FakeProc) -> None:
    assert ClaudeCLIClient(model="haiku").complete(REQUEST).text == '{"ok": true}'


def test_env_is_allow_listed_and_dangerous_variables_never_reach_the_child(
    fake: FakeProc, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in DANGEROUS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LC_CTYPE", "UTF-8")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:3128")
    monkeypatch.setenv("SOME_UNRELATED_SECRET", "x")

    ClaudeCLIClient(model="sonnet").complete(REQUEST)
    env = fake.calls[0]["env"]
    assert isinstance(env, dict)
    assert not set(DANGEROUS) & set(env)
    assert "SOME_UNRELATED_SECRET" not in env
    assert env["LC_CTYPE"] == "UTF-8"
    assert env["HTTPS_PROXY"] == "http://proxy.example.com:3128"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "321"
    assert env["CLAUDE_CODE_MAX_RETRIES"] == "0"
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    allowed = set(claude_cli.PASSTHROUGH_ENV) | set(claude_cli.FIXED_ENV) | {"CLAUDE_CODE_MAX_OUTPUT_TOKENS"}
    assert all(key in allowed or key.startswith("LC_") for key in env)


def test_child_env_reads_only_the_allow_list_from_a_given_source() -> None:
    env = claude_cli.child_env({"HOME": "/home/example", "PATH": "/usr/bin", **DANGEROUS})
    assert env == {"HOME": "/home/example", "PATH": "/usr/bin", **claude_cli.FIXED_ENV}


def test_cwd_is_an_empty_private_dir_under_the_llm_cache(fake: FakeProc) -> None:
    ClaudeCLIClient(model="sonnet").complete(REQUEST)
    cwd = fake.calls[0]["cwd"]
    assert isinstance(cwd, Path)
    assert cwd.parent == paths.llm_dir("claude")
    assert stat.S_IMODE(cwd.stat().st_mode) == 0o700
    assert list(cwd.iterdir()) == []


def test_stdin_is_one_client_composed_user_line(fake: FakeProc) -> None:
    ClaudeCLIClient(model="sonnet").complete(LLMRequest(system="", user="/not-a-command @not/a/path 你好"))
    stdin = fake.calls[0]["stdin"]
    assert isinstance(stdin, str)
    assert stdin.endswith("\n")
    assert stdin.count("\n") == 1
    assert json.loads(stdin) == {
        "type": "user",
        "message": {"role": "user", "content": "/not-a-command @not/a/path 你好"},
        "parent_tool_use_id": None,
        "client_composed": True,
    }


def test_success_returns_text_usage_and_elapsed(fake: FakeProc) -> None:
    response = ClaudeCLIClient(model="haiku").complete(REQUEST)
    assert response.text == '{"ok": true}'
    assert response.usage == Usage(input_tokens=2 + 100 + 575, output_tokens=9)
    assert response.elapsed_s == 1.25


def test_missing_model_usage_means_unknown_usage(fake: FakeProc) -> None:
    fake.stdout = stream(init_message(), result_message(modelUsage={}))
    assert ClaudeCLIClient(model="haiku").complete(REQUEST).usage is None


@pytest.mark.parametrize(
    ("init", "result", "reason"),
    [
        ({"apiKeySource": "ANTHROPIC_API_KEY"}, {}, "apiKeySource"),
        ({"tools": ["Bash"]}, {}, "tools"),
        ({"mcp_servers": [{"name": "x", "status": "connected"}]}, {}, "MCP"),
        ({"plugins": [{"name": "user-plugin", "path": "/opt/plugins/x"}]}, {}, "plugins"),
        ({}, {"modelUsage": {"m": {"provider": "bedrock", "inputTokens": 1}}}, "firstParty"),
        ({}, {"permission_denials": [{"tool_name": "Bash"}]}, "tool use"),
    ],
)
def test_isolation_violation_raises(
    fake: FakeProc, init: dict[str, object], result: dict[str, object], reason: str
) -> None:
    fake.stdout = stream(init_message(**init), result_message(**result))
    with pytest.raises(LLMUnavailableError, match="isolation violation") as caught:
        ClaudeCLIClient(model="haiku").complete(REQUEST)
    assert reason in str(caught.value)


def test_isolation_is_checked_before_a_reported_error(fake: FakeProc) -> None:
    fake.stdout = stream(init_message(apiKeySource="ANTHROPIC_API_KEY"), result_message(is_error=True))
    fake.returncode = 1
    with pytest.raises(LLMUnavailableError, match="isolation violation"):
        ClaudeCLIClient(model="haiku").complete(REQUEST)


def test_api_error_raises_with_category_and_status(fake: FakeProc) -> None:
    fake.stdout = stream(
        init_message(),
        assistant(error="model_not_found", api_error_status=404),
        result_message(is_error=True, terminal_reason="api_error", api_error_status=404, result="bad model"),
    )
    fake.returncode = 1
    with pytest.raises(LLMUnavailableError, match=r"model_not_found, HTTP 404\): bad model"):
        ClaudeCLIClient(model="nope").complete(REQUEST)


def test_auth_failure_names_the_fix(fake: FakeProc) -> None:
    fake.stdout = stream(init_message(), assistant(error="authentication_failed"), result_message(is_error=True))
    fake.returncode = 1
    with pytest.raises(LLMUnavailableError, match="claude auth status"):
        ClaudeCLIClient(model="haiku").complete(REQUEST)


def test_rate_limit_reports_the_reset_time(fake: FakeProc) -> None:
    fake.stdout = stream(
        init_message(),
        {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": 1_800_000_000}},
        result_message(is_error=True, modelUsage={}),
    )
    fake.returncode = 1
    with pytest.raises(LLMUnavailableError, match=r"usage limit reached; wait until 2027-01-15T08:00\+00:00"):
        ClaudeCLIClient(model="haiku").complete(REQUEST)


def test_truncated_reply_is_an_output_error(fake: FakeProc) -> None:
    fake.stdout = stream(init_message(), result_message(stop_reason="max_tokens"))
    with pytest.raises(LLMOutputError, match="max_output_tokens") as caught:
        ClaudeCLIClient(model="haiku").complete(REQUEST)
    assert (caught.value.responded, caught.value.usage) == (True, Usage(input_tokens=677, output_tokens=9))


def test_an_isolation_violation_still_reports_what_the_reply_cost(fake: FakeProc) -> None:
    fake.stdout = stream(init_message(tools=["Bash"]), result_message())
    with pytest.raises(LLMUnavailableError, match="tools are enabled") as caught:
        ClaudeCLIClient(model="haiku").complete(REQUEST)
    assert caught.value.responded
    assert caught.value.usage == Usage(input_tokens=677, output_tokens=9)


def test_a_rejected_login_bills_nothing(fake: FakeProc) -> None:
    fake.stdout = stream(init_message(), assistant(error="authentication_failed"), result_message(is_error=True))
    fake.returncode = 1
    with pytest.raises(LLMUnavailableError) as caught:
        ClaudeCLIClient(model="haiku").complete(REQUEST)
    assert not caught.value.responded


def test_empty_reply_is_an_output_error(fake: FakeProc) -> None:
    fake.stdout = stream(init_message(), result_message(result="  "))
    with pytest.raises(LLMOutputError, match="empty"):
        ClaudeCLIClient(model="haiku").complete(REQUEST)


@pytest.mark.parametrize(
    "stdout",
    [
        "not json\n",
        "[1, 2]\n",
        stream(init_message()),
        stream(result_message()),
        stream(init_message(), init_message(), result_message()),
        stream(init_message(), result_message(), result_message()),
    ],
)
def test_malformed_stream_is_unusable(fake: FakeProc, stdout: str) -> None:
    fake.stdout = stdout
    with pytest.raises(LLMUnavailableError, match="usage unknown"):
        ClaudeCLIClient(model="haiku").complete(REQUEST)


def test_no_stream_includes_redacted_stderr_tail(fake: FakeProc, home: Path) -> None:
    fake.stdout = ""
    fake.returncode = 1
    fake.stderr = f"boom in {home}/somewhere"
    with pytest.raises(LLMUnavailableError, match="exit 1") as caught:
        ClaudeCLIClient(model="haiku").complete(REQUEST)
    assert str(home) not in str(caught.value)
    assert "boom in ~/somewhere" in str(caught.value)


def test_timeout_is_unavailable(fake: FakeProc) -> None:
    fake.raises = ProcTimeout("claude", 900.0)
    with pytest.raises(LLMUnavailableError, match="did not finish within 900s"):
        ClaudeCLIClient(model="haiku").complete(REQUEST)
    assert fake.calls[0]["timeout_s"] == 900.0


def test_missing_cli_is_a_dependency_error(fake: FakeProc, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: None)
    with pytest.raises(DependencyError, match="claude CLI not found"):
        ClaudeCLIClient(model="haiku").complete(REQUEST)
    assert fake.calls == []


def test_constructor_rejects_bad_options() -> None:
    with pytest.raises(ConfigError, match="model"):
        ClaudeCLIClient(model="")
    with pytest.raises(ConfigError, match="effort"):
        ClaudeCLIClient(model="sonnet", effort="extreme")
    assert ClaudeCLIClient(model="sonnet").id == "claude-cli:sonnet"


def _status(**fields: object) -> str:
    status: dict[str, object] = {
        "loggedIn": True,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "email": "user@example.com",
        "orgId": "org-fake",
    }
    status.update(fields)
    return json.dumps(status)


def test_readiness_ready_without_echoing_the_email(fake: FakeProc, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    fake.stdout = _status()
    status = claude_cli.readiness()
    assert status.ready
    assert "example.com" not in status.detail
    call = fake.calls[0]
    assert call["argv"] == [FAKE_CLAUDE, "auth", "status"]
    env = call["env"]
    assert isinstance(env, dict)
    assert "ANTHROPIC_API_KEY" not in env


@pytest.mark.parametrize(
    ("fields", "returncode", "reason"),
    [
        ({"loggedIn": False, "authMethod": "none"}, 1, "not logged in"),
        ({"authMethod": "api_key"}, 0, "'api_key'"),
        ({"apiProvider": "bedrock"}, 0, "'bedrock'"),
    ],
)
def test_readiness_not_ready(fake: FakeProc, fields: dict[str, object], returncode: int, reason: str) -> None:
    fake.stdout = _status(**fields)
    fake.returncode = returncode
    status = claude_cli.readiness()
    assert not status.ready
    assert reason in status.detail
    assert "example.com" not in status.detail


def test_readiness_handles_missing_cli_timeout_and_garbage(fake: FakeProc, monkeypatch: pytest.MonkeyPatch) -> None:
    fake.stdout = "Logged in as someone"
    assert "unreadable" in claude_cli.readiness().detail
    fake.raises = ProcTimeout("claude", 10.0)
    assert not claude_cli.readiness().ready
    monkeypatch.setattr(claude_cli.shutil, "which", lambda name: None)
    assert "not found" in claude_cli.readiness().detail
