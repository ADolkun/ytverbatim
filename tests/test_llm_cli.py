from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from click.testing import CliRunner

from ytverbatim.cli import llm as llm_cli
from ytverbatim.cli import main
from ytverbatim.config import LLMOptions
from ytverbatim.errors import ConfigError, DependencyError
from ytverbatim.llm import (
    ClaudeCLIClient,
    OpenAICompatClient,
    apply_spec,
    check_ready,
    claude_cli,
    client_from_options,
    openai_compat,
)
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Readiness, Usage


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "ytverbatim" / "config.toml"
    path.parent.mkdir()
    return path


@dataclass
class ProbeClient:
    requests: list[LLMRequest] = field(default_factory=list)

    @property
    def id(self) -> str:
        return "fake:probe"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text='{"ok": true}', usage=Usage(30, 4), elapsed_s=0.5)


def test_check_without_provider_fails(config: Path) -> None:
    result = CliRunner().invoke(main, ["llm", "check"])
    assert result.exit_code == 1
    assert "no LLM provider configured" in result.output


def test_check_claude_ready_sends_no_generation(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_cli, "readiness", lambda: Readiness(True, "logged in through claude.ai (firstParty)"))
    monkeypatch.setattr(llm_cli, "client_from_options", _forbidden)
    result = CliRunner().invoke(main, ["llm", "check", "--provider", "claude-cli"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "claude-cli: ready - logged in through claude.ai (firstParty)"


def test_check_not_ready_exits_nonzero(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_cli, "readiness", lambda: Readiness(False, "not logged in"))
    result = CliRunner().invoke(main, ["llm", "check", "--provider", "claude-cli"])
    assert result.exit_code == 1
    assert "claude-cli: not ready - not logged in" in result.output


def test_check_openai_uses_configured_base_url(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config.write_text('[llm]\nprovider = "openai-compat"\nbase_url = "http://127.0.0.1:9/v1"\n', encoding="utf-8")
    seen: list[tuple[str | None, str | None]] = []

    def fake(base_url: str | None, api_key_env: str | None = None) -> Readiness:
        seen.append((base_url, api_key_env))
        return Readiness(True, "answered")

    monkeypatch.setattr(openai_compat, "readiness", fake)
    result = CliRunner().invoke(main, ["llm", "check"])
    assert result.exit_code == 0, result.output
    assert seen == [("http://127.0.0.1:9/v1", None)]


def test_probe_sends_one_json_call_and_reports_usage(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    probe = ProbeClient()
    monkeypatch.setattr(claude_cli, "readiness", lambda: Readiness(True, "ok"))
    monkeypatch.setattr(llm_cli, "client_from_options", lambda options: probe)
    result = CliRunner().invoke(main, ["llm", "check", "--provider", "claude-cli", "--model", "haiku", "--probe"])
    assert result.exit_code == 0, result.output
    assert len(probe.requests) == 1
    assert 'probe fake:probe: {"ok":true} (1 call(s), input_tokens=30, output_tokens=4)' in result.output


def test_probe_errors_are_reported(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_cli, "readiness", lambda: Readiness(True, "ok"))
    result = CliRunner().invoke(main, ["llm", "check", "--provider", "claude-cli", "--probe"])
    assert result.exit_code == 1
    assert "needs a model" in result.output


def _forbidden(options: LLMOptions) -> object:
    raise AssertionError("readiness must not build a client")


def test_client_from_options() -> None:
    claude = client_from_options(LLMOptions(provider="claude-cli", model="sonnet", effort="low", timeout_s=60.0))
    assert isinstance(claude, ClaudeCLIClient)
    assert claude.id == "claude-cli:sonnet@low", "effort changes replies, so it is part of the identity"
    assert ClaudeCLIClient(model="sonnet").id == "claude-cli:sonnet"
    compat = client_from_options(LLMOptions(provider="openai-compat", model="m", base_url="http://127.0.0.1:9/v1"))
    assert isinstance(compat, OpenAICompatClient)
    assert compat.id == "openai-compat:m@127.0.0.1:9"


def test_llm_spec_keeps_its_own_providers_settings_and_drops_the_others() -> None:
    compat = LLMOptions(provider="openai-compat", model="m", base_url="http://127.0.0.1:9/v1", timeout_s=60.0)
    switched = apply_spec(compat, "claude-cli:claude-sonnet-5-5")
    assert switched == LLMOptions(provider="claude-cli", model="claude-sonnet-5-5", timeout_s=60.0)
    assert apply_spec(compat, "openai-compat:other") == replace(compat, model="other")
    claude = LLMOptions(provider="claude-cli", model="sonnet", effort="high")
    assert apply_spec(claude, "openai-compat:m").effort is None
    assert apply_spec(claude, "none").provider == "none"
    with pytest.raises(ConfigError, match="PROVIDER:MODEL"):
        apply_spec(claude, "sonnet")


def test_check_ready_is_local_and_fails_on_a_missing_cli_or_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _name: None)
    with pytest.raises(DependencyError, match="claude CLI not found"):
        check_ready(LLMOptions(provider="claude-cli", model="sonnet"))
    with pytest.raises(ConfigError, match="base_url"):
        check_ready(LLMOptions(provider="openai-compat", model="m"))
    monkeypatch.delenv("SYNTHETIC_KEY", raising=False)
    with pytest.raises(ConfigError, match="SYNTHETIC_KEY"):
        check_ready(
            LLMOptions(
                provider="openai-compat", model="m", base_url="https://x.example/v1", api_key_env="SYNTHETIC_KEY"
            )
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (LLMOptions(), "no LLM provider"),
        (LLMOptions(provider="claude-cli"), "needs a model"),
        (LLMOptions(provider="claude-cli", model="s", base_url="http://x"), "base_url"),
        (LLMOptions(provider="claude-cli", model="s", api_key_env="K"), "api_key_env"),
        (LLMOptions(provider="openai-compat", model="m", base_url="http://x", effort="low"), "effort"),
        (LLMOptions(provider="openai-compat", model="m"), "base_url"),
    ],
)
def test_client_from_options_rejects(options: LLMOptions, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        client_from_options(options)
