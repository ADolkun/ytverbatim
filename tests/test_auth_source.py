import json
import stat
from pathlib import Path

import pytest

from ytverbatim import paths
from ytverbatim.auth import source
from ytverbatim.auth.source import BrowserSource, CookieFileSource
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "YTVERBATIM_HOME", source.ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("chrome", BrowserSource("chrome")),
        ("chrome:Default", BrowserSource("chrome", "Default")),
        ("chrome:Guest Dir", BrowserSource("chrome", "Guest Dir")),
        ("brave:Default", BrowserSource("brave", "Default")),
        ("edge", BrowserSource("edge")),
        ("none", None),
    ],
)
def test_parse_spec(spec: str, expected: object) -> None:
    assert source.parse_spec(spec) == expected


def test_parse_cookie_file_expands_user(home: Path) -> None:
    assert source.parse_spec("cookie-file:~/c.txt") == CookieFileSource(home / "c.txt")


@pytest.mark.parametrize(
    ("spec", "state"),
    [
        ("firefox", AuthState.BROWSER_NOT_FOUND),
        ("Chrome", AuthState.BROWSER_NOT_FOUND),
        ("", AuthState.BROWSER_NOT_FOUND),
        ("cookie-file:", AuthState.COOKIE_FILE_INVALID),
    ],
)
def test_bad_specs_raise(spec: str, state: AuthState) -> None:
    with pytest.raises(AuthError) as caught:
        source.parse_spec(spec)
    assert caught.value.state is state


@pytest.mark.parametrize("spec", ["none", "chrome", "chrome:Work", "cookie-file:/tmp/c.txt"])
def test_format_round_trips(spec: str) -> None:
    assert source.format_spec(source.parse_spec(spec)) == spec


def test_unset_means_anonymous() -> None:
    assert source.resolve_source(env={}) is None


def test_precedence_cli_then_env_then_saved() -> None:
    source.save_source(BrowserSource("chrome", "Saved"))
    env = {source.ENV_VAR: "chrome:Env"}
    assert source.resolve_source("chrome:Cli", env=env) == BrowserSource("chrome", "Cli")
    assert source.resolve_source(None, env=env) == BrowserSource("chrome", "Env")
    assert source.resolve_source(None, env={source.ENV_VAR: "  "}) == BrowserSource("chrome", "Saved")
    assert source.resolve_source("none", env=env) is None


def test_env_is_read_from_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(source.ENV_VAR, "brave")
    assert source.resolve_source() == BrowserSource("brave")


def test_misconfigured_source_never_falls_back_to_anonymous() -> None:
    with pytest.raises(AuthError):
        source.resolve_source(None, env={source.ENV_VAR: "firefox"})
    paths.auth_source_file().write_text("{not json", encoding="utf-8")
    with pytest.raises(AuthError) as caught:
        source.resolve_source(None, env={})
    assert caught.value.state is AuthState.NO_SOURCE
    paths.auth_source_file().write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(AuthError):
        source.resolve_source(None, env={})


def test_save_source_is_private() -> None:
    path = source.save_source(None)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1, "spec": "none"}
    assert source.resolve_source(None, env={}) is None
