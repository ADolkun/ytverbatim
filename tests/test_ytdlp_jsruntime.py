from collections.abc import Sequence

import pytest

from ytverbatim.errors import DependencyError
from ytverbatim.proc import ProcResult
from ytverbatim.ytdlp import jsruntime


@pytest.fixture(autouse=True)
def no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(jsruntime.ENV_OVERRIDE, raising=False)


def _path(monkeypatch: pytest.MonkeyPatch, *found: str) -> None:
    monkeypatch.setattr(jsruntime.shutil, "which", lambda name: f"/opt/bin/{name}" if name in found else None)


def _node(monkeypatch: pytest.MonkeyPatch, stdout: str, code: int = 0) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(argv: Sequence[str], *, timeout_s: float, **_: object) -> ProcResult:
        calls.append(list(argv))
        return ProcResult(code, stdout, "", 0.01)

    monkeypatch.setattr(jsruntime.proc, "run", run)
    return calls


def test_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(jsruntime.ENV_OVERRIDE, "node:/opt/node/bin/node")
    _path(monkeypatch, "deno")
    found = jsruntime.detect()
    assert found == jsruntime.JsRuntime("node", ("--no-js-runtimes", "--js-runtimes", "node:/opt/node/bin/node"), None)
    assert found.args.index("--no-js-runtimes") < found.args.index("--js-runtimes")


def test_deno_needs_no_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _path(monkeypatch, "deno", "node")
    calls = _node(monkeypatch, "v24.0.0\n")
    assert jsruntime.detect() == jsruntime.JsRuntime("deno", (), None)
    assert calls == []


def test_recent_node(monkeypatch: pytest.MonkeyPatch) -> None:
    _path(monkeypatch, "node")
    calls = _node(monkeypatch, "v22.3.0\n")
    assert jsruntime.detect() == jsruntime.JsRuntime("node", ("--js-runtimes", "node"), None)
    assert calls == [["/opt/bin/node", "--version"]]


def test_old_node_yields_no_args_and_a_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    _path(monkeypatch, "node")
    _node(monkeypatch, "v20.11.1\n")
    found = jsruntime.detect()
    assert (found.name, found.args) == (None, ())
    assert found.finding is not None
    assert "node 20" in found.finding


def test_no_runtime_yields_a_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    _path(monkeypatch)
    found = jsruntime.detect()
    assert (found.name, found.args) == (None, ())
    assert found.finding is not None
    assert "deno" in found.finding


@pytest.mark.parametrize(("stdout", "code"), [("garbage", 0), ("v24.0.0", 1)])
def test_unparseable_or_failing_node(monkeypatch: pytest.MonkeyPatch, stdout: str, code: int) -> None:
    _path(monkeypatch, "node")
    _node(monkeypatch, stdout, code)
    assert jsruntime.node_major_version() is None


def test_node_that_cannot_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _path(monkeypatch, "node")

    def run(argv: Sequence[str], *, timeout_s: float, **_: object) -> ProcResult:
        raise DependencyError("required executable is not runnable: node")

    monkeypatch.setattr(jsruntime.proc, "run", run)
    assert jsruntime.node_major_version() is None
