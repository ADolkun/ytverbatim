import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from ytverbatim import proc
from ytverbatim.auth import snapshot
from ytverbatim.auth.source import BrowserSource
from ytverbatim.ytdlp.runner import SubprocessRunner

PLANTED = "import sys; sys.stderr.write('PLANTED'); sys.exit(97)\n"


def _plant(cwd: Path) -> None:
    (cwd / "json.py").write_text(PLANTED, encoding="utf-8")
    package = cwd / "yt_dlp"
    package.mkdir()
    (package / "__init__.py").write_text(PLANTED, encoding="utf-8")
    (package / "__main__.py").write_text(PLANTED, encoding="utf-8")


def _snapshot_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    captured: list[str] = []

    def fake_run(argv: Sequence[str], **_: object) -> proc.ProcResult:
        captured.extend(argv)
        return proc.ProcResult(0, '{"ok": true}', "", 0.0)

    monkeypatch.setattr(proc, "run", fake_run)
    snapshot.snapshot_browser(BrowserSource("chrome"), tmp_path / "snap.txt")
    return captured


def _ytdlp_argv(_tmp_path: Path, _monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return SubprocessRunner(js_args=()).argv(["--version"], cookies=None)


@pytest.mark.parametrize("build", [_snapshot_argv, _ytdlp_argv], ids=["snapshot-child", "ytdlp"])
def test_child_ignores_modules_planted_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, build: Callable[[Path, pytest.MonkeyPatch], list[str]]
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    _plant(cwd)
    argv = build(tmp_path, monkeypatch)
    env = {**os.environ, "HOME": str(tmp_path)}
    env.pop("PYTHONSAFEPATH", None)

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=120, check=False)

    unprotected = run([arg for arg in argv if arg != "-P"])
    assert unprotected.returncode == 97, "the planted modules must be importable without -P, or this test is void"

    protected = run(argv)
    assert "-P" in argv
    assert protected.returncode != 97
    assert "PLANTED" not in protected.stdout + protected.stderr
