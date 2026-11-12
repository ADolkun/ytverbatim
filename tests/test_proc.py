import os
import sys
import textwrap
import time
from pathlib import Path

import pytest

from ytverbatim import proc
from ytverbatim.errors import DependencyError, ProcTimeout


@pytest.fixture
def script(tmp_path: Path) -> Path:
    path = tmp_path / "fake.py"
    path.write_text(
        textwrap.dedent(
            """
            import os, sys, time
            mode = sys.argv[1]
            if mode == "echo":
                sys.stdout.write("out:" + sys.stdin.read())
                sys.stderr.write("err")
                sys.exit(int(sys.argv[2]))
            if mode == "env":
                print(os.environ.get("YTV_TEST_VAR", "<unset>"), os.getcwd())
            if mode == "sleep":
                open(sys.argv[2], "w").write(str(os.getpid()))
                time.sleep(60)
            """
        ),
        encoding="utf-8",
    )
    return path


def test_success_captures_streams(script: Path) -> None:
    result = proc.run([sys.executable, str(script), "echo", "0"], timeout_s=30, stdin="héllo")
    assert (result.returncode, result.stdout, result.stderr) == (0, "out:héllo", "err")
    assert result.elapsed_s >= 0


def test_nonzero_exit_does_not_raise(script: Path) -> None:
    result = proc.run([sys.executable, str(script), "echo", "7"], timeout_s=30)
    assert result.returncode == 7
    assert result.stdout == "out:"


def test_stdin_is_closed_when_not_given(script: Path) -> None:
    result = proc.run([sys.executable, str(script), "echo", "0"], timeout_s=30)
    assert result.stdout == "out:"


def test_env_and_cwd_are_used(script: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    result = proc.run(
        [sys.executable, str(script), "env"],
        timeout_s=30,
        env={"YTV_TEST_VAR": "yes", "PATH": os.environ.get("PATH", "")},
        cwd=work,
    )
    value, cwd = result.stdout.split()
    assert value == "yes"
    assert Path(cwd).resolve() == work.resolve()


def test_default_env_is_inherited(script: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YTV_TEST_VAR", "inherited")
    result = proc.run([sys.executable, str(script), "env"], timeout_s=30)
    assert result.stdout.split()[0] == "inherited"


def test_timeout_kills_the_child(script: Path, tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    started = time.monotonic()
    with pytest.raises(ProcTimeout) as info:
        proc.run([sys.executable, str(script), "sleep", str(pid_file)], timeout_s=1.5)
    assert time.monotonic() - started < 30
    assert info.value.timeout_s == 1.5
    assert info.value.command == Path(sys.executable).name
    pid = int(pid_file.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_missing_binary_is_a_dependency_error() -> None:
    with pytest.raises(DependencyError, match="ytverbatim-no-such-binary"):
        proc.run(["ytverbatim-no-such-binary"], timeout_s=5)


def test_non_executable_file_is_a_dependency_error(tmp_path: Path) -> None:
    target = tmp_path / "plain"
    target.write_text("x", encoding="utf-8")
    target.chmod(0o644)
    with pytest.raises(DependencyError, match="not runnable"):
        proc.run([str(target)], timeout_s=5)


def test_missing_cwd_is_not_reported_as_a_missing_binary(script: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        proc.run([sys.executable, str(script), "env"], timeout_s=5, cwd=tmp_path / "absent")
