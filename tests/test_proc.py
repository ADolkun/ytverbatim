import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from ytverbatim import proc, sysmem
from ytverbatim.errors import DependencyError, ProcMemoryExceeded, ProcTimeout

MIB = 1 << 20


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
            if mode == "slow-echo":
                time.sleep(0.6)
                data = sys.stdin.read()
                sys.stdout.write(f"{len(data)}:{data[:3]}")
            if mode == "chatty":
                for i in range(6):
                    sys.stdout.write(str(i) * 100_000)
                    sys.stderr.write("é" * 50_000)
                    sys.stdout.flush()
                    sys.stderr.flush()
                    time.sleep(0.1)
            if mode == "hog":
                import subprocess
                helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
                open(sys.argv[2], "w").write(f"{os.getpid()} {helper.pid}")
                held = []
                for _ in range(15):
                    held.append(b"x" * (100 << 20))
                    time.sleep(0.05)
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


def _plenty() -> int:
    return 50


def _group_gone(pgid: int) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_memory_watch_kills_a_hog_and_its_whole_group(script: Path, tmp_path: Path) -> None:
    pid_file = tmp_path / "pids"
    watch = proc.MemoryWatch(600 * MIB, poll_s=0.1, available_percent=_plenty)
    with pytest.raises(ProcMemoryExceeded) as info:
        proc.run([sys.executable, str(script), "hog", str(pid_file)], timeout_s=60, memory_watch=watch)
    error = info.value
    assert error.trigger == "footprint"
    assert error.footprint_bytes is not None
    assert 600 * MIB < error.footprint_bytes < 1600 * MIB
    assert (error.cap_bytes, error.available_percent, error.floor_percent) == (600 * MIB, 50, 10)
    assert "physical footprint reached" in str(error)
    assert "over the 0.59 GiB cap" in str(error)
    hog, helper = map(int, pid_file.read_text(encoding="utf-8").split())
    assert _group_gone(hog)
    with pytest.raises(ProcessLookupError):
        os.kill(helper, 0)


def test_memory_watch_stops_the_child_when_the_machine_runs_short(script: Path, tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    watch = proc.MemoryWatch(1 << 40, poll_s=0.1, available_percent=lambda: 7)
    started = time.monotonic()
    with pytest.raises(ProcMemoryExceeded) as info:
        proc.run([sys.executable, str(script), "sleep", str(pid_file)], timeout_s=60, memory_watch=watch)
    assert time.monotonic() - started < 30
    assert info.value.trigger == "available"
    assert info.value.available_percent == 7
    assert "fell to 7%, under the 10% floor" in str(info.value)
    assert _group_gone(int(pid_file.read_text(encoding="utf-8")))


def test_memory_watch_collects_output_across_polls(script: Path) -> None:
    watch = proc.MemoryWatch(1 << 40, poll_s=0.1, available_percent=_plenty)
    result = proc.run([sys.executable, str(script), "chatty"], timeout_s=30, memory_watch=watch)
    assert result.returncode == 0
    assert result.stdout == "".join(str(i) * 100_000 for i in range(6))
    assert result.stderr == "é" * 300_000


def test_memory_watch_refuses_stdin(script: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_spawn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("must refuse before starting a process")

    monkeypatch.setattr(subprocess, "Popen", no_spawn)
    watch = proc.MemoryWatch(1 << 40, poll_s=0.1, available_percent=_plenty)
    with pytest.raises(ValueError, match="stdin"):
        proc.run([sys.executable, str(script), "slow-echo"], timeout_s=30, stdin="abc", memory_watch=watch)


def test_memory_watch_still_times_out(script: Path, tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    watch = proc.MemoryWatch(1 << 40, poll_s=0.4, available_percent=_plenty)
    started = time.monotonic()
    with pytest.raises(ProcTimeout):
        proc.run([sys.executable, str(script), "sleep", str(pid_file)], timeout_s=1.0, memory_watch=watch)
    assert time.monotonic() - started < 10
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text(encoding="utf-8")), 0)


def test_without_a_watch_nothing_polls(script: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timeouts: list[float | None] = []
    original = subprocess.Popen.communicate

    def spy(self: subprocess.Popen[str], input: Any = None, timeout: float | None = None) -> Any:
        timeouts.append(timeout)
        return original(self, input, timeout)

    def unexpected(*_: object) -> None:
        raise AssertionError("memory read without a watch")

    monkeypatch.setattr(subprocess.Popen, "communicate", spy)
    monkeypatch.setattr(sysmem, "footprint", unexpected)
    monkeypatch.setattr(sysmem, "available_percent", unexpected)
    result = proc.run([sys.executable, str(script), "slow-echo"], timeout_s=30, stdin="abc")
    assert result.stdout == "3:abc"
    assert timeouts == [30]


def test_a_child_that_exits_as_a_trigger_fires_is_collected_not_reported_killed(script: Path) -> None:
    waits: list[float | None] = []
    original = subprocess.Popen.communicate

    def slow_to_notice(self: subprocess.Popen[str], input: Any = None, timeout: float | None = None) -> Any:
        waits.append(timeout)
        # The first wait times out even though the child exits during it, as a wait that ends just too early would.
        if len(waits) == 1:
            self.wait()
            raise subprocess.TimeoutExpired(self.args, timeout or 0.0)
        return original(self, input, timeout)

    def short_of_memory() -> int:
        return 3

    watch = proc.MemoryWatch(1 << 40, poll_s=0.1, available_percent=short_of_memory)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(subprocess.Popen, "communicate", slow_to_notice)
        result = proc.run([sys.executable, str(script), "echo", "0"], timeout_s=30, memory_watch=watch)
    assert (result.returncode, result.stdout) == (0, "out:")
    assert len(waits) == 2
