import fcntl
import importlib
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import ModuleType

import pytest
from test_proc import _gone

from evals import mlx_server
from ytverbatim.asr.worker import MLX_LIMIT_FRACTION
from ytverbatim.errors import DependencyError

CAP = 12 << 30


class FakeMlx(ModuleType):
    def __init__(self, calls: list[str]) -> None:
        super().__init__("mlx.core")
        self.calls = calls

    def set_cache_limit(self, limit: int) -> int:
        self.calls.append(f"cache_limit {limit}")
        return 0

    def set_memory_limit(self, limit: int) -> int:
        self.calls.append(f"memory_limit {limit}")
        return 0

    def clear_cache(self) -> None:
        self.calls.append("clear_cache")


class FakeServer(ModuleType):
    def __init__(self, calls: list[str]) -> None:
        super().__init__("mlx_lm.server")
        self.calls = calls

    def main(self) -> None:
        self.calls.append("server " + " ".join(sys.argv))


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Fake `mlx.core` and `mlx_lm.server`, so no test touches the real allocator or starts a server."""
    log: list[str] = []
    monkeypatch.setitem(sys.modules, "mlx.core", FakeMlx(log))
    monkeypatch.setitem(sys.modules, "mlx_lm.server", FakeServer(log))
    monkeypatch.setattr(sys, "argv", ["mlx_server"])
    monkeypatch.setattr(mlx_server, "_exit_with_parent", lambda: log.append("exit_with_parent"))
    return log


def test_the_allocator_is_bounded_before_the_server_runs_with_the_rest_of_argv(calls: list[str]) -> None:
    mlx_server.main(["--memory-cap-bytes", str(CAP), "--", "--model", "/models/m", "--port", "8123"])
    assert calls == [
        "exit_with_parent",
        f"cache_limit {mlx_server.CACHE_LIMIT_BYTES}",
        f"memory_limit {int(CAP * MLX_LIMIT_FRACTION)}",
        "server mlx_lm.server --model /models/m --port 8123",
    ]
    assert mlx_server.CACHE_LIMIT_BYTES == 1 << 30


def test_without_mlx_lm_it_names_the_install_command(calls: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx_lm.server", None)
    with pytest.raises(DependencyError, match="uv sync --group local-llm"):
        mlx_server.main(["--memory-cap-bytes", str(CAP), "--", "--model", "/models/m"])
    assert all(not call.startswith("server") for call in calls)


def test_without_mlx_it_names_the_install_command(calls: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    with pytest.raises(DependencyError, match="uv sync --group local-llm"):
        mlx_server.main(["--memory-cap-bytes", str(CAP), "--"])
    assert calls == ["exit_with_parent"]


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'transformers'", name="transformers"),
        ImportError("cannot import name 'Thing' from 'transformers'"),
    ],
)
def test_a_broken_dependency_of_mlx_lm_keeps_its_own_message(
    calls: list[str], monkeypatch: pytest.MonkeyPatch, error: ImportError
) -> None:
    real = importlib.import_module

    def importer(name: str) -> object:
        if name == "mlx_lm.server":
            raise error
        return real(name)

    monkeypatch.setattr(importlib, "import_module", importer)
    with pytest.raises(ImportError) as info:
        mlx_server.main(["--memory-cap-bytes", str(CAP), "--"])
    assert info.value is error


@pytest.mark.parametrize(
    "argv",
    [
        ["--memory-cap-bytes", str(CAP), "--model", "/models/m"],
        ["--", "--model", "/models/m"],
        ["--memory-cap-bytes", "0", "--"],
        ["--memory-cap-bytes", "12GiB", "--"],
    ],
)
def test_bad_arguments_exit_with_usage_before_anything_loads(calls: list[str], argv: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        mlx_server.main(argv)
    assert info.value.code == 2
    assert calls == ["exit_with_parent"]


REPO = Path(__file__).resolve().parents[1]

# Runs the real `mlx_server.main` with fake MLX modules; the fake server writes its pid and serves forever.
SERVER = """
import os, sys, time, types
root, pid_file = sys.argv[1:]
sys.path.insert(0, root)
core = types.ModuleType("mlx.core")
core.set_cache_limit = core.set_memory_limit = lambda limit: 0
server = types.ModuleType("mlx_lm.server")
def serve():
    open(pid_file + ".tmp", "w").write(str(os.getpid()))
    os.replace(pid_file + ".tmp", pid_file)
    time.sleep(60)
server.main = serve
sys.modules.update({"mlx.core": core, "mlx_lm.server": server})
from evals import mlx_server
mlx_server.PARENT_POLL_S = 0.05
mlx_server.main(["--memory-cap-bytes", str(1 << 40), "--"])
"""

# The eval's shape: the lock held, its descriptor passed on, the server under `background` inside `exit_on_signals`.
PARENT = """
import fcntl, os, sys, time
from pathlib import Path
from ytverbatim import proc
server, root, pid_file, lock, log = sys.argv[1:]
fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
watch = proc.MemoryWatch(1 << 40, poll_s=0.05)
with proc.exit_on_signals(), proc.background(
    [sys.executable, server, root, pid_file], log=Path(log), memory_watch=watch, pass_fds=(fd,)
):
    time.sleep(60)
"""


def _lock_free(path: Path) -> bool:
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    finally:
        os.close(fd)
    return True


@pytest.mark.parametrize("signum", [signal.SIGKILL, signal.SIGTERM, signal.SIGHUP])
def test_the_server_and_its_lock_do_not_outlive_a_killed_parent(tmp_path: Path, signum: signal.Signals) -> None:
    server, parent_script = tmp_path / "server.py", tmp_path / "parent.py"
    server.write_text(textwrap.dedent(SERVER), encoding="utf-8")
    parent_script.write_text(textwrap.dedent(PARENT), encoding="utf-8")
    pid_file, lock, log = tmp_path / "pid", tmp_path / "lock", tmp_path / "server.log"
    parent = subprocess.Popen(
        [sys.executable, str(parent_script), str(server), str(REPO), str(pid_file), str(lock), str(log)]
    )
    try:
        deadline = time.monotonic() + 20
        while not pid_file.exists():
            assert parent.poll() is None, log.read_text(encoding="utf-8") if log.exists() else "no log"
            assert time.monotonic() < deadline, log.read_text(encoding="utf-8") if log.exists() else "no log"
            time.sleep(0.02)
        child = int(pid_file.read_text(encoding="utf-8"))
        parent.send_signal(signum)
        returncode = parent.wait(timeout=20)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
    gone = _gone(lambda: os.kill(child, 0))
    if not gone:
        os.kill(child, signal.SIGKILL)
    assert gone
    assert returncode == (-signal.SIGKILL if signum == signal.SIGKILL else 128 + signum)
    assert _lock_free(lock)
