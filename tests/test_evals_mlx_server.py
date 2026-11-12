import sys
from types import ModuleType

import pytest

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
    return log


def test_the_allocator_is_bounded_before_the_server_runs_with_the_rest_of_argv(calls: list[str]) -> None:
    mlx_server.main(["--memory-cap-bytes", str(CAP), "--", "--model", "/models/m", "--port", "8123"])
    assert calls == [
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
    assert calls == []


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
    assert calls == []
