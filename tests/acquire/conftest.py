from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ytverbatim import proc
from ytverbatim.errors import YtDlpError
from ytverbatim.ytdlp.types import Signal, YtDlpResult

Handler = Callable[[list[str], str], YtDlpResult]


@dataclass(frozen=True)
class RunnerCall:
    args: list[str]
    cookies: Path | None
    timeout_s: float
    label: str


@dataclass
class FakeRunner:
    handler: Handler
    calls: list[RunnerCall] = field(default_factory=list)

    def run(self, args: Sequence[str], *, cookies: Path | None, timeout_s: float, label: str) -> YtDlpResult:
        self.calls.append(RunnerCall(list(args), cookies, timeout_s, label))
        return self.handler(list(args), label)


class Fakes:
    """Constructors for the doubles; a fixture because pyrefly cannot resolve imports between test files."""

    @staticmethod
    def runner(handler: Handler) -> FakeRunner:
        return FakeRunner(handler)

    @staticmethod
    def ok(
        stdout: str = "",
        *,
        primary: Signal | None = None,
        secondary: tuple[Signal, ...] = (),
        warnings: tuple[str, ...] = (),
    ) -> YtDlpResult:
        return YtDlpResult(0, stdout, "", primary, secondary, warnings)

    @staticmethod
    def fail(primary: Signal | None, *secondary: Signal) -> YtDlpError:
        return YtDlpError(primary, secondary, "synthetic failure", 1)

    @staticmethod
    def raises(error: YtDlpError) -> Handler:
        def handler(_args: list[str], _label: str) -> YtDlpResult:
            raise error

        return handler


class FakeProc:
    """Stands in for `proc.run`; responses are keyed by executable name."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.responses: dict[str, Callable[[list[str]], proc.ProcResult]] = {}

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout_s: float,
        env: object = None,
        cwd: object = None,
        stdin: str | None = None,
    ) -> proc.ProcResult:
        args = list(argv)
        self.calls.append(args)
        return self.responses[Path(args[0]).name](args)

    def probe_returns(self, stdout: str, returncode: int = 0, stderr: str = "") -> None:
        self.responses["ffprobe"] = lambda _argv: proc.ProcResult(returncode, stdout, stderr, 0.0)


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()


@pytest.fixture
def fake_proc(monkeypatch: pytest.MonkeyPatch) -> FakeProc:
    fake = FakeProc()
    monkeypatch.setattr(proc, "run", fake)
    return fake
