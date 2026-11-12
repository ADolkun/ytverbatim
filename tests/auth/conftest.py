import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pytest

from ytverbatim.auth import chromium, snapshot, source
from ytverbatim.auth.snapshot import Snapshot, SnapshotSummary
from ytverbatim.auth.source import CookieSource
from ytverbatim.errors import AuthError, YtDlpError
from ytverbatim.ytdlp import jsruntime, version
from ytverbatim.ytdlp.types import Signal, YtDlpResult

PROFILES = {
    "Default": {"name": "Person 1", "user_name": "alice@example.com"},
    "Work": {"name": "Work", "user_name": "bob@example.com"},
}


@dataclass(frozen=True)
class RunnerCall:
    args: list[str]
    cookies: Path | None


@dataclass
class FakeRunner:
    outcome: YtDlpResult | YtDlpError
    calls: list[RunnerCall] = field(default_factory=list)

    def run(self, args: Sequence[str], *, cookies: Path | None, timeout_s: float, label: str) -> YtDlpResult:
        self.calls.append(RunnerCall(list(args), cookies))
        if isinstance(self.outcome, YtDlpError):
            raise self.outcome
        return self.outcome


@dataclass
class FakeSnapshot:
    """Stands in for the browser snapshot (which would need Chrome and the Keychain)."""

    error: AuthError | None = None
    counts: dict[str, int] = field(default_factory=lambda: {"youtube.com": 12, "google.com": 30})
    sources: list[CookieSource] = field(default_factory=list)

    @contextmanager
    def __call__(self, src: CookieSource, **_: object) -> Generator[Snapshot]:
        self.sources.append(src)
        if self.error is not None:
            raise self.error
        profile = src.profile if isinstance(src, source.BrowserSource) else None
        yield Snapshot(Path("/nonexistent/snapshot.txt"), SnapshotSummary(self.counts, 0, True, (), profile))


class Kit:
    def __init__(self, home: Path, snap: FakeSnapshot) -> None:
        self.home = home
        self.snapshot = snap

    @staticmethod
    def ok(*secondary: Signal, stdout: str = "fakeLikedId\n") -> FakeRunner:
        return FakeRunner(YtDlpResult(0, stdout, "", None, secondary, ()))

    @staticmethod
    def fail(primary: Signal | None, *secondary: Signal) -> FakeRunner:
        return FakeRunner(YtDlpError(primary, secondary, "ERROR: synthetic fakeLikedId", 1))

    @property
    def chrome_root(self) -> Path:
        return chromium.browser_root("chrome")


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "YTVERBATIM_HOME", source.ENV_VAR, jsruntime.ENV_OVERRIDE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    root = chromium.browser_root("chrome")
    for directory in PROFILES:
        (root / directory).mkdir(parents=True)
        (root / directory / "Cookies").write_bytes(b"sqlite")
    (root / "Local State").write_text(
        json.dumps({"profile": {"last_used": "Default", "info_cache": PROFILES}}), encoding="utf-8"
    )
    released = date.today() - timedelta(days=10)
    monkeypatch.setattr(version, "installed", lambda: version.YtDlpVersion("2099.01.01", released))
    monkeypatch.setattr(jsruntime, "detect", lambda: jsruntime.JsRuntime("deno", (), None))
    return tmp_path


@pytest.fixture
def kit(home: Path, monkeypatch: pytest.MonkeyPatch) -> Kit:
    snap = FakeSnapshot()
    monkeypatch.setattr(snapshot, "cookie_snapshot", snap)
    return Kit(home, snap)


@pytest.fixture
def fakes() -> type[Kit]:
    """The runner doubles without the fake snapshot, for tests that take real cookie-file snapshots."""
    return Kit
