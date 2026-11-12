import json
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pytest

from ytverbatim import paths, proc
from ytverbatim.auth import snapshot
from ytverbatim.auth.snapshot import Snapshot, SnapshotSummary
from ytverbatim.errors import YtDlpError
from ytverbatim.ytdlp import jsruntime, version
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


INFO_DURATION = 100.0
FIXTURES = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class YtCall:
    args: list[str]
    cookies: Path | None
    label: str

    @property
    def video_id(self) -> str:
        return self.args[-1].rsplit("=", 1)[-1]


@dataclass
class FakeYtDlp:
    """A scripted yt-dlp: metadata, captions, audio and expansion succeed unless a failure is queued.

    `failures[(label, cookied)]` holds errors raised in order by matching calls; `availability` and
    `secondary` shape a video's successful `-J`.
    """

    availability: dict[str, str] = field(default_factory=dict)
    secondary: dict[tuple[str, bool], tuple[Signal, ...]] = field(default_factory=dict)
    failures: dict[tuple[str, bool], list[YtDlpError]] = field(default_factory=dict)
    stderr: dict[tuple[str, bool], str] = field(default_factory=dict)
    playlist: list[str] = field(default_factory=list)
    no_captions: set[str] = field(default_factory=set)
    # Per video: its declared language, and its caption files by track code (an "-orig" code is the auto track).
    language: dict[str, str] = field(default_factory=dict)
    tracks: dict[str, dict[str, str]] = field(default_factory=dict)
    calls: list[YtCall] = field(default_factory=list)

    def run(self, args: Sequence[str], *, cookies: Path | None, timeout_s: float, label: str) -> YtDlpResult:
        call = YtCall(list(args), cookies, label)
        self.calls.append(call)
        key = (label, cookies is not None)
        queued = self.failures.get(key)
        if queued:
            raise queued.pop(0)
        secondary = self.secondary.get(key, ())
        stderr = self.stderr.get(key, "")
        if label == "metadata":
            return self._ok(json.dumps(self._info(call.video_id)), secondary, stderr)
        if label == "captions":
            target = args[args.index("-o") + 1].removeprefix("subtitle:").replace("%(ext)s", "")
            files = self.tracks.get(Path(target).parent.parent.name)
            if files is None:
                files = {"en-orig": (FIXTURES / "auto_en_orig.json3").read_text("utf-8")}
            for code, content in files.items():
                Path(f"{target}{code}.json3").write_text(content, "utf-8")
        elif label == "audio":
            Path(args[args.index("-o") + 1].replace("%(ext)s", "webm")).write_bytes(b"fake audio")
        elif label == "expand":
            return self._ok("\n".join(self.playlist) + "\n", secondary, stderr)
        return self._ok("", secondary, stderr)

    def _info(self, video_id: str) -> dict[str, object]:
        return {
            "id": video_id,
            "title": "Synthetic title",
            "channel": "Synthetic channel",
            "channel_id": "UCxxxxxxxxxxxxxxxxxxxxxx",
            "duration": INFO_DURATION,
            "language": self.language.get(video_id, "en"),
            "availability": self.availability.get(video_id, "public"),
            "chapters": [{"start_time": 0, "end_time": 50, "title": "One"}, {"start_time": 50, "end_time": 100}],
            "automatic_captions": self._auto(video_id),
            "subtitles": {code: [] for code in self.tracks.get(video_id, {}) if not code.endswith("-orig")},
        }

    def _auto(self, video_id: str) -> dict[str, list[object]]:
        if video_id in self.no_captions:
            return {}
        if video_id in self.tracks:
            return {code: [] for code in self.tracks[video_id] if code.endswith("-orig")}
        return {"en-orig": [], "en": [], "fr": []}

    @staticmethod
    def _ok(stdout: str, secondary: tuple[Signal, ...], stderr: str) -> YtDlpResult:
        return YtDlpResult(0, stdout, stderr, None, secondary, ())

    def labels(self, label: str, *, cookied: bool | None = None) -> list[YtCall]:
        return [c for c in self.calls if c.label == label and (cookied is None or (c.cookies is not None) == cookied)]


@dataclass
class FakeCookieSnapshot:
    """Writes a real placeholder file per snapshot so tests can check reuse and deletion."""

    taken: list[Path] = field(default_factory=list)

    @contextmanager
    def __call__(self, src: object, **_: object) -> Generator[Snapshot]:
        path = paths.snapshots_dir() / f"fake-{len(self.taken)}.txt"
        path.write_text("placeholder snapshot\n", encoding="utf-8")
        self.taken.append(path)
        try:
            yield Snapshot(path, SnapshotSummary({"youtube.com": 3, "google.com": 4}, 0, True, ()))
        finally:
            path.unlink(missing_ok=True)


@dataclass
class SessionKit:
    home: Path
    out: Path
    snapshots: FakeCookieSnapshot
    proc: FakeProc
    slept: list[float] = field(default_factory=list[float])
    sleep_args: list[tuple[str, ...]] = field(default_factory=list[tuple[str, ...]])

    @staticmethod
    def yt() -> FakeYtDlp:
        return FakeYtDlp()

    @staticmethod
    def fail(primary: Signal | None, *secondary: Signal, preview: str = "ERROR: synthetic failure") -> YtDlpError:
        return YtDlpError(primary, secondary, preview, 1)


@pytest.fixture
def session_kit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_proc: FakeProc) -> SessionKit:
    for name in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "YTVERBATIM_HOME", "YTVERBATIM_AUTH"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    released = date.today() - timedelta(days=10)
    monkeypatch.setattr(version, "installed", lambda: version.YtDlpVersion("2099.01.01", released))
    monkeypatch.setattr(jsruntime, "detect", lambda: jsruntime.JsRuntime("deno", (), None))
    snaps = FakeCookieSnapshot()
    monkeypatch.setattr(snapshot, "cookie_snapshot", snaps)
    fake_proc.probe_returns(f"{INFO_DURATION}\n")

    def ffmpeg(argv: list[str]) -> proc.ProcResult:
        Path(argv[-1]).write_bytes(b"RIFF fake wav")
        return proc.ProcResult(0, "", "", 0.0)

    fake_proc.responses["ffmpeg"] = ffmpeg
    return SessionKit(home, tmp_path / "out", snaps, fake_proc)
