import fcntl
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from ytverbatim import paths
from ytverbatim.errors import ProcTimeout, YtDlpError
from ytverbatim.proc import ProcResult
from ytverbatim.ytdlp import runner as runner_mod
from ytverbatim.ytdlp.runner import SLEEP_PRESET, SubprocessRunner, redact
from ytverbatim.ytdlp.types import Signal, YtDlpRunner

VID = "BaW_jenozKc"
TRANSIENT = f"ERROR: [youtube] {VID}: Unable to download webpage: HTTP Error 503: Service Unavailable"
FORBIDDEN = "ERROR: unable to download video data: HTTP Error 403: Forbidden"
ROTATED = "WARNING: [youtube] The provided YouTube account cookies are no longer valid. They have likely been rotated."


class FakeProc:
    def __init__(self, *results: tuple[int, str]) -> None:
        self.results = list(results)
        self.calls: list[list[str]] = []
        self.lock_held: list[bool] = []

    def __call__(self, argv: Sequence[str], *, timeout_s: float, **_: object) -> ProcResult:
        self.calls.append(list(argv))
        self.lock_held.append(_lock_is_held())
        code, stderr = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        return ProcResult(code, "{}" if code == 0 else "", stderr, 0.1)


def _lock_is_held() -> bool:
    fd = os.open(paths.locks_dir() / "ytdlp.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


@pytest.fixture(autouse=True)
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))


def _install(monkeypatch: pytest.MonkeyPatch, *results: tuple[int, str]) -> FakeProc:
    fake = FakeProc(*results)
    monkeypatch.setattr(runner_mod.proc, "run", fake)
    return fake


def _runner(sleeps: list[float]) -> SubprocessRunner:
    return SubprocessRunner(js_args=("--js-runtimes", "node"), sleep=sleeps.append)


def test_satisfies_protocol() -> None:
    runner: YtDlpRunner = SubprocessRunner(js_args=())
    assert runner is not None


def test_argv_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _install(monkeypatch, (0, ""))
    cookies = tmp_path / "snap.txt"
    runner = SubprocessRunner(js_args=("--js-runtimes", "node"), sleep_args=SLEEP_PRESET)
    runner.run(["-J", "--no-playlist", VID], cookies=cookies, timeout_s=30, label="meta")
    assert fake.calls == [
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--ignore-config",
            "--no-progress",
            "--js-runtimes",
            "node",
            "-t",
            "sleep",
            "--cookies",
            str(cookies),
            "-J",
            "--no-playlist",
            VID,
        ]
    ]


def test_argv_without_cookies_or_extras() -> None:
    argv = SubprocessRunner(js_args=()).argv(["-J", VID], cookies=None)
    assert argv == [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-progress", "-J", VID]


def test_js_args_detected_once_when_not_given(monkeypatch: pytest.MonkeyPatch) -> None:
    detections: list[int] = []

    def detect() -> runner_mod.jsruntime.JsRuntime:
        detections.append(1)
        return runner_mod.jsruntime.JsRuntime("node", ("--js-runtimes", "node"), None)

    monkeypatch.setattr(runner_mod.jsruntime, "detect", detect)
    runner = SubprocessRunner()
    assert "--js-runtimes" in runner.argv([VID], cookies=None)
    runner.argv([VID], cookies=None)
    assert detections == [1]


def test_success_returns_and_logs_warnings(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _install(monkeypatch, (0, ROTATED))
    with caplog.at_level(logging.WARNING, logger="ytverbatim.ytdlp.runner"):
        result = _runner([]).run(["-J", VID], cookies=None, timeout_s=30, label="meta")
    assert (result.returncode, result.stdout, result.primary) == (0, "{}", None)
    assert result.secondary == (Signal.ROTATED,)
    assert result.warnings == (ROTATED.removeprefix("WARNING: "),)
    assert any("cookies are no longer valid" in r.getMessage() for r in caplog.records)


def test_warnings_and_preview_hide_home(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    home = str(Path.home())
    _install(monkeypatch, (0, f"WARNING: wrote {home}/x/cookies.txt"))
    with caplog.at_level(logging.WARNING):
        result = _runner([]).run([VID], cookies=None, timeout_s=30, label="meta")
    assert result.warnings == ("wrote ~/x/cookies.txt",)
    assert home not in caplog.text

    _install(monkeypatch, (1, f"ERROR: cannot open {home}/x/cookies.txt"))
    with pytest.raises(YtDlpError) as err:
        _runner([]).run([VID], cookies=None, timeout_s=30, label="meta")
    assert home not in err.value.preview
    assert "~/x/cookies.txt" in err.value.preview


def test_redact_leaves_other_text() -> None:
    assert redact("/tmp/example/file") == "/tmp/example/file"


def test_transient_retries_twice_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, (1, TRANSIENT))
    sleeps: list[float] = []
    with pytest.raises(YtDlpError) as err:
        _runner(sleeps).run([VID], cookies=None, timeout_s=30, label="meta")
    assert len(fake.calls) == 3
    assert sleeps == [5.0, 10.0]
    assert (err.value.primary, err.value.returncode) == (Signal.TRANSIENT, 1)


def test_transient_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, (1, TRANSIENT), (0, ""))
    sleeps: list[float] = []
    result = _runner(sleeps).run([VID], cookies=None, timeout_s=30, label="meta")
    assert result.returncode == 0
    assert (len(fake.calls), sleeps) == (2, [5.0])


def test_http_403_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, (1, FORBIDDEN))
    sleeps: list[float] = []
    with pytest.raises(YtDlpError) as err:
        _runner(sleeps).run([VID], cookies=None, timeout_s=30, label="audio")
    assert (len(fake.calls), sleeps, err.value.primary) == (2, [5.0], Signal.HTTP_403)


@pytest.mark.parametrize(
    ("stderr", "signal"),
    [
        (f"ERROR: [youtube] {VID}: Sign in to confirm you’re not a bot. Use --cookies", Signal.BOT_CHECK),
        (f"ERROR: [youtube] {VID}: Your IP is likely being blocked by Youtube", Signal.IP_BLOCKED),
        (f"ERROR: [youtube] {VID}: YouTube is requiring a captcha challenge before playback", Signal.CAPTCHA),
        (f"ERROR: [youtube] {VID}: This content isn’t available, try again later.", Signal.RATE_LIMITED),
        ("ERROR: HTTP Error 429: Too Many Requests", Signal.HTTP_429),
        ("ERROR: HTTP Error 511: Network Authentication Required", None),
        ("ERROR: something nobody has seen before", None),
        ("Traceback (most recent call last):\nKeyError: 'x'", None),
    ],
)
def test_never_retried(monkeypatch: pytest.MonkeyPatch, stderr: str, signal: Signal | None) -> None:
    fake = _install(monkeypatch, (1, stderr))
    sleeps: list[float] = []
    with pytest.raises(YtDlpError) as err:
        _runner(sleeps).run([VID], cookies=None, timeout_s=30, label="meta")
    assert (len(fake.calls), sleeps, err.value.primary) == (1, [], signal)


def test_members_only_with_n_challenge_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    stderr = "\n".join(
        [
            f"WARNING: [youtube] {VID}: n challenge solving failed: Some formats may be missing.",
            f"ERROR: [youtube] {VID}: Join this channel to get access to members-only content like this video.",
        ]
    )
    fake = _install(monkeypatch, (1, stderr))
    with pytest.raises(YtDlpError) as err:
        _runner([]).run([VID], cookies=None, timeout_s=30, label="meta")
    assert len(fake.calls) == 1
    assert err.value.primary is Signal.MEMBERS_ONLY
    assert Signal.N_CHALLENGE in err.value.secondary


def test_no_formats_after_js_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    stderr = "\n".join(
        [
            f"WARNING: [youtube] {VID}: n challenge solving failed: Some formats may be missing.",
            f"ERROR: [youtube] {VID}: Requested format is not available.",
        ]
    )
    fake = _install(monkeypatch, (1, stderr))
    with pytest.raises(YtDlpError) as err:
        _runner([]).run([VID], cookies=None, timeout_s=30, label="audio")
    assert len(fake.calls) == 1
    assert (err.value.primary, err.value.secondary) == (Signal.NO_FORMATS, (Signal.N_CHALLENGE,))


def test_rotated_on_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(
        monkeypatch, (1, f"{ROTATED}\nERROR: [youtube] {VID}: Join this channel to get access to members-only content")
    )
    with pytest.raises(YtDlpError) as err:
        _runner([]).run([VID], cookies=None, timeout_s=30, label="meta")
    assert err.value.secondary == (Signal.ROTATED,)


def test_lock_is_held_during_each_call_and_released_after(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, (1, TRANSIENT), (0, ""))
    _runner([]).run([VID], cookies=None, timeout_s=30, label="meta")
    assert fake.lock_held == [True, True]
    assert not _lock_is_held()
    assert (paths.locks_dir() / "ytdlp.lock").exists()


def test_lock_released_when_the_process_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(argv: Sequence[str], *, timeout_s: float, **_: object) -> ProcResult:
        raise ProcTimeout("python", timeout_s)

    monkeypatch.setattr(runner_mod.proc, "run", timeout)
    with pytest.raises(ProcTimeout):
        _runner([]).run([VID], cookies=None, timeout_s=1, label="meta")
    assert not _lock_is_held()
