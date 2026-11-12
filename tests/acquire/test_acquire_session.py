import json
import logging
import stat
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ytverbatim import paths
from ytverbatim.acquire import session
from ytverbatim.acquire.session import BatchReport, FetchOptions, VideoOutcome, VideoStatus
from ytverbatim.auth import preflight
from ytverbatim.auth.source import CookieFileSource
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError, BatchStopError
from ytverbatim.ytdlp.runner import SLEEP_PRESET
from ytverbatim.ytdlp.signals import classify
from ytverbatim.ytdlp.types import Signal, YtDlpRunner

PUBLIC = "pubVideo001"
PUBLIC_2 = "pubVideo002"
MEMBERS = "memVideo001"
NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
SOURCE = CookieFileSource(Path("/nonexistent/cookies.txt"))


def run(kit, yt, targets: Sequence[str], *, source=SOURCE, now: datetime = NOW, **options: object) -> BatchReport:
    kit.sleep_args = []

    def factory(sleep: Sequence[str]) -> YtDlpRunner:
        kit.sleep_args.append(tuple(sleep))
        return yt

    return session.fetch(
        targets,
        source=source,
        out=kit.out,
        options=FetchOptions(**options),
        runner_factory=factory,
        sleep=kit.slept.append,
        clock=lambda: now,
    )


def only(report: BatchReport) -> VideoOutcome:
    (outcome,) = report.videos
    return outcome


def test_anonymous_public_success_takes_no_snapshot(session_kit) -> None:
    yt = session_kit.yt()
    report = run(session_kit, yt, [PUBLIC])
    outcome = only(report)
    assert outcome.status is VideoStatus.OK and report.exit_code == 0
    assert session_kit.snapshots.taken == []
    assert len(yt.labels("metadata")) == 1
    assert [c for c in yt.calls if "-J" in c.args] == yt.labels("metadata")
    assert all(c.cookies is None for c in yt.calls)
    assert "-v" not in yt.labels("metadata")[0].args

    vdir = session_kit.out / PUBLIC
    assert stat.S_IMODE((vdir / "info.json").stat().st_mode) == 0o600
    for name in ("metadata.json", "access.json", "audio.webm", "audio.wav", "captions/tracks.json"):
        assert (vdir / name).is_file(), name
    access = json.loads((vdir / "access.json").read_text())
    assert access["cookies_used"] is False and access["auth_source"] == "cookie-file"
    assert access["member"] is None and access["availability"] == "public"
    assert outcome.chapters == 2
    assert outcome.captions == ("en-orig (auto, json3, word timings)",)
    batch = json.loads((session_kit.out / "batch_report.json").read_text())
    assert batch["videos"][0]["status"] == "ok" and batch["stopped"] is None


def test_members_escalation_reuses_one_snapshot_and_sets_member(session_kit) -> None:
    yt = session_kit.yt()
    yt.availability[MEMBERS] = "subscriber_only"
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    yt.secondary[("metadata", True)] = (Signal.AUTH_COOKIES_FOUND,)
    outcome = only(run(session_kit, yt, [MEMBERS]))

    assert outcome.status is VideoStatus.OK, outcome
    assert outcome.access is not None
    assert outcome.access.cookies_used and outcome.access.member is True
    assert outcome.access.logged_in is True and outcome.access.premium is False
    (snap,) = session_kit.snapshots.taken
    cookied = [c for c in yt.calls if c.cookies is not None]
    assert {c.label for c in cookied} == {"metadata", "captions", "audio"}
    assert all(c.cookies == snap for c in cookied)
    assert "-v" in yt.labels("metadata", cookied=True)[0].args
    assert not snap.exists()


def test_escalation_without_source_is_an_auth_error(session_kit) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.LOGIN_REQUIRED)]
    report = run(session_kit, yt, [MEMBERS], source=None)
    outcome = only(report)
    assert outcome.error == AuthState.AUTH_REQUIRED.value and outcome.auth_error
    assert "auth use" in outcome.fix
    assert report.exit_code == 2
    assert yt.labels("metadata", cookied=True) == [] and session_kit.snapshots.taken == []


def test_policy_never_does_not_escalate(session_kit) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    outcome = only(run(session_kit, yt, [MEMBERS], policy="never"))
    assert outcome.error == AuthState.AUTH_REQUIRED.value
    assert session_kit.snapshots.taken == []


def test_policy_always_needs_a_source(session_kit) -> None:
    with pytest.raises(AuthError) as caught:
        run(session_kit, session_kit.yt(), [PUBLIC], source=None, policy="always")
    assert caught.value.state is AuthState.NO_SOURCE


def test_rotation_retries_once_with_a_fresh_snapshot(session_kit) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    yt.failures[("metadata", True)] = [session_kit.fail(Signal.MEMBERS_ONLY, Signal.ROTATED)]
    outcome = only(run(session_kit, yt, [MEMBERS]))
    assert outcome.status is VideoStatus.OK
    assert session_kit.slept == [session.ROTATION_WAIT_S]
    first, second = session_kit.snapshots.taken
    assert [c.cookies for c in yt.labels("metadata", cookied=True)] == [first, second]
    assert yt.labels("audio")[0].cookies == second


def test_rotation_twice_is_an_auth_error_and_poisons_later_cookied_videos(session_kit) -> None:
    yt = session_kit.yt()
    rotated = [session_kit.fail(Signal.MEMBERS_ONLY, Signal.ROTATED) for _ in range(2)]
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY) for _ in range(2)]
    yt.failures[("metadata", True)] = rotated
    report = run(session_kit, yt, [MEMBERS, "memVideo002"])
    first, second = report.videos
    assert first.error == AuthState.ROTATED.value and second.error == AuthState.ROTATED.value
    assert session_kit.slept == [session.ROTATION_WAIT_S]
    assert len(session_kit.snapshots.taken) == 2
    assert len(yt.labels("metadata", cookied=True)) == 2
    assert report.exit_code == 2


def test_rotated_on_success_invalidates_preflight(session_kit, monkeypatch: pytest.MonkeyPatch) -> None:
    invalidated: list[object] = []
    monkeypatch.setattr(preflight, "invalidate", invalidated.append)
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    yt.secondary[("metadata", True)] = (Signal.ROTATED,)
    outcome = only(run(session_kit, yt, [MEMBERS]))
    assert outcome.status is VideoStatus.OK
    assert invalidated == [SOURCE]
    assert session_kit.slept == []


def test_bot_check_writes_cooldown_and_stops_without_escalating(session_kit) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.BOT_CHECK)]
    report = run(session_kit, yt, [PUBLIC, PUBLIC_2])
    first, second = report.videos
    assert report.exit_code == 3 and report.stopped is not None
    assert first.error == Signal.BOT_CHECK.value and "Do not add cookies" in first.fix
    assert second.status is VideoStatus.NOT_ATTEMPTED
    assert session_kit.snapshots.taken == [] and yt.labels("metadata", cookied=True) == []
    assert len(yt.labels("metadata")) == 1
    cooldown = json.loads(paths.cooldown_file().read_text())
    assert cooldown["until"] == (NOW + timedelta(hours=1)).isoformat(timespec="seconds")
    assert cooldown["signal"] == "bot_check"
    batch = json.loads((session_kit.out / "batch_report.json").read_text())
    assert batch["stopped"]["signal"] == "bot_check"
    assert [v["status"] for v in batch["videos"]] == ["error", "not_attempted"]


def test_bot_check_on_cookied_call_also_stops(session_kit) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    yt.failures[("metadata", True)] = [session_kit.fail(Signal.CAPTCHA, Signal.ROTATED)]
    report = run(session_kit, yt, [MEMBERS])
    assert report.exit_code == 3 and session_kit.slept == []


def test_http_429_on_metadata_or_audio_stops_but_on_captions_only_drops_them(session_kit) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.HTTP_429)]
    assert run(session_kit, yt, [PUBLIC]).exit_code == 3
    assert json.loads(paths.cooldown_file().read_text())["signal"] == "http_429"

    paths.cooldown_file().unlink()
    yt = session_kit.yt()
    yt.failures[("audio", False)] = [session_kit.fail(Signal.HTTP_429)]
    assert run(session_kit, yt, [PUBLIC_2]).exit_code == 3

    paths.cooldown_file().unlink()
    yt = session_kit.yt()
    yt.failures[("captions", False)] = [session_kit.fail(Signal.HTTP_429)]
    report = run(session_kit, yt, ["pubVideo003"])
    outcome = only(report)
    assert report.exit_code == 0 and outcome.captions == ()
    assert outcome.access is not None and any("captions skipped" in w for w in outcome.access.warnings)
    assert not (session_kit.out / "pubVideo003" / "captions" / "tracks.json").exists()
    assert not paths.cooldown_file().exists()


def test_stop_signal_among_secondary_signals_never_escalates(session_kit) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY, Signal.BOT_CHECK)]
    report = run(session_kit, yt, [MEMBERS, PUBLIC])
    first, second = report.videos
    assert report.exit_code == 3 and report.stopped is not None and report.stopped.signal is Signal.BOT_CHECK
    assert first.error == "bot_check" and second.status is VideoStatus.NOT_ATTEMPTED
    assert session_kit.snapshots.taken == [] and yt.labels("metadata", cookied=True) == []
    assert json.loads(paths.cooldown_file().read_text())["signal"] == "bot_check"


def test_a_429_among_secondary_signals_stops_except_on_captions(session_kit) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.LOGIN_REQUIRED, Signal.HTTP_429)]
    assert run(session_kit, yt, [MEMBERS]).exit_code == 3
    assert session_kit.snapshots.taken == []

    paths.cooldown_file().unlink()
    yt = session_kit.yt()
    yt.failures[("captions", False)] = [session_kit.fail(Signal.HTTP_429, Signal.CAPTCHA)]
    assert run(session_kit, yt, [PUBLIC_2]).exit_code == 3


def test_stop_signal_on_a_successful_call_finishes_the_video_then_stops(session_kit) -> None:
    yt = session_kit.yt()
    yt.secondary[("metadata", False)] = (Signal.RATE_LIMITED,)
    yt.secondary[("audio", False)] = (Signal.BOT_CHECK,)
    report = run(session_kit, yt, [PUBLIC, PUBLIC_2])
    first, second = report.videos
    assert first.status is VideoStatus.OK and first.audio_s is not None
    assert second.status is VideoStatus.NOT_ATTEMPTED
    assert report.exit_code == 3 and report.stopped is not None and report.stopped.signal is Signal.RATE_LIMITED
    assert [c.video_id for c in yt.labels("metadata")] == [PUBLIC]
    cooldown = json.loads(paths.cooldown_file().read_text())
    assert cooldown["signal"] == "rate_limited" and cooldown["hours"] == 1
    assert json.loads((session_kit.out / "batch_report.json").read_text())["stopped"]["signal"] == "rate_limited"


def test_a_429_on_a_successful_captions_call_does_not_stop(session_kit) -> None:
    yt = session_kit.yt()
    yt.secondary[("captions", False)] = (Signal.HTTP_429,)
    report = run(session_kit, yt, [PUBLIC, PUBLIC_2])
    assert report.exit_code == 0 and report.stopped is None
    assert not paths.cooldown_file().exists()


def test_stop_signal_while_expanding_stops_before_any_video(session_kit) -> None:
    yt = session_kit.yt()
    yt.playlist = [PUBLIC]
    yt.secondary[("expand", False)] = (Signal.BOT_CHECK,)
    with pytest.raises(BatchStopError):
        run(session_kit, yt, ["https://www.youtube.com/playlist?list=PLsynthetic"])
    assert yt.labels("metadata") == []
    assert paths.cooldown_file().exists()


def test_stop_signal_on_the_login_probe_stops_the_batch(session_kit) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    yt.failures[("metadata", True)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    yt.failures[("auth-probe", True)] = [session_kit.fail(Signal.BOT_CHECK)]
    report = run(session_kit, yt, [MEMBERS, PUBLIC])
    assert report.exit_code == 3 and report.videos[1].status is VideoStatus.NOT_ATTEMPTED
    assert json.loads(paths.cooldown_file().read_text())["hours"] == 1


def test_auth_uncertain_is_one_videos_failure_not_the_sources(session_kit) -> None:
    yt = session_kit.yt()
    yt.availability["memVideo002"] = "subscriber_only"
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY) for _ in range(2)]
    yt.failures[("metadata", True)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    yt.failures[("auth-probe", True)] = [session_kit.fail(None)]
    report = run(session_kit, yt, [MEMBERS, "memVideo002"])
    first, second = report.videos
    assert first.error == AuthState.AUTH_UNCERTAIN.value
    assert second.status is VideoStatus.OK, second
    assert second.access is not None and second.access.member is True
    assert len(yt.labels("metadata", cookied=True)) == 2


def test_cooldown_refuses_to_start_unless_ignored(session_kit) -> None:
    session.start_cooldown(Signal.RATE_LIMITED, NOW)
    yt = session_kit.yt()
    with pytest.raises(BatchStopError) as caught:
        run(session_kit, yt, [PUBLIC], now=NOW + timedelta(minutes=30))
    assert caught.value.signal is Signal.RATE_LIMITED
    assert yt.calls == []
    report = run(session_kit, yt, [PUBLIC], now=NOW + timedelta(minutes=30), ignore_cooldown=True)
    assert report.exit_code == 0
    assert run(session_kit, session_kit.yt(), [PUBLIC_2], now=NOW + timedelta(hours=2)).exit_code == 0


def test_fetch_takes_the_per_video_lock_that_run_uses(session_kit) -> None:
    yt = session_kit.yt()
    with session.video_lock(session_kit.out / PUBLIC):
        report = run(session_kit, yt, [PUBLIC, PUBLIC_2])
    first, second = report.videos
    assert (first.status, first.error) == (VideoStatus.ERROR, "busy") and PUBLIC in first.detail
    assert second.status is VideoStatus.OK
    assert {call.video_id for call in yt.labels("metadata")} == {PUBLIC_2}


def test_cooldown_doubles_on_repeats_and_is_capped(session_kit) -> None:
    now = NOW
    durations: list[float] = []
    for _ in range(7):
        until = session.start_cooldown(Signal.BOT_CHECK, now).until
        durations.append((until - now) / timedelta(hours=1))
        now = until + timedelta(minutes=5)
    assert durations == [1, 2, 4, 8, 16, 24, 24]
    reset = session.start_cooldown(Signal.BOT_CHECK, now + timedelta(hours=25)).until
    assert reset - (now + timedelta(hours=25)) == timedelta(hours=1)
    assert set(json.loads(paths.cooldown_file().read_text())) >= {"until", "signal"}


@pytest.mark.parametrize(
    ("probe_error", "expected"),
    [
        (None, AuthState.NOT_MEMBER),
        (Signal.LOGIN_REQUIRED, AuthState.COOKIE_FILE_LOGGED_OUT),
        (None, AuthState.AUTH_UNCERTAIN),
    ],
)
def test_not_member_only_after_a_logged_in_probe(session_kit, probe_error, expected) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    yt.failures[("metadata", True)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    if probe_error is not None:
        yt.failures[("auth-probe", True)] = [session_kit.fail(probe_error)]
    elif expected is AuthState.AUTH_UNCERTAIN:
        yt.failures[("auth-probe", True)] = [session_kit.fail(None)]
    outcome = only(run(session_kit, yt, [MEMBERS]))
    assert outcome.error == expected.value
    (probe,) = yt.labels("auth-probe")
    assert ":ytfav" in probe.args


def test_per_run_cap_limits_expansion_and_videos(session_kit) -> None:
    yt = session_kit.yt()
    yt.playlist = [f"plVideo{n:04d}" for n in range(5)]
    report = run(session_kit, yt, ["https://www.youtube.com/playlist?list=PLsynthetic"], max_videos=2)
    assert [v.video_id for v in report.videos] == ["plVideo0000", "plVideo0001"]
    assert report.capped
    (expand,) = yt.labels("expand")
    assert expand.args[expand.args.index("--playlist-end") + 1] == "2"


def test_sleep_args_depend_on_video_count(session_kit) -> None:
    run(session_kit, session_kit.yt(), [PUBLIC])
    assert session_kit.sleep_args[-1] == ("--sleep-requests", "0.75", "--sleep-subtitles", "5")
    run(session_kit, session_kit.yt(), [PUBLIC, PUBLIC_2])
    assert session_kit.sleep_args[-1] == SLEEP_PRESET


def test_members_playlist_and_cookie_channels_use_cookies_first(session_kit) -> None:
    yt = session_kit.yt()
    yt.playlist = [MEMBERS]
    run(session_kit, yt, ["https://www.youtube.com/playlist?list=UUMOxxxxxxxxxxxxxxxxxxxxxx"])
    assert yt.labels("expand")[0].cookies is not None
    assert yt.labels("metadata", cookied=False) == []
    assert len(yt.labels("metadata", cookied=True)) == 1

    yt = session_kit.yt()
    yt.playlist = [PUBLIC]
    run(session_kit, yt, ["https://www.youtube.com/@SyntheticChannel"], cookie_channels=("@syntheticchannel",))
    assert yt.labels("expand")[0].cookies is None
    assert yt.labels("metadata", cookied=False) == []


def test_resume_skips_validated_artifacts(session_kit) -> None:
    run(session_kit, session_kit.yt(), [PUBLIC])
    yt = session_kit.yt()
    outcome = only(run(session_kit, yt, [PUBLIC]))
    assert yt.calls == []
    assert outcome.status is VideoStatus.OK
    assert outcome.reused == ("info", "captions", "audio", "wav")
    assert outcome.captions == ("en-orig (auto, json3, word timings)",)

    (session_kit.out / PUBLIC / "audio.wav").unlink()
    yt = session_kit.yt()
    outcome = only(run(session_kit, yt, [PUBLIC]))
    assert yt.calls == [] and outcome.reused == ("info", "captions", "audio")
    assert (session_kit.out / PUBLIC / "audio.wav").is_file()


def test_video_without_captions_records_an_empty_track_list(session_kit) -> None:
    yt = session_kit.yt()
    yt.no_captions.add(PUBLIC)
    outcome = only(run(session_kit, yt, [PUBLIC]))
    assert outcome.status is VideoStatus.OK and outcome.captions == ()
    assert yt.labels("captions") == []
    tracks = json.loads((session_kit.out / PUBLIC / "captions" / "tracks.json").read_text())
    assert tracks == {"tracks": []}
    yt = session_kit.yt()
    assert only(run(session_kit, yt, [PUBLIC])).reused == ("info", "captions", "audio", "wav")
    assert yt.calls == []


def test_permanent_failure_is_recorded_and_the_batch_continues(session_kit) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [
        session_kit.fail(Signal.UNAVAILABLE, preview="ERROR: [youtube] x: Video unavailable")
    ]
    report = run(session_kit, yt, [PUBLIC, PUBLIC_2])
    first, second = report.videos
    assert first.error == "unavailable" and first.detail.startswith("ERROR:") and first.fix
    assert second.status is VideoStatus.OK
    assert report.exit_code == 1


def test_exit_codes() -> None:
    ok = VideoOutcome(PUBLIC, VideoStatus.OK)
    failed = VideoOutcome(PUBLIC_2, VideoStatus.ERROR, error="geo")
    auth = VideoOutcome(MEMBERS, VideoStatus.ERROR, error="not_member", auth_error=True)
    assert BatchReport((ok,)).exit_code == 0
    assert BatchReport((ok, failed)).exit_code == 1
    assert BatchReport((failed, auth)).exit_code == 2
    assert BatchReport((ok,), stopped=BatchStopError(Signal.BOT_CHECK, NOW)).exit_code == 3


SECRET = "SECRET-ARGV-MARKER"
VERBOSE_STDERR = (
    f"[debug] Command-line config: ['-v', '--cookies', '/tmp/{SECRET}.txt', '-J']\n"
    "[debug] [youtube] Found YouTube account cookies\n"
    "[debug] [youtube] Detected YouTube Premium subscription\n"
)


def _files_text(root: Path) -> str:
    return "\n".join(p.read_text("utf-8", errors="replace") for p in root.rglob("*") if p.is_file())


def test_premium_parsed_from_verbose_and_never_written_raw(session_kit, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    yt = session_kit.yt()
    yt.availability[MEMBERS] = "subscriber_only"
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    yt.stderr[("metadata", True)] = VERBOSE_STDERR
    yt.secondary[("metadata", True)] = classify(VERBOSE_STDERR).secondary
    outcome = only(run(session_kit, yt, [MEMBERS]))
    assert outcome.access is not None and outcome.access.premium is True
    entries = json.loads(paths.preflight_cache_file().read_text())["entries"]
    assert [entry["premium"]["value"] for entry in entries.values()] == [True]
    assert SECRET not in _files_text(session_kit.out) + _files_text(session_kit.home)
    assert SECRET not in caplog.text


def test_failed_verbose_call_drops_debug_lines(session_kit) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    preview = f"{VERBOSE_STDERR}ERROR: [youtube] {MEMBERS}: something new and unclassified"
    yt.failures[("metadata", True)] = [session_kit.fail(None, preview=preview)]
    outcome = only(run(session_kit, yt, [MEMBERS]))
    assert outcome.error == "unclassified" and "something new" in outcome.detail
    assert SECRET not in _files_text(session_kit.out) + _files_text(session_kit.home)
