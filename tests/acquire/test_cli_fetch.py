from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest
from click.testing import CliRunner

from ytverbatim.acquire import session
from ytverbatim.cli import fetch as cli_fetch
from ytverbatim.cli import main
from ytverbatim.ytdlp.types import Signal


def _invoke(kit, yt, monkeypatch: pytest.MonkeyPatch, *args: str):
    def runner(*, sleep_args: Sequence[str] = ()) -> object:
        kit.sleep_args.append(tuple(sleep_args))
        return yt

    monkeypatch.setattr(cli_fetch, "SubprocessRunner", runner)
    return CliRunner().invoke(main, ["fetch", "--out", str(kit.out), *args])


def test_fetch_is_registered() -> None:
    assert "fetch" in CliRunner().invoke(main, ["--help"]).output


def test_public_video_prints_one_line_and_exits_zero(session_kit, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _invoke(session_kit, session_kit.yt(), monkeypatch, "pubVideo001")
    assert result.exit_code == 0, result.output
    line = next(line for line in result.output.splitlines() if line.startswith("✓"))
    assert "pubVideo001" in line and "anonymous" in line and "2 chapters" in line and "audio 100s" in line
    assert "1 ok, 0 failed" in result.output
    assert str(session_kit.home) not in result.output


def test_members_without_source_exits_two_with_fix(session_kit, monkeypatch: pytest.MonkeyPatch) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    result = _invoke(session_kit, yt, monkeypatch, "memVideo001")
    assert result.exit_code == 2, result.output
    assert "auth_required" in result.output and "Fix:" in result.output


def test_permanent_failure_exits_one(session_kit, monkeypatch: pytest.MonkeyPatch) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.GEO)]
    result = _invoke(session_kit, yt, monkeypatch, "pubVideo001", "pubVideo002")
    assert result.exit_code == 1, result.output
    assert "1 ok, 1 failed" in result.output


def test_bot_check_exits_three_and_later_runs_are_refused(session_kit, monkeypatch: pytest.MonkeyPatch) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.BOT_CHECK)]
    result = _invoke(session_kit, yt, monkeypatch, "pubVideo001")
    assert result.exit_code == 3, result.output

    yt = session_kit.yt()
    refused = _invoke(session_kit, yt, monkeypatch, "pubVideo002")
    assert refused.exit_code == 3 and "--ignore-cooldown" in refused.output
    assert yt.calls == []
    allowed = _invoke(session_kit, yt, monkeypatch, "pubVideo002", "--ignore-cooldown")
    assert allowed.exit_code == 0, allowed.output


def test_expired_cooldown_does_not_refuse(session_kit, monkeypatch: pytest.MonkeyPatch) -> None:
    session.start_cooldown(Signal.RATE_LIMITED, datetime.now(UTC) - timedelta(hours=3))
    assert _invoke(session_kit, session_kit.yt(), monkeypatch, "pubVideo001").exit_code == 0


def test_options_reach_the_session(session_kit, monkeypatch: pytest.MonkeyPatch) -> None:
    yt = session_kit.yt()
    yt.playlist = [f"plVideo{n:04d}" for n in range(3)]
    result = _invoke(
        session_kit,
        yt,
        monkeypatch,
        "https://www.youtube.com/playlist?list=PLsynthetic",
        "--max-videos",
        "2",
        "--no-audio",
        "--no-captions",
    )
    assert result.exit_code == 0, result.output
    assert {c.label for c in yt.calls} == {"expand", "metadata"}
    assert len(yt.labels("metadata")) == 2
    assert "cap was reached" in result.output
    assert session_kit.sleep_args[-1] == ("-t", "sleep")


def test_invalid_target_exits_one(session_kit, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _invoke(session_kit, session_kit.yt(), monkeypatch, "https://example.com/nope")
    assert result.exit_code == 1 and "not a YouTube" in result.output
