from datetime import timedelta
from pathlib import Path

import pytest

import ytverbatim
from ytverbatim import config, paths
from ytverbatim.acquire import session
from ytverbatim.auth.cooldown import read_cooldown, start_cooldown, utcnow
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError, BatchStopError, YtDlpError
from ytverbatim.ytdlp.types import Signal

LIST_URL = "https://www.youtube.com/playlist?list=PLsynthetic"
MEMBERS_URL = "https://www.youtube.com/playlist?list=UUMOxxxxxxxxxxxxxxxxxxxxxx"
CHANNEL_URL = "https://www.youtube.com/@SyntheticChannel"
COOKIE_FILE = "cookie-file:/nonexistent/cookies.txt"


@pytest.fixture
def yt(session_kit, monkeypatch: pytest.MonkeyPatch):
    fake = session_kit.yt()
    monkeypatch.setattr(session, "default_runner", lambda _sleep: fake)
    return fake


def options(**auth: object) -> config.Options:
    return config.load_options(overrides={"auth": auth})


def test_it_is_part_of_the_library_surface() -> None:
    assert ytverbatim.expand is session.expand
    assert "expand" in ytverbatim.__all__


def test_a_video_costs_no_request_even_during_a_cooldown(session_kit, yt) -> None:
    start_cooldown(Signal.BOT_CHECK, utcnow())
    assert session.expand("https://youtu.be/aaaaaaaaaaa") == ["aaaaaaaaaaa"]
    assert yt.calls == []


def test_a_cooldown_refuses_before_any_request(session_kit, yt) -> None:
    start_cooldown(Signal.BOT_CHECK, utcnow())
    yt.playlist = ["aaaaaaaaaaa"]
    with pytest.raises(BatchStopError) as stopped:
        session.expand(LIST_URL)
    assert stopped.value.signal is Signal.BOT_CHECK
    assert yt.calls == []


def test_an_ended_cooldown_does_not_refuse(session_kit, yt) -> None:
    start_cooldown(Signal.BOT_CHECK, utcnow() - timedelta(hours=3))
    yt.playlist = ["aaaaaaaaaaa"]
    assert session.expand(LIST_URL) == ["aaaaaaaaaaa"]


def test_ids_come_back_in_order_without_repeats_and_uncapped(session_kit, yt) -> None:
    listed = [f"vid{n:08d}" for n in range(60)]
    yt.playlist = [*listed, listed[0]]
    assert options().ytdlp.max_videos_per_run < len(listed)
    assert session.expand(LIST_URL) == listed
    assert "--playlist-end" not in yt.calls[0].args


def test_limit_becomes_playlist_end(session_kit, yt) -> None:
    yt.playlist = ["aaaaaaaaaaa"]
    session.expand(LIST_URL, limit=5)
    args = yt.calls[0].args
    assert args[args.index("--playlist-end") + 1] == "5"
    assert yt.calls[0].label == "expand"


def test_a_public_playlist_is_listed_anonymously(session_kit, yt) -> None:
    yt.playlist = ["aaaaaaaaaaa"]
    session.expand(LIST_URL, auth=COOKIE_FILE)
    assert yt.calls[0].cookies is None and session_kit.snapshots.taken == []


def test_a_members_playlist_is_listed_with_a_cookie_snapshot(session_kit, yt) -> None:
    yt.playlist = ["aaaaaaaaaaa"]
    assert session.expand(MEMBERS_URL, auth=COOKIE_FILE) == ["aaaaaaaaaaa"]
    assert yt.calls[0].cookies is not None and len(session_kit.snapshots.taken) == 1
    assert not Path(session_kit.snapshots.taken[0]).exists(), "the snapshot is deleted once listed"


def test_a_members_playlist_without_a_source_is_refused_not_listed_anonymously(session_kit, yt) -> None:
    with pytest.raises(AuthError) as refused:
        session.expand(MEMBERS_URL)
    assert refused.value.state is AuthState.NO_SOURCE and "auth use" in refused.value.fix
    assert yt.calls == []


def test_policy_never_lists_a_members_playlist_anonymously(session_kit, yt) -> None:
    yt.playlist = ["aaaaaaaaaaa"]
    session.expand(MEMBERS_URL, auth=COOKIE_FILE, options=options(policy="never"))
    assert yt.calls[0].cookies is None and session_kit.snapshots.taken == []


def test_a_cookie_channel_is_still_listed_anonymously(session_kit, yt) -> None:
    yt.playlist = ["aaaaaaaaaaa"]
    session.expand(CHANNEL_URL, auth=COOKIE_FILE, options=options(cookie_channels=("@syntheticchannel",)))
    assert yt.calls[0].cookies is None


def test_a_bot_check_writes_the_cooldown_and_raises(session_kit, yt) -> None:
    yt.failures[("expand", False)] = [session_kit.fail(Signal.BOT_CHECK)]
    with pytest.raises(BatchStopError) as stopped:
        session.expand(LIST_URL)
    assert stopped.value.signal is Signal.BOT_CHECK
    cooldown = read_cooldown()
    assert cooldown is not None and cooldown.signal is Signal.BOT_CHECK and cooldown.active(utcnow())
    with pytest.raises(BatchStopError):
        session.expand(LIST_URL)
    assert len(yt.calls) == 1, "the second call is refused before any request"


def test_a_stop_signal_on_a_successful_listing_also_raises(session_kit, yt) -> None:
    yt.playlist = ["aaaaaaaaaaa"]
    yt.secondary[("expand", False)] = (Signal.RATE_LIMITED,)
    with pytest.raises(BatchStopError) as stopped:
        session.expand(LIST_URL)
    assert stopped.value.signal is Signal.RATE_LIMITED
    cooldown = read_cooldown()
    assert cooldown is not None and cooldown.active(utcnow())


def test_a_failed_listing_raises_with_a_fix(session_kit, yt) -> None:
    yt.failures[("expand", False)] = [session_kit.fail(Signal.PRIVATE)]
    with pytest.raises(YtDlpError) as failed:
        session.expand(LIST_URL)
    assert failed.value.primary is Signal.PRIVATE and "Private" in failed.value.fix
    assert not paths.cooldown_file().exists()


def test_a_target_that_is_not_youtube_is_refused(session_kit, yt) -> None:
    with pytest.raises(ValueError, match="not a YouTube"):
        session.expand("https://example.com/watch?v=aaaaaaaaaaa")
    assert yt.calls == []
