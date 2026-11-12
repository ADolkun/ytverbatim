from datetime import UTC, datetime

from ytverbatim.auth.state import AuthState
from ytverbatim.errors import (
    AuthError,
    BatchStopError,
    ConfigError,
    UnavailableError,
    YtDlpError,
    YtverbatimError,
    stderr_preview,
)
from ytverbatim.ytdlp.types import Signal


def test_hierarchy_and_attributes() -> None:
    until = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    stop = BatchStopError(Signal.BOT_CHECK, until)
    assert isinstance(stop, YtverbatimError)
    assert (stop.signal, stop.until) == (Signal.BOT_CHECK, until)
    assert "bot_check" in str(stop)
    assert "2026-01-02T03:04:05" in str(stop)
    assert str(BatchStopError(Signal.RATE_LIMITED, None)) == "batch stopped by rate_limited"

    auth = AuthError(AuthState.LOGGED_OUT, "sign in")
    assert auth.state is AuthState.LOGGED_OUT
    assert str(auth) == "logged_out: sign in"
    assert str(AuthError(AuthState.NO_SOURCE)) == "no_source"

    err = YtDlpError(Signal.MEMBERS_ONLY, (Signal.N_CHALLENGE,), "boom", 1)
    assert (err.primary, err.secondary, err.preview, err.returncode) == (
        Signal.MEMBERS_ONLY,
        (Signal.N_CHALLENGE,),
        "boom",
        1,
    )
    assert "members_only" in str(err)
    assert "unclassified" in str(YtDlpError(None, (), "x", 2))

    assert UnavailableError("geo").kind == "geo"
    assert issubclass(ConfigError, YtverbatimError)


def test_stderr_preview_keeps_both_ends() -> None:
    assert stderr_preview("  short  ") == "short"
    long = "A" * 1000 + "M" * 500 + "Z" * 1000
    preview = stderr_preview(long)
    assert preview.startswith("A" * 800)
    assert preview.endswith("Z" * 800)
    assert "[truncated 900 chars]" in preview
