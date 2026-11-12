import json
import re
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ytverbatim import paths
from ytverbatim.auth import cookiefile, preflight
from ytverbatim.auth.cookiefile import CookieEntry
from ytverbatim.auth.cooldown import read_cooldown, start_cooldown
from ytverbatim.auth.fixes import FIXES, fix_for
from ytverbatim.auth.source import BrowserSource, CookieFileSource
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError, YtverbatimError
from ytverbatim.ytdlp import jsruntime, version
from ytverbatim.ytdlp.types import Signal

DOCS = Path(__file__).resolve().parents[2] / "docs" / "auth.md"
DEFAULT = BrowserSource("chrome", "Default")
NOW = 1_800_000_000.0


def _anchors() -> set[str]:
    headings = re.findall(r"^#+ (.+)$", DOCS.read_text(encoding="utf-8"), flags=re.M)
    return {re.sub(r"[^\w\- ]", "", h).strip().lower().replace(" ", "-") for h in headings}


def test_every_state_has_one_fix_and_a_docs_section() -> None:
    assert set(FIXES) == set(AuthState)
    anchors = _anchors()
    for state in AuthState:
        assert state.value in anchors, state
        rendered = fix_for(state)
        assert "{" not in rendered and "}" not in rendered, state


def test_fix_links_point_at_existing_anchors() -> None:
    anchors = _anchors()
    for state in AuthState:
        for anchor in re.findall(r"docs/auth\.md#([\w-]+)", fix_for(state)):
            assert anchor in anchors, (state, anchor)


def test_fix_context_fills_slots() -> None:
    assert "Wait until 12:00" in fix_for(AuthState.BOT_CHECK, until="12:00")
    assert "profile 'Work'" in fix_for(AuthState.LOGGED_OUT, browser="chrome", profile="Work")
    assert "yt-dlp 2026.01.01 is 99 days old" in fix_for(AuthState.YTDLP_STALE, version="2026.01.01", days="99")
    assert "Premium: yes" in fix_for(AuthState.OK, premium="yes")


def test_ok_probe_reports_signed_in(kit) -> None:
    runner = kit.ok()
    report = preflight.preflight(DEFAULT, runner, now=NOW)
    assert report.state is AuthState.OK and report.ok
    assert report.profile == "Default"
    assert report.logged_in is True
    assert report.premium is None
    assert report.fix.startswith("Signed in · Premium: unknown · verified ")
    assert report.counts == {"youtube.com": 12, "google.com": 30}
    assert not report.cached
    [call] = runner.calls
    assert call.args == list(preflight.PROBE_ARGS)
    assert call.cookies is not None
    assert "fakeLikedId" not in json.dumps(report.to_json())


def test_empty_liked_list_is_still_signed_in(kit) -> None:
    assert preflight.preflight(DEFAULT, kit.ok(stdout=""), now=NOW).state is AuthState.OK


def test_no_source(kit) -> None:
    report = preflight.preflight(None, kit.ok(), now=NOW)
    assert report.state is AuthState.NO_SOURCE
    assert report.source == "none"
    assert report.fix == fix_for(AuthState.NO_SOURCE)


def test_missing_browser(kit) -> None:
    runner = kit.ok()
    report = preflight.preflight(BrowserSource("brave", "Default"), runner, now=NOW)
    assert report.state is AuthState.BROWSER_NOT_FOUND
    assert "brave data not found" in report.fix
    assert runner.calls == [] and kit.snapshot.sources == []


def test_unknown_profile_lists_the_choices(kit) -> None:
    report = preflight.preflight(BrowserSource("chrome", "Nope"), kit.ok(), now=NOW)
    assert report.state is AuthState.PROFILE_NOT_FOUND
    assert "'Default' (Person 1)" in report.detail and "'Work'" in report.detail
    assert "chrome profile 'Nope'" in report.fix
    assert "example.com" not in report.detail


def test_bare_browser_with_two_profiles_refuses_to_guess(kit) -> None:
    report = preflight.preflight(BrowserSource("chrome"), kit.ok(), now=NOW)
    assert report.state is AuthState.PROFILE_NOT_FOUND
    assert kit.snapshot.sources == []


@pytest.mark.parametrize(
    "state",
    [AuthState.KEYCHAIN_DENIED, AuthState.LOGGED_OUT, AuthState.PROFILE_NOT_FOUND, AuthState.BROWSER_NOT_FOUND],
)
def test_snapshot_errors_become_states_without_a_probe(kit, state: AuthState) -> None:
    kit.snapshot.error = AuthError(state, "synthetic")
    runner = kit.ok()
    report = preflight.preflight(DEFAULT, runner, now=NOW)
    assert report.state is state
    assert report.detail == "synthetic"
    assert report.logged_in is None
    assert runner.calls == []


def test_keychain_fix_names_the_browser_item(kit) -> None:
    kit.snapshot.error = AuthError(AuthState.KEYCHAIN_DENIED)
    report = preflight.preflight(DEFAULT, kit.ok(), now=NOW)
    assert "'Chrome Safe Storage'" in report.fix


@pytest.mark.parametrize(
    ("exit_ok", "signals", "state"),
    [
        (True, (Signal.ROTATED,), AuthState.ROTATED),
        (False, (Signal.LOGIN_REQUIRED, Signal.ROTATED), AuthState.ROTATED),
        (False, (Signal.LOGIN_REQUIRED,), AuthState.LOGGED_OUT),
        (False, (Signal.BOT_CHECK,), AuthState.BOT_CHECK),
        (False, (Signal.CAPTCHA,), AuthState.CAPTCHA),
        (False, (Signal.IP_BLOCKED,), AuthState.IP_BLOCKED),
        (False, (Signal.RATE_LIMITED,), AuthState.RATE_LIMITED),
        (False, (Signal.HTTP_429,), AuthState.RATE_LIMITED),
        (False, (Signal.TRANSIENT,), AuthState.AUTH_UNCERTAIN),
        (False, (None,), AuthState.AUTH_UNCERTAIN),
    ],
)
def test_probe_outcomes(kit, exit_ok: bool, signals: tuple[Signal | None, ...], state: AuthState) -> None:
    primary, *secondary = signals
    runner = kit.ok(*signals) if exit_ok else kit.fail(primary, *secondary)
    report = preflight.preflight(DEFAULT, runner, now=NOW)
    assert report.state is state
    assert report.logged_in is True
    assert "fakeLikedId" not in report.detail
    assert not paths.preflight_cache_file().exists()


@pytest.mark.parametrize(
    ("exit_ok", "signals", "state"),
    [
        (False, (Signal.BOT_CHECK,), AuthState.BOT_CHECK),
        (False, (Signal.LOGIN_REQUIRED, Signal.IP_BLOCKED), AuthState.IP_BLOCKED),
        (False, (None, Signal.HTTP_429), AuthState.RATE_LIMITED),
        (True, (Signal.CAPTCHA,), AuthState.CAPTCHA),
    ],
)
def test_a_stop_signal_on_the_probe_starts_a_cooldown(
    kit, exit_ok: bool, signals: tuple[Signal | None, ...], state: AuthState
) -> None:
    primary, *secondary = signals
    runner = kit.ok(*[s for s in signals if s is not None]) if exit_ok else kit.fail(primary, *secondary)
    report = preflight.preflight(DEFAULT, runner, now=NOW)
    assert report.state is state and not report.probe_skipped
    cooldown = read_cooldown()
    assert cooldown is not None and report.cooldown == cooldown
    assert cooldown.until == datetime.fromtimestamp(NOW, UTC) + timedelta(hours=1)
    assert "Wait until " in report.fix and "an hour has passed" not in report.fix


def test_active_cooldown_skips_the_probe_unless_ignored(kit) -> None:
    started = start_cooldown(Signal.BOT_CHECK, datetime.fromtimestamp(NOW, UTC))
    runner = kit.ok()
    report = preflight.preflight(DEFAULT, runner, refresh=True, now=NOW + 60)
    assert runner.calls == [] and report.probe_skipped and report.cooldown == started
    assert report.state is AuthState.BOT_CHECK and report.logged_in is True
    assert preflight.preflight(DEFAULT, runner, refresh=True, now=NOW + 60, ignore_cooldown=True).ok
    assert len(runner.calls) == 1
    assert preflight.preflight(DEFAULT, kit.ok(), refresh=True, now=NOW + 7200).ok


def test_a_stop_during_an_ignored_cooldown_extends_it(kit) -> None:
    start_cooldown(Signal.BOT_CHECK, datetime.fromtimestamp(NOW, UTC))
    report = preflight.preflight(DEFAULT, kit.fail(Signal.BOT_CHECK), now=NOW + 60, ignore_cooldown=True)
    cooldown = read_cooldown()
    assert cooldown is not None and cooldown.hours == 2 and report.cooldown == cooldown


def test_unreadable_cooldown_refuses_the_probe(kit) -> None:
    paths.cooldown_file().write_text("{", encoding="utf-8")
    runner = kit.ok()
    with pytest.raises(YtverbatimError, match=r"cooldown\.json is unreadable"):
        preflight.preflight(DEFAULT, runner, refresh=True, now=NOW)
    assert runner.calls == []
    assert preflight.preflight(DEFAULT, runner, refresh=True, now=NOW, ignore_cooldown=True).ok


def test_cookie_file_rejected_by_youtube_is_a_file_state(kit) -> None:
    report = preflight.preflight(CookieFileSource(kit.home / "c.txt"), kit.fail(Signal.LOGIN_REQUIRED), now=NOW)
    assert report.state is AuthState.COOKIE_FILE_LOGGED_OUT


def _write_cookie_file(path: Path, *, signed_in: bool) -> Path:
    rows = [CookieEntry(".youtube.com", "/", True, 4_000_000_000, "PREF", "x")]
    if signed_in:
        rows += [
            CookieEntry(".youtube.com", "/", True, 4_000_000_000, "LOGIN" + "_INFO", "secret-login"),
            CookieEntry(".youtube.com", "/", True, 4_000_000_000, "SAPISID", "secret-sapisid"),
        ]
    path.write_text(cookiefile.render(rows), encoding="utf-8")
    return path


def test_real_cookie_file_snapshot_states(home: Path, fakes) -> None:
    bad = home / "bad.txt"
    bad.write_text("{}", encoding="utf-8")
    assert preflight.preflight(CookieFileSource(bad), fakes.ok(), now=NOW).state is AuthState.COOKIE_FILE_INVALID
    out = _write_cookie_file(home / "out.txt", signed_in=False)
    assert preflight.preflight(CookieFileSource(out), fakes.ok(), now=NOW).state is AuthState.COOKIE_FILE_LOGGED_OUT
    good = _write_cookie_file(home / "good.txt", signed_in=True)
    runner = fakes.ok()
    report = preflight.preflight(CookieFileSource(good), runner, now=NOW)
    assert report.state is AuthState.OK
    assert report.source == "cookie-file:~/good.txt"
    assert runner.calls[0].cookies is not None and runner.calls[0].cookies != good


def test_tooling_warnings(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jsruntime, "detect", lambda: jsruntime.JsRuntime(None, (), "none"))
    monkeypatch.setattr(version, "installed", lambda: version.YtDlpVersion("2020.01.01", version.date(2020, 1, 1)))
    report = preflight.preflight(DEFAULT, kit.ok(Signal.PO_TOKEN_GVS), now=NOW)
    assert report.state is AuthState.OK
    assert report.warnings == ("po_token_required", "js_runtime_missing", "ytdlp_stale")


def test_ok_result_is_cached_for_an_hour(kit) -> None:
    first = kit.ok()
    preflight.preflight(DEFAULT, first, now=NOW)
    second = kit.ok()
    cached = preflight.preflight(DEFAULT, second, now=NOW + preflight.TTL_S - 1)
    assert cached.state is AuthState.OK and cached.cached
    assert cached.counts == {"youtube.com": 12, "google.com": 30}
    assert cached.checked_at.timestamp() == NOW
    assert second.calls == [] and len(kit.snapshot.sources) == 1

    expired = kit.ok()
    assert not preflight.preflight(DEFAULT, expired, now=NOW + preflight.TTL_S).cached
    assert len(expired.calls) == 1


def test_refresh_skips_the_cache(kit) -> None:
    preflight.preflight(DEFAULT, kit.ok(), now=NOW)
    runner = kit.fail(Signal.LOGIN_REQUIRED)
    report = preflight.preflight(DEFAULT, runner, refresh=True, now=NOW + 1)
    assert report.state is AuthState.LOGGED_OUT
    assert len(runner.calls) == 1


def test_cache_is_keyed_by_source_and_profile(kit) -> None:
    preflight.preflight(DEFAULT, kit.ok(), now=NOW)
    other = kit.ok()
    assert not preflight.preflight(BrowserSource("chrome", "Work"), other, now=NOW + 1).cached
    assert len(other.calls) == 1
    assert preflight.cache_key(DEFAULT, "Default") != preflight.cache_key(BrowserSource("chrome", "Work"), "Work")
    assert preflight.cache_key(BrowserSource("chrome"), "Default") != preflight.cache_key(DEFAULT, "Default")


def test_failures_are_not_cached(kit) -> None:
    preflight.preflight(DEFAULT, kit.fail(Signal.LOGIN_REQUIRED), now=NOW)
    runner = kit.ok()
    assert not preflight.preflight(DEFAULT, runner, now=NOW + 1).cached
    assert len(runner.calls) == 1


def test_invalidate_drops_the_cached_login(kit) -> None:
    preflight.preflight(DEFAULT, kit.ok(), now=NOW)
    preflight.invalidate(DEFAULT)
    runner = kit.ok()
    assert not preflight.preflight(DEFAULT, runner, now=NOW + 1).cached


def test_recorded_premium_is_reported_and_outlives_the_ttl(kit) -> None:
    preflight.record_premium(DEFAULT, True, now=NOW)
    report = preflight.preflight(DEFAULT, kit.ok(), now=NOW)
    assert report.premium is True
    assert "Premium: yes" in report.fix
    later = preflight.preflight(DEFAULT, kit.ok(), now=NOW + 10 * preflight.TTL_S)
    assert later.premium is True and not later.cached


def test_corrupt_cache_costs_one_probe(kit) -> None:
    paths.preflight_cache_file().write_text("not json", encoding="utf-8")
    runner = kit.ok()
    assert preflight.preflight(DEFAULT, runner, now=NOW).state is AuthState.OK
    assert len(runner.calls) == 1


def test_cache_file_holds_no_secrets_or_home_paths(home: Path, fakes) -> None:
    good = _write_cookie_file(home / "good.txt", signed_in=True)
    preflight.preflight(CookieFileSource(good), fakes.ok(), now=NOW)
    preflight.record_premium(CookieFileSource(good), False, now=NOW)
    cache = paths.preflight_cache_file()
    text = cache.read_text(encoding="utf-8")
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600
    for forbidden in ("secret-login", "secret-sapisid", "example.com", str(home), "good.txt", "fakeLikedId", "~"):
        assert forbidden not in text
    entries = json.loads(text)["entries"]
    assert len(entries) == 1
    [entry] = entries.values()
    assert set(entry) == {"result", "premium"}
