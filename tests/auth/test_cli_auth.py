import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from ytverbatim import paths
from ytverbatim.auth import cookiefile, source
from ytverbatim.auth.cookiefile import CookieEntry
from ytverbatim.auth.cooldown import read_cooldown, start_cooldown
from ytverbatim.auth.source import BrowserSource, CookieFileSource
from ytverbatim.cli import auth as cli_auth
from ytverbatim.cli import main
from ytverbatim.ytdlp.types import Signal

JSON_KEYS = {
    "state",
    "ok",
    "fix",
    "detail",
    "source",
    "profile",
    "logged_in",
    "premium",
    "counts",
    "checked_at",
    "cached",
    "ytdlp_version",
    "ytdlp_age_days",
    "js_runtime",
    "warnings",
    "cooldown_until",
    "probe_skipped",
}


def _use_runner(monkeypatch: pytest.MonkeyPatch, runner: object) -> None:
    monkeypatch.setattr(cli_auth, "SubprocessRunner", lambda: runner)


def _invoke(*args: str):
    return CliRunner().invoke(main, ["auth", *args])


def test_auth_is_registered() -> None:
    assert "auth" in CliRunner().invoke(main, ["--help"]).output
    assert {"status", "doctor", "profiles", "use"} <= set(_invoke("--help").output.split())


def test_status_json_shape_and_cache(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    runner = kit.ok()
    _use_runner(monkeypatch, runner)
    first = _invoke("status", "--json", "--refresh")
    assert first.exit_code == 0, first.output
    payload = json.loads(first.output)
    assert set(payload) == JSON_KEYS
    assert payload["state"] == "ok" and payload["ok"] is True and payload["cached"] is False
    assert payload["source"] == "chrome:Default" and payload["profile"] == "Default"
    assert payload["premium"] is None

    second = json.loads(_invoke("status", "--json").output)
    assert second["cached"] is True
    assert len(runner.calls) == 1


def test_status_human_output(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    _use_runner(monkeypatch, kit.ok())
    result = _invoke("status")
    assert result.exit_code == 0
    first, *rest = result.output.splitlines()
    assert first.startswith("✓ Signed in · Premium: unknown · verified ")
    assert "  source: chrome:Default · profile Default" in rest
    assert "fakeLikedId" not in result.output


def test_status_auth_problem_exits_2(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    _use_runner(monkeypatch, kit.fail(Signal.LOGIN_REQUIRED))
    result = _invoke("status")
    assert result.exit_code == 2
    assert result.output.startswith("✗ logged_out: ")
    assert "  Fix: chrome profile 'Default' is not signed in" in result.output


def test_status_without_source(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_runner(monkeypatch, kit.ok())
    result = _invoke("status", "--json")
    assert result.exit_code == 2
    assert json.loads(result.output)["state"] == "no_source"


def test_status_with_broken_saved_source_exits_2(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    paths.auth_source_file().write_text("{", encoding="utf-8")
    _use_runner(monkeypatch, kit.ok())
    result = _invoke("status")
    assert result.exit_code == 2
    assert "no_source" in result.output and str(kit.home) not in result.output


def test_profiles_masks_emails_by_default() -> None:
    result = _invoke("profiles")
    assert result.exit_code == 0
    assert "a***@example.com" in result.output and "b***@example.com" in result.output
    assert "alice@" not in result.output and "bob@" not in result.output
    assert "Default" in result.output and "Work" in result.output
    assert "ytverbatim auth use chrome:<DIR>" in result.output


def test_profiles_show_emails() -> None:
    assert "alice@example.com" in _invoke("profiles", "--show-emails").output


def test_profiles_missing_browser_exits_2() -> None:
    result = _invoke("profiles", "--browser", "brave")
    assert result.exit_code == 2
    assert "browser_not_found" in result.output


def test_use_saves_a_valid_profile() -> None:
    result = _invoke("use", "chrome:Default")
    assert result.exit_code == 0, result.output
    assert source.resolve_source() == BrowserSource("chrome", "Default")


def test_use_rejects_an_unknown_profile_with_the_list() -> None:
    result = _invoke("use", "chrome:Nope")
    assert result.exit_code == 2
    assert "✗ profile_not_found" in result.output
    assert "'Default' (Person 1)" in result.output and "'Work'" in result.output
    assert not paths.auth_source_file().exists()


def test_use_bare_browser_refuses_to_guess() -> None:
    result = _invoke("use", "chrome")
    assert result.exit_code == 2
    assert "profile_not_found" in result.output


def test_use_bare_browser_saves_the_only_profile(kit) -> None:
    (kit.chrome_root / "Work" / "Cookies").unlink()
    assert _invoke("use", "chrome").exit_code == 0
    assert source.resolve_source() == BrowserSource("chrome", "Default")


def test_use_unknown_browser() -> None:
    result = _invoke("use", "firefox")
    assert result.exit_code == 2
    assert "browser_not_found" in result.output


def _cookie_file(path: Path, *, signed_in: bool) -> Path:
    rows = [CookieEntry(".youtube.com", "/", True, 4_000_000_000, "PREF", "x")]
    if signed_in:
        rows += [
            CookieEntry(".youtube.com", "/", True, 4_000_000_000, "LOGIN" + "_INFO", "secret-login"),
            CookieEntry(".youtube.com", "/", True, 4_000_000_000, "SAPISID", "secret-sapisid"),
        ]
    path.write_text(cookiefile.render(rows), encoding="utf-8")
    return path


def test_use_cookie_file_validation(home: Path) -> None:
    missing = _invoke("use", f"cookie-file:{home / 'missing.txt'}")
    assert missing.exit_code == 2 and "cookie_file_invalid" in missing.output
    assert str(home) not in missing.output

    signed_out = _cookie_file(home / "out.txt", signed_in=False)
    result = _invoke("use", f"cookie-file:{signed_out}")
    assert result.exit_code == 2 and "cookie_file_logged_out" in result.output

    good = _cookie_file(home / "good.txt", signed_in=True)
    result = _invoke("use", f"cookie-file:{good}")
    assert result.exit_code == 0, result.output
    assert "cookie-file:~/good.txt" in result.output
    assert "secret" not in result.output
    assert source.resolve_source() == CookieFileSource(good)


def test_use_none_clears() -> None:
    _invoke("use", "chrome:Default")
    result = _invoke("use", "none")
    assert result.exit_code == 0
    assert source.resolve_source() is None


def test_doctor_prints_one_line_per_check(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/opt/bin/{name}")
    _use_runner(monkeypatch, kit.ok())
    result = _invoke("doctor")
    assert result.exit_code == 2
    lines = result.output.splitlines()
    assert lines[0] == "✓ ffmpeg: ffmpeg found"
    assert any(line.startswith("✗ source: no_source") for line in lines)
    assert any(line.startswith("    Fix: No auth source set") for line in lines)


def test_status_during_a_cooldown_skips_the_probe_unless_ignored(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    start_cooldown(Signal.BOT_CHECK, datetime.now(UTC))
    runner = kit.ok()
    _use_runner(monkeypatch, runner)
    result = _invoke("status", "--refresh")
    assert result.exit_code == 2 and runner.calls == []
    assert result.output.startswith("✗ bot_check: cooldown active until ")
    assert "Wait until " in result.output and "an hour has passed" not in result.output
    assert "--ignore-cooldown" in result.output
    payload = json.loads(_invoke("status", "--json", "--refresh").output)
    assert payload["probe_skipped"] is True and payload["cooldown_until"] is not None

    allowed = _invoke("status", "--refresh", "--ignore-cooldown")
    assert allowed.exit_code == 0 and len(runner.calls) == 1


def test_status_probe_stop_signal_starts_a_cooldown(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    _use_runner(monkeypatch, kit.fail(Signal.BOT_CHECK))
    result = _invoke("status", "--refresh")
    assert result.exit_code == 2 and result.output.startswith("✗ bot_check: the :ytfav probe hit bot_check")
    cooldown = read_cooldown()
    assert cooldown is not None and cooldown.until > datetime.now(UTC) + timedelta(minutes=59)
    runner = kit.ok()
    _use_runner(monkeypatch, runner)
    assert _invoke("status", "--refresh").exit_code == 2 and runner.calls == []


def test_doctor_during_a_cooldown_skips_the_probe_unless_ignored(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: f"/opt/bin/{name}")
    source.save_source(BrowserSource("chrome", "Default"))
    start_cooldown(Signal.RATE_LIMITED, datetime.now(UTC))
    runner = kit.ok()
    _use_runner(monkeypatch, runner)
    result = _invoke("doctor")
    assert result.exit_code == 2 and runner.calls == []
    assert any(line.startswith("! login: skipped: cooldown active") for line in result.output.splitlines())
    assert any(line.startswith("✗ cooldown: rate_limited") for line in result.output.splitlines())
    _invoke("doctor", "--ignore-cooldown")
    assert len(runner.calls) == 1
