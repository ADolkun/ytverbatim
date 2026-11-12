import json
import shutil
from datetime import UTC, datetime, timedelta

import pytest

from ytverbatim import paths
from ytverbatim.auth import doctor, source
from ytverbatim.auth.doctor import Status
from ytverbatim.auth.source import BrowserSource
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError
from ytverbatim.ytdlp import jsruntime, version
from ytverbatim.ytdlp.types import Signal

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: f"/opt/bin/{name}")


def _by_name(checks: list[doctor.Check]) -> dict[str, doctor.Check]:
    return {c.name: c for c in checks}


def test_all_good(kit) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    runner = kit.ok()
    checks = doctor.run_doctor(runner, now=NOW)
    assert [c.name for c in checks] == [
        "ffmpeg",
        "ffprobe",
        "js runtime",
        "yt-dlp",
        "source",
        "snapshot",
        "login",
        "cooldown",
    ]
    assert all(c.status is Status.OK for c in checks), checks
    assert doctor.exit_code(checks) == 0
    assert len(runner.calls) == 1


def test_doctor_always_probes_afresh(kit) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    doctor.run_doctor(kit.ok(), now=NOW)
    runner = kit.ok()
    doctor.run_doctor(runner, now=NOW)
    assert len(runner.calls) == 1


def test_missing_tools_are_other_failures(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "ffprobe" else f"/opt/bin/{name}")
    monkeypatch.setattr(jsruntime, "detect", lambda: jsruntime.JsRuntime(None, (), "no runtime"))
    checks = _by_name(doctor.run_doctor(kit.ok(), now=NOW))
    assert checks["ffprobe"].status is Status.FAIL and "brew install ffmpeg" in checks["ffprobe"].fix
    assert checks["js runtime"].status is Status.FAIL and "Deno" in checks["js runtime"].fix
    assert doctor.exit_code(list(checks.values())) == 1


def test_node_runtime_notes_deno_once(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jsruntime, "detect", lambda: jsruntime.JsRuntime("node", ("--js-runtimes", "node"), None))
    checks = doctor.run_doctor(kit.ok(), now=NOW)
    js = _by_name(checks)["js runtime"]
    assert js.status is Status.OK and "Deno's sandbox is stricter" in js.message
    assert sum("Deno" in c.message for c in checks) == 1


def test_stale_ytdlp_warns(kit, monkeypatch: pytest.MonkeyPatch) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    released = datetime.now().date() - timedelta(days=60)
    monkeypatch.setattr(version, "installed", lambda: version.YtDlpVersion("2020.01.01", released))
    checks = doctor.run_doctor(kit.ok(), now=NOW)
    ytdlp = _by_name(checks)["yt-dlp"]
    assert ytdlp.status is Status.WARN
    assert (
        ytdlp.fix == "yt-dlp 2020.01.01 is 60 days old and YouTube changes often. Upgrade yt-dlp in this environment."
    )
    assert doctor.exit_code(checks) == 0


def test_no_source_is_an_auth_failure(kit) -> None:
    checks = _by_name(doctor.run_doctor(kit.ok(), now=NOW))
    assert checks["source"].status is Status.FAIL
    assert checks["source"].message.startswith("no_source")
    assert "snapshot" not in checks and "login" not in checks
    assert doctor.exit_code(list(checks.values())) == 2


def test_bad_profile_fails_the_source_line(kit) -> None:
    source.save_source(BrowserSource("chrome", "Nope"))
    checks = _by_name(doctor.run_doctor(kit.ok(), now=NOW))
    assert checks["source"].status is Status.FAIL
    assert checks["source"].message.startswith("profile_not_found")


def test_snapshot_failure_skips_login(kit) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    kit.snapshot.error = AuthError(AuthState.KEYCHAIN_DENIED, "refused")
    runner = kit.ok()
    checks = _by_name(doctor.run_doctor(runner, now=NOW))
    assert checks["source"].status is Status.OK
    assert checks["snapshot"].status is Status.FAIL and "Always Allow" in checks["snapshot"].fix
    assert checks["login"].status is Status.WARN
    assert runner.calls == []


def test_login_failure(kit) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    checks = _by_name(doctor.run_doctor(kit.fail(Signal.LOGIN_REQUIRED), now=NOW))
    assert checks["snapshot"].status is Status.OK
    assert checks["login"].status is Status.FAIL and checks["login"].message.startswith("logged_out")


def test_po_token_warning(kit) -> None:
    source.save_source(BrowserSource("chrome", "Default"))
    checks = _by_name(doctor.run_doctor(kit.ok(Signal.PO_TOKEN_SUBS), now=NOW))
    assert checks["po token"].status is Status.WARN and "docs/auth.md#po-token" in checks["po token"].fix


@pytest.mark.parametrize(
    ("payload", "status", "state"),
    [
        ({"until": (NOW + timedelta(hours=1)).isoformat(), "signal": "rate_limited"}, Status.FAIL, "rate_limited"),
        ({"until": (NOW + timedelta(hours=1)).timestamp(), "signal": "bot_check"}, Status.FAIL, "bot_check"),
        ({"until": (NOW - timedelta(hours=1)).isoformat(), "signal": "bot_check"}, Status.OK, None),
        ({"signal": "bot_check"}, Status.FAIL, None),
    ],
)
def test_cooldown(kit, payload: dict[str, object], status: Status, state: str | None) -> None:
    paths.cooldown_file().write_text(json.dumps(payload), encoding="utf-8")
    cooldown = _by_name(doctor.run_doctor(kit.ok(), now=NOW))["cooldown"]
    assert cooldown.status is status
    if state is not None:
        assert cooldown.message.startswith(state)
        assert "Wait until " in cooldown.fix and "an hour has passed" not in cooldown.fix
