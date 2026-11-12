import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from ytverbatim import paths
from ytverbatim.auth.fixes import fix_for
from ytverbatim.auth.preflight import YTDLP_STALE_DAYS, preflight
from ytverbatim.auth.source import resolve_source
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError, DependencyError
from ytverbatim.ytdlp import jsruntime, version
from ytverbatim.ytdlp.types import Signal, YtDlpRunner

_COOLDOWN_STATES = {
    Signal.BOT_CHECK.value: AuthState.BOT_CHECK,
    Signal.CAPTCHA.value: AuthState.CAPTCHA,
    Signal.IP_BLOCKED.value: AuthState.IP_BLOCKED,
    Signal.RATE_LIMITED.value: AuthState.RATE_LIMITED,
    Signal.HTTP_429.value: AuthState.RATE_LIMITED,
}


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    message: str
    fix: str = ""
    # A failure here is an auth problem (exit 2) rather than a missing tool (exit 1).
    auth: bool = False


def run_doctor(runner: YtDlpRunner, *, now: datetime | None = None) -> list[Check]:
    """Run every check in order; the login checks always probe YouTube afresh instead of trusting the cache."""
    checks = [_binary("ffmpeg"), _binary("ffprobe"), _js_runtime(), _ytdlp()]
    checks += _auth_checks(runner)
    checks.append(_cooldown(now or datetime.now(UTC)))
    return checks


def _binary(name: str) -> Check:
    if shutil.which(name):
        return Check(name, Status.OK, f"{name} found")
    return Check(name, Status.FAIL, f"{name} not found on PATH", f"Install it: brew install ffmpeg (provides {name}).")


def _js_runtime() -> Check:
    found = jsruntime.detect()
    if found.name is None:
        return Check("js runtime", Status.FAIL, found.finding or "none found", fix_for(AuthState.JS_RUNTIME_MISSING))
    if found.name == "node":
        # yt-dlp's node sandbox can still read local files such as the cookie database (yt-dlp #17799).
        return Check("js runtime", Status.OK, "node (Deno's sandbox is stricter; consider: brew install deno)")
    return Check("js runtime", Status.OK, found.name)


def _ytdlp() -> Check:
    try:
        installed = version.installed()
    except DependencyError as exc:
        return Check("yt-dlp", Status.FAIL, str(exc), "Reinstall ytverbatim in this environment.")
    age = installed.age_days()
    if age > YTDLP_STALE_DAYS:
        fix = fix_for(AuthState.YTDLP_STALE, version=installed.version, days=str(age))
        return Check("yt-dlp", Status.WARN, f"{installed.version} is {age} days old", fix)
    return Check("yt-dlp", Status.OK, f"{installed.version} ({age} days old)")


def _auth_checks(runner: YtDlpRunner) -> list[Check]:
    try:
        source = resolve_source()
    except AuthError as exc:
        return [_failed("source", exc.state, exc.detail, fix_for(exc.state))]
    if source is None:
        return [_failed("source", AuthState.NO_SOURCE, "no auth source configured", fix_for(AuthState.NO_SOURCE))]
    report = preflight(source, runner, refresh=True)
    checks: list[Check] = []
    if report.state in (AuthState.BROWSER_NOT_FOUND, AuthState.PROFILE_NOT_FOUND):
        return [_failed("source", report.state, report.detail, report.fix)]
    profile = f" · profile {report.profile}" if report.profile else ""
    checks.append(Check("source", Status.OK, f"{report.source}{profile}", auth=True))
    if report.logged_in is None:
        checks.append(_failed("snapshot", report.state, report.detail, report.fix))
        checks.append(Check("login", Status.WARN, "skipped: no usable cookie snapshot", auth=True))
        return checks
    counts = ", ".join(f"{n} {domain}" for domain, n in report.counts.items())
    checks.append(Check("snapshot", Status.OK, f"signed-in cookies found ({counts})", auth=True))
    if report.ok:
        checks.append(Check("login", Status.OK, "YouTube accepted the cookies (:ytfav)", auth=True))
    else:
        checks.append(_failed("login", report.state, report.detail, report.fix))
    if AuthState.PO_TOKEN_REQUIRED.value in report.warnings:
        po_fix = fix_for(AuthState.PO_TOKEN_REQUIRED)
        checks.append(Check("po token", Status.WARN, "YouTube asked for a PO token", po_fix, auth=True))
    return checks


def _failed(name: str, state: AuthState, detail: str, fix: str) -> Check:
    return Check(name, Status.FAIL, f"{state.value}: {detail}" if detail else state.value, fix, auth=True)


def _cooldown(now: datetime) -> Check:
    """Read the cooldown the fetch session writes after a bot or rate-limit signal."""
    path = paths.cooldown_file()
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Check("cooldown", Status.OK, "none active", auth=True)
    except (OSError, ValueError) as exc:
        return Check("cooldown", Status.FAIL, f"cooldown.json is unreadable ({type(exc).__name__})", auth=True)
    fields = raw if isinstance(raw, dict) else {}
    until = _parse_until(fields.get("until"))
    if until is None:
        return Check("cooldown", Status.FAIL, "cooldown.json has no valid 'until' time", auth=True)
    if until <= now:
        return Check("cooldown", Status.OK, "none active", auth=True)
    signal = fields.get("signal")
    state = _COOLDOWN_STATES.get(signal, AuthState.BOT_CHECK) if isinstance(signal, str) else AuthState.BOT_CHECK
    shown = until.astimezone().strftime("%Y-%m-%d %H:%M")
    return _failed("cooldown", state, f"active until {shown}", fix_for(state, until=shown))


def _parse_until(value: object) -> datetime | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return datetime.fromtimestamp(value, UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None


def exit_code(checks: list[Check]) -> int:
    failed = [c for c in checks if c.status is Status.FAIL]
    if any(c.auth for c in failed):
        return 2
    return 1 if failed else 0
