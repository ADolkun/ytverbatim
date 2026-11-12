import json
import time
from pathlib import Path
from typing import NoReturn

import click

from ytverbatim.auth import chromium, cookiefile, snapshot
from ytverbatim.auth.doctor import Status, exit_code, run_doctor
from ytverbatim.auth.fixes import fix_for
from ytverbatim.auth.preflight import AuthReport, fix_context, preflight
from ytverbatim.auth.source import BrowserSource, CookieFileSource, format_spec, parse_spec, resolve_source, save_source
from ytverbatim.auth.state import AuthState
from ytverbatim.errors import AuthError, YtverbatimError
from ytverbatim.ytdlp.runner import SubprocessRunner, redact

EXIT_AUTH = 2
EXIT_OTHER = 1
_MARKS = {Status.OK: "✓", Status.WARN: "!", Status.FAIL: "✗"}


def _fail_auth(exc: AuthError, **context: str | None) -> NoReturn:
    click.echo(f"✗ {exc.state.value}: {exc.detail}" if exc.detail else f"✗ {exc.state.value}", err=True)
    click.echo(f"  Fix: {fix_for(exc.state, **context)}", err=True)
    raise SystemExit(EXIT_AUTH)


def _fail_other(exc: YtverbatimError) -> NoReturn:
    click.echo(f"error: {redact(str(exc))}", err=True)
    raise SystemExit(EXIT_OTHER)


@click.group()
def auth() -> None:
    """Configure and check the browser login used for members-only and Premium videos."""


_IGNORE_COOLDOWN = click.option(
    "--ignore-cooldown", is_flag=True, help="Probe YouTube even though a bot or rate-limit cooldown is active."
)


@auth.command()
@click.option("--json", "as_json", is_flag=True, help="Print the report as one JSON object.")
@click.option("--refresh", is_flag=True, help="Ignore the one-hour cache and probe YouTube again.")
@_IGNORE_COOLDOWN
def status(*, as_json: bool, refresh: bool, ignore_cooldown: bool) -> None:
    """Show whether YouTube accepts the configured login (snapshot plus one :ytfav probe)."""
    try:
        source = resolve_source()
        report = preflight(source, SubprocessRunner(), refresh=refresh, ignore_cooldown=ignore_cooldown)
    except AuthError as exc:
        _fail_auth(exc)
    except YtverbatimError as exc:
        _fail_other(exc)
    if as_json:
        click.echo(json.dumps(report.to_json(), ensure_ascii=False))
    else:
        _print_report(report)
    if not report.ok:
        raise SystemExit(EXIT_AUTH)


def _print_report(report: AuthReport) -> None:
    if report.ok:
        click.echo(f"✓ {report.fix}{' (cached)' if report.cached else ''}")
    else:
        click.echo(f"✗ {report.state.value}: {report.detail}" if report.detail else f"✗ {report.state.value}")
        click.echo(f"  Fix: {report.fix}")
    if report.probe_skipped:
        click.echo("  (--ignore-cooldown probes anyway, at the account's risk.)")
    profile = f" · profile {report.profile}" if report.profile else ""
    click.echo(f"  source: {report.source}{profile}")
    if report.counts:
        click.echo(f"  cookies: {', '.join(f'{n} {domain}' for domain, n in report.counts.items())}")
    click.echo(f"  yt-dlp {report.ytdlp_version} · JS runtime: {report.js_runtime or 'none'}")
    for warning in report.warnings:
        context = {"version": report.ytdlp_version, "days": str(report.ytdlp_age_days)}
        click.echo(f"  ! {fix_for(AuthState(warning), **context)}")


@auth.command()
@_IGNORE_COOLDOWN
def doctor(*, ignore_cooldown: bool) -> None:
    """Check every dependency of authenticated fetching, one line each with its fix."""
    try:
        checks = run_doctor(SubprocessRunner(), ignore_cooldown=ignore_cooldown)
    except YtverbatimError as exc:
        _fail_other(exc)
    for check in checks:
        click.echo(f"{_MARKS[check.status]} {check.name}: {check.message}")
        if check.fix and check.status is not Status.OK:
            click.echo(f"    Fix: {check.fix}")
    raise SystemExit(exit_code(checks))


@auth.command()
@click.option(
    "--browser", type=click.Choice(chromium.SUPPORTED_BROWSERS), default="chrome", show_default=True, help="Browser."
)
@click.option("--show-emails", is_flag=True, help="Show full account emails instead of masked ones.")
def profiles(browser: str, *, show_emails: bool) -> None:
    """List browser profiles with their directory names, for `ytverbatim auth use BROWSER:DIR`."""
    try:
        found = chromium.list_profiles(browser, emails="full" if show_emails else "masked")
    except AuthError as exc:
        _fail_auth(exc, browser=browser)
    if not found:
        click.echo(f"no {browser} profiles found")
        raise SystemExit(EXIT_AUTH)
    width = max(len(p.dir) for p in found)
    for profile in found:
        notes = [
            "cookies" if profile.cookie_db is not None else "no cookies",
            *(["last used"] if profile.last_used else []),
            *([profile.email] if profile.email else []),
        ]
        click.echo(f"{profile.dir:<{width}}  {profile.name!r}  ({', '.join(notes)})")
    click.echo(f"Use one with: ytverbatim auth use {browser}:<DIR>")


@auth.command()
@click.argument("spec")
def use(spec: str) -> None:
    """Validate and save the auth source: BROWSER[:DIR], cookie-file:PATH, or none to clear it.

    Validation is offline: the profile must have a cookie database, or the file must be a readable,
    signed-in cookies.txt. Run `ytverbatim auth status` afterwards to check the login with YouTube.
    """
    try:
        source = parse_spec(spec)
        match source:
            case None:
                save_source(None)
                click.echo("✓ auth source cleared; videos are fetched anonymously")
                return
            case BrowserSource(browser=browser, profile=profile):
                # Saved with the directory resolved, so adding a second profile later cannot change accounts.
                source = BrowserSource(browser, chromium.cookie_db(browser, profile).profile)
            case CookieFileSource(path=path):
                _validate_cookie_file(path)
        save_source(source)
    except AuthError as exc:
        _fail_auth(exc, **_spec_context(spec))
    except YtverbatimError as exc:
        _fail_other(exc)
    click.echo(f"✓ auth source set to {redact(format_spec(source))}; check it with: ytverbatim auth status --refresh")


def _validate_cookie_file(path: Path) -> None:
    if not snapshot.is_logged_in(cookiefile.read_cookie_file(path), time.time()):
        raise AuthError(
            AuthState.COOKIE_FILE_LOGGED_OUT, "the cookie file has no signed-in YouTube session (or it expired)"
        )


def _spec_context(spec: str) -> dict[str, str | None]:
    browser, _, profile = spec.partition(":")
    if browser in chromium.SUPPORTED_BROWSERS:
        return fix_context(BrowserSource(browser, profile or None))
    return {}
