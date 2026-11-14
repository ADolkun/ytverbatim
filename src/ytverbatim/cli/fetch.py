from pathlib import Path
from typing import NoReturn

import click

from ytverbatim import config, ids, paths
from ytverbatim.acquire.session import (
    REPORT_FILE,
    BatchReport,
    FetchOptions,
    Policy,
    VideoOutcome,
    VideoStatus,
    fetch,
    stop_fix,
)
from ytverbatim.auth.fixes import fix_for
from ytverbatim.auth.preflight import fix_context
from ytverbatim.auth.source import resolve_source
from ytverbatim.errors import AuthError, BatchStopError, YtverbatimError
from ytverbatim.ytdlp.runner import SubprocessRunner, redact

EXIT_FAILED = 1
EXIT_AUTH = 2
EXIT_STOPPED = 3


@click.command(name="fetch")
@click.argument("urls", nargs=-1, required=True)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), help="Output root (one folder per video).")
@click.option("--auth", "auth_spec", metavar="SPEC", help="Auth source for this run: BROWSER[:DIR], cookie-file:PATH.")
@click.option("--policy", type=click.Choice(["auto", "always", "never"]), help="When to use cookies (default auto).")
@click.option("--captions/--no-captions", default=True, show_default=True, help="Download caption tracks.")
@click.option("--audio/--no-audio", default=True, show_default=True, help="Download audio and convert it to WAV.")
@click.option("--max-videos", type=click.IntRange(min=1), help="Cap on videos per run (default 50).")
@click.option("--ignore-cooldown", is_flag=True, help="Start even though a bot or rate-limit cooldown is active.")
def fetch_command(
    urls: tuple[str, ...],
    out: Path | None,
    auth_spec: str | None,
    policy: Policy | None,
    *,
    captions: bool,
    audio: bool,
    max_videos: int | None,
    ignore_cooldown: bool,
) -> None:
    """Fetch metadata, captions and audio for videos, playlists or channels.

    Videos are fetched anonymously first; the configured browser login is used only when YouTube asks
    for it. Exit codes: 0 ok, 1 some videos failed, 2 auth error, 3 batch stopped (cooldown).
    """
    bad = [url for url in urls if ids.classify(url) is None]
    if bad:
        click.echo(f"error: not a YouTube video, playlist or channel: {', '.join(bad)}", err=True)
        raise SystemExit(EXIT_FAILED)
    source = None
    try:
        source = resolve_source(auth_spec)
        options = config.load_options(
            overrides={"auth": {"policy": policy}, "ytdlp": {"max_videos_per_run": max_videos}}
        )
        root = out if out is not None else paths.default_output_root()
        report = fetch(
            urls,
            source=source,
            out=root,
            options=FetchOptions(
                policy=options.auth.policy,
                captions=captions,
                audio=audio,
                cookie_channels=options.auth.cookie_channels,
                max_videos=options.ytdlp.max_videos_per_run,
                ignore_cooldown=ignore_cooldown,
                creator_note=options.ytdlp.creator_note,
                video=options.ytdlp.video,
            ),
            runner_factory=lambda sleep: SubprocessRunner(sleep_args=sleep),
            on_video=_print_outcome,
        )
    except BatchStopError as exc:
        _stopped(exc)
    except AuthError as exc:
        click.echo(f"✗ {exc.state.value}: {redact(exc.detail)}" if exc.detail else f"✗ {exc.state.value}", err=True)
        click.echo(f"  Fix: {fix_for(exc.state, **fix_context(source))}", err=True)
        raise SystemExit(EXIT_AUTH) from None
    except YtverbatimError as exc:
        click.echo(f"error: {redact(str(exc))}", err=True)
        raise SystemExit(EXIT_FAILED) from None
    _print_summary(report, root)
    raise SystemExit(report.exit_code)


def _print_outcome(outcome: VideoOutcome) -> None:
    if outcome.status is VideoStatus.OK:
        click.echo(f"✓ {outcome.video_id}  {' · '.join(_facts(outcome))}")
        return
    detail = f": {outcome.detail}" if outcome.detail else ""
    click.echo(f"✗ {outcome.video_id}  {outcome.error}{detail}")
    if outcome.fix:
        click.echo(f"    Fix: {outcome.fix}")


def _facts(outcome: VideoOutcome) -> list[str]:
    access = outcome.access
    facts: list[str] = []
    if access is not None:
        if access.cookies_used:
            premium = "unknown" if access.premium is None else "yes" if access.premium else "no"
            facts.append(f"cookies ({access.auth_source}) · premium {premium}")
            if access.member:
                facts.append("member")
        else:
            facts.append("anonymous")
    facts.append(f"{outcome.chapters} chapters")
    if outcome.captions:
        facts.append(f"captions {', '.join(outcome.captions)}")
    if outcome.audio_s is not None:
        facts.append(f"audio {outcome.audio_s:.0f}s")
    if outcome.reused:
        facts.append(f"reused {', '.join(outcome.reused)}")
    return facts


def _print_summary(report: BatchReport, root: Path) -> None:
    counts = {status: sum(v.status is status for v in report.videos) for status in VideoStatus}
    parts = [f"{counts[VideoStatus.OK]} ok", f"{counts[VideoStatus.ERROR]} failed"]
    if counts[VideoStatus.NOT_ATTEMPTED]:
        parts.append(f"{counts[VideoStatus.NOT_ATTEMPTED]} not attempted")
    click.echo(f"{', '.join(parts)} · report: {redact(str(root / REPORT_FILE))}")
    if report.capped:
        click.echo("! the per-run video cap was reached; raise it with --max-videos to fetch more")
    if report.stopped is not None:
        click.echo(f"■ batch stopped by {report.stopped.signal.value}", err=True)
        click.echo(f"  Fix: {stop_fix(report.stopped.signal, report.stopped.until)}", err=True)


def _stopped(exc: BatchStopError) -> NoReturn:
    click.echo(f"■ {exc}", err=True)
    click.echo(f"  Fix: {stop_fix(exc.signal, exc.until)}", err=True)
    click.echo("  (--ignore-cooldown starts anyway, at the account's risk.)", err=True)
    raise SystemExit(EXIT_STOPPED)
