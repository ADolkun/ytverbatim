from pathlib import Path
from typing import NoReturn

import click

from ytverbatim import config, ids, paths, pipeline
from ytverbatim.acquire.session import REPORT_FILE, BatchReport, VideoStatus, stop_fix
from ytverbatim.auth.fixes import fix_for
from ytverbatim.auth.preflight import fix_context
from ytverbatim.auth.source import resolve_source
from ytverbatim.cli.fetch import EXIT_AUTH, EXIT_FAILED, EXIT_STOPPED
from ytverbatim.errors import AuthError, BatchStopError, YtverbatimError
from ytverbatim.ytdlp.runner import redact


@click.command(name="run")
@click.argument("urls", nargs=-1, required=True)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), help="Output root (one folder per video).")
@click.option("--auth", "auth_spec", metavar="SPEC", help="Auth source for this run: BROWSER[:DIR], cookie-file:PATH.")
@click.option("--engine", "engine", metavar="ID", help="ASR hypothesis to run (default w-prod).")
@click.option("--captions/--no-captions", default=True, show_default=True, help="Download caption tracks.")
@click.option("--force", is_flag=True, help="Redo videos that already have a context.json.")
@click.option("--ignore-cooldown", is_flag=True, help="Start even though a bot or rate-limit cooldown is active.")
def run_command(
    urls: tuple[str, ...],
    out: Path | None,
    auth_spec: str | None,
    engine: str | None,
    *,
    captions: bool,
    force: bool,
    ignore_cooldown: bool,
) -> None:
    """Turn videos, playlists or channels into context.json and context.md.

    Fetches each video, transcribes it locally and writes both files next to the downloaded media. A video
    that already has a context.json is skipped unless --force is given.
    Exit codes: 0 ok, 1 some videos failed, 2 auth error, 3 batch stopped (cooldown).
    """
    bad = [url for url in urls if ids.classify(url) is None]
    if bad:
        click.echo(f"error: not a YouTube video, playlist or channel: {', '.join(bad)}", err=True)
        raise SystemExit(EXIT_FAILED)
    source = None
    try:
        source = resolve_source(auth_spec)
        options = config.load_options(overrides={"asr": {"engine": engine}})
        root = out if out is not None else paths.default_output_root()
        report = pipeline.process_many(
            urls,
            options,
            out=root,
            source=source,
            captions=captions,
            force=force,
            ignore_cooldown=ignore_cooldown,
            progress=_print_progress,
            on_video=_print_result,
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


def _print_progress(event: pipeline.ProgressEvent) -> None:
    detail = f" {event.detail}" if event.detail else ""
    click.echo(f"… {event.video_id}  {event.stage}{detail}", err=True)


def _print_result(result: pipeline.VideoResult) -> None:
    outcome = result.outcome
    if outcome.status is not VideoStatus.OK:
        detail = f": {outcome.detail}" if outcome.detail else ""
        click.echo(f"✗ {outcome.video_id}  {outcome.error}{detail}")
        if outcome.fix:
            click.echo(f"    Fix: {outcome.fix}")
        return
    document, markdown = result.document, result.markdown
    if document is None or markdown is None:
        return
    flags = sum(document.quality.flags_by_kind.values())
    facts = [
        f"{len(document.segments)} segments",
        f"{len(document.paragraphs)} paragraphs",
        f"{len(document.chapters)} chapters",
        f"{flags} chunk flags",
    ]
    if outcome.reused and "context" in outcome.reused:
        facts.append("already done")
    else:
        facts.append(f"{result.elapsed_s:.0f}s")
    click.echo(f"✓ {outcome.video_id}  {redact(str(markdown))}  ({' · '.join(facts)})")


def _print_summary(report: BatchReport, root: Path) -> None:
    ok = sum(v.status is VideoStatus.OK for v in report.videos)
    failed = sum(v.status is VideoStatus.ERROR for v in report.videos)
    skipped = sum(v.status is VideoStatus.NOT_ATTEMPTED for v in report.videos)
    parts = [f"{ok} ok", f"{failed} failed"]
    if skipped:
        parts.append(f"{skipped} not attempted")
    click.echo(f"{', '.join(parts)} · report: {redact(str(root / REPORT_FILE))}")
    if report.capped:
        click.echo("! the per-run video cap was reached; raise ytdlp.max_videos_per_run to process more")
    if report.stopped is not None:
        click.echo(f"■ batch stopped by {report.stopped.signal.value}", err=True)
        click.echo(f"  Fix: {stop_fix(report.stopped.signal, report.stopped.until)}", err=True)


def _stopped(exc: BatchStopError) -> NoReturn:
    click.echo(f"■ {exc}", err=True)
    click.echo(f"  Fix: {stop_fix(exc.signal, exc.until)}", err=True)
    click.echo("  (--ignore-cooldown starts anyway, at the account's risk.)", err=True)
    raise SystemExit(EXIT_STOPPED)
