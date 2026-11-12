from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import click

from ytverbatim import config, ids, paths, pipeline
from ytverbatim.acquire.session import REPORT_FILE, BatchReport, VideoStatus, stop_fix
from ytverbatim.auth.fixes import fix_for
from ytverbatim.auth.preflight import fix_context
from ytverbatim.auth.source import resolve_source
from ytverbatim.cli.fetch import EXIT_AUTH, EXIT_FAILED, EXIT_STOPPED
from ytverbatim.contract import LLMUsage
from ytverbatim.errors import AuthError, BatchStopError, EngineMemoryError, YtverbatimError
from ytverbatim.llm import apply_spec
from ytverbatim.ytdlp.runner import redact


@click.command(name="run")
@click.argument("urls", nargs=-1, required=True)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), help="Output root (one folder per video).")
@click.option("--auth", "auth_spec", metavar="SPEC", help="Auth source for this run: BROWSER[:DIR], cookie-file:PATH.")
@click.option("--engine", "engine", metavar="ID", help="ASR hypothesis to run (default w-prod).")
@click.option(
    "--source",
    "transcript_source",
    type=click.Choice(["asr", "captions"]),
    help="Transcribe with ASR (default), or trust the manual caption track and fall back to ASR without one.",
)
@click.option(
    "--correct/--no-correct", default=None, help="LLM correction of misheard words (default off; needs --llm)."
)
@click.option(
    "--chapters", type=click.Choice(["auto", "none"]), help="Chapter headings: auto uses YouTube's chapters (default)."
)
@click.option(
    "--llm",
    "llm_spec",
    metavar="SPEC",
    help="LLM for punctuation, correction and numeric verdicts: PROVIDER:MODEL (claude-cli:claude-sonnet-5-5) or none.",
)
@click.option("--captions/--no-captions", default=True, show_default=True, help="Download caption tracks.")
@click.option("--force", is_flag=True, help="Redo videos from scratch, discarding every stored stage.")
@click.option("--ignore-cooldown", is_flag=True, help="Start even though a bot or rate-limit cooldown is active.")
def run_command(  # noqa: PLR0913
    urls: tuple[str, ...],
    out: Path | None,
    auth_spec: str | None,
    engine: str | None,
    transcript_source: str | None,
    chapters: str | None,
    llm_spec: str | None,
    *,
    correct: bool | None,
    captions: bool,
    force: bool,
    ignore_cooldown: bool,
) -> None:
    """Turn videos, playlists or channels into context.json and context.md.

    Fetches each video, transcribes it locally, restores punctuation and verifies numbers, and writes both files
    next to the downloaded media. Every stage is stored, so a rerun redoes only what changed options reach; a video
    whose context.json is current is skipped unless --force is given.
    Exit codes: 0 ok, 1 some videos failed, 2 auth error, 3 batch stopped (cooldown).
    """
    bad = [url for url in urls if ids.classify(url) is None]
    if bad:
        click.echo(f"error: not a YouTube video, playlist or channel: {', '.join(bad)}", err=True)
        raise SystemExit(EXIT_FAILED)
    source = None
    try:
        source = resolve_source(auth_spec)
        overrides: dict[str, dict[str, object]] = {
            "asr": {"engine": engine, "source": transcript_source},
            "text": {"correct": correct, "chapters": chapters},
        }
        options = config.load_options(overrides=overrides)
        if llm_spec is not None:
            options = replace(options, llm=apply_spec(options.llm, llm_spec))
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
    except EngineMemoryError as exc:
        click.echo(f"■ {redact(exc.detail)}", err=True)
        click.echo(f"  Fix: {exc.fix}", err=True)
        raise SystemExit(EXIT_FAILED) from None
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
    flags = sum(len(segment.flags) for segment in document.segments)
    facts = [
        f"{len(document.segments)} segments",
        f"{len(document.paragraphs)} paragraphs",
        f"{len(document.chapters)} chapters",
        f"{flags} flags",
        f"punctuation: {document.transcript.punctuation}",
    ]
    if outcome.reused and "context" in outcome.reused:
        facts.append("already done")
    else:
        facts.append(f"{result.elapsed_s:.0f}s")
    click.echo(f"✓ {outcome.video_id}  {redact(str(markdown))}  ({' · '.join(facts)})")
    for notice in result.notices:
        click.echo(f"  ! {notice}")
    if result.timings:
        click.echo(f"    stages: {' · '.join(_timing(t) for t in result.timings)}")
    if document.llm_usage:
        # What the document cost in total, batches reused from earlier runs included; not what this run spent.
        click.echo(f"    document LLM tokens: {' · '.join(_usage(u) for u in document.llm_usage)}")


def _usage(usage: LLMUsage) -> str:
    def tokens(count: int | None) -> str:
        return "?" if count is None else str(count)

    return f"{usage.stage} {usage.calls} call(s), {tokens(usage.input_tokens)} in / {tokens(usage.output_tokens)} out"


def _timing(timing: pipeline.StageTiming) -> str:
    return f"{timing.stage} stored" if timing.reused else f"{timing.stage} {timing.seconds:.0f}s"


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
