"""ASR selection eval runner: ``uv run python -m evals.run_asr <command>``.

Order: ``vad`` → ``asr --hyp H`` (per hypothesis, then ``cleanup --hyp H``) → ``captions`` → ``screen`` →
``session`` → ``listen`` → ``report`` → ``decide``. Every command is resumable and writes only under
``$YTVERBATIM_EVAL_DIR``, except ``decide``, which writes the aggregate ``evals/results/asr-<date>.md``.
"""

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import click

from evals import disk
from evals.decide import decide, render_decision
from evals.judge import JUDGE_MODEL, PROMPT_VERSION, load_verdicts, run_judge
from evals.listen import listen as listen_loop
from evals.manifest import Manifest, Video, eval_dir, load_manifest
from evals.report import REPORT_JSON, REPORT_MD, SCREEN_FILE, Report, build_report, render_markdown
from evals.screen import (
    ScreenResult,
    build_screen,
    collect_spans,
    judge_items,
    recommendation_text,
    screen_table,
)
from evals.session import DEFAULT_N, build_session, write_session
from evals.store import (
    CAPTION_HYPOTHESIS,
    REFERENCE,
    Layout,
    Plan,
    RunRecord,
    append_history,
    dump_jsonl,
    file_complete,
    hypotheses_on_disk,
    incomplete_files,
    load_plan,
    now,
    require_plan,
    rows_by_file,
    save_plan,
    videos,
    widened_windows,
    write_json,
    write_rows,
)
from ytverbatim import atomic, config, paths
from ytverbatim.acquire.captions import CaptionKind, parse_json3
from ytverbatim.acquire.metadata import trim_info
from ytverbatim.asr import registry
from ytverbatim.asr.chunks import DEFAULT_MAX_CHUNK_S, Region, alignment_windows, plan_chunks
from ytverbatim.asr.context import context_from_metadata
from ytverbatim.asr.models import ChunkHypothesis, FileJob, WorkerRequest
from ytverbatim.asr.protocol import EngineDef
from ytverbatim.asr.runner import WorkerResult, memory_cap_bytes, run_worker, worker_lock
from ytverbatim.contract.models import Word
from ytverbatim.errors import ConfigError, YtverbatimError
from ytverbatim.llm import ClaudeCLIClient

RESULTS_DIR = Path(__file__).parent / "results"
DISPUTES_FILE = "disputes.jsonl"

Regions = Callable[[Path], tuple[list[Region], float, list[float]]]
Worker = Callable[[WorkerRequest], WorkerResult]


def default_hf_cache() -> Path:
    return paths.home_dir() / "hf-candidates"


def layout() -> Layout:
    return Layout(eval_dir())


def _vad_regions(wav: Path) -> tuple[list[Region], float, list[float]]:
    # The audio and VAD modules need numpy and MLX from the asr extra; only this command uses them.
    from ytverbatim.asr.audio import frame_energy, load_wav
    from ytverbatim.asr.vad import speech_regions

    audio = load_wav(wav)
    return speech_regions(audio), audio.duration, frame_energy(audio)


def plan_file(store: Layout, video: Video, regions: Regions = _vad_regions) -> Plan:
    from ytverbatim.asr.vad import DEFAULT_VAD_PARAMS, SILERO_REVISION

    speech, duration, energy = regions(store.wav(video))
    chunks = plan_chunks(speech, duration, max_chunk_s=DEFAULT_MAX_CHUNK_S, frame_energy=energy)
    windows = [(w.start, w.end) for w in alignment_windows(chunks)]
    return Plan(
        file=video.id,
        duration_s=duration,
        speech=tuple((r.start, r.end) for r in speech),
        chunks=tuple(chunks),
        windows=widened_windows(windows, duration),
        max_chunk_s=DEFAULT_MAX_CHUNK_S,
        vad={k: float(v) for k, v in asdict(DEFAULT_VAD_PARAMS).items()},
        silero_revision=SILERO_REVISION,
    )


def file_job(store: Layout, video: Video, definition: EngineDef, plan: Plan) -> FileJob:
    """Build the worker job exactly as production would from the video's metadata.

    ``language`` is the uploader-declared metadata language (``None`` when absent); the registry's per-engine
    language policy, not this runner, decides what each engine does with it.
    """
    metadata = trim_info(json.loads((store.root / video.meta).read_text(encoding="utf-8")))
    return FileJob(
        id=video.id,
        wav=str(store.wav(video)),
        language=metadata.language,
        context=context_from_metadata(metadata) if definition.spec.supports_context else None,
        chunks=None if definition.spec.max_chunk_s is None else plan.chunks,
    )


@dataclass(frozen=True)
class AsrOutcome:
    decoded_files: tuple[str, ...]
    skipped_files: tuple[str, ...]
    download_bytes: int
    results: tuple[WorkerResult, ...]


def run_hypothesis(
    store: Layout,
    manifest: Manifest,
    hypothesis: str,
    files: Sequence[str],
    cache: Path,
    *,
    definition: EngineDef | None = None,
    worker: Worker = run_worker,
    free: Callable[[Path], int] = disk.free_bytes,
    memory_cap: int | None = None,
) -> AsrOutcome:
    definition = definition or registry.get(hypothesis)
    done = rows_by_file(store, hypothesis)
    pending: list[Video] = []
    skipped: list[str] = []
    for video in videos(manifest, files):
        plan = require_plan(store, video.id)
        if file_complete(definition, plan, done.get(video.id, [])):
            skipped.append(video.id)
        else:
            pending.append(video)
    if not pending:
        return AsrOutcome((), tuple(skipped), 0, ())
    download = disk.check_disk(definition, cache, free=free)
    results: list[WorkerResult] = []
    # One worker per file keeps each output directory's request fingerprint fixed, so any subset of files can be
    # run, interrupted and resumed in any order; the price is one model load per file.
    for video in pending:
        request = WorkerRequest(
            engine=definition,
            out_dir=str(store.file_run_dir(hypothesis, video.id)),
            files=(file_job(store, video, definition, require_plan(store, video.id)),),
            hf_hub_cache=str(cache),
            memory_cap_bytes=memory_cap,
        )
        result = worker(request)
        append_history(
            store,
            hypothesis,
            RunRecord(
                at=now(),
                files=(video.id,),
                file_seconds=video.audio_duration_s,
                metrics=result.metrics,
                children_max_rss_bytes=result.children_max_rss_bytes,
                elapsed_s=result.elapsed_s,
            ),
        )
        results.append(result)
    return AsrOutcome(tuple(v.id for v in pending), tuple(skipped), download, tuple(results))


def cleanup_hypothesis(
    store: Layout,
    manifest: Manifest,
    hypothesis: str,
    cache: Path,
    *,
    delete: Callable[[Path, str], int] = disk.delete_repo,
) -> dict[str, int]:
    """Delete a finished hypothesis's weights from the candidate cache; return bytes freed per repo."""
    definition = registry.get(hypothesis)
    if definition.spec.family in disk.DEFAULT_CACHE_FAMILIES:
        raise ConfigError(f"{hypothesis} uses the default Hugging Face cache, which cleanup never touches")
    disk.require_candidate_cache(cache)
    missing = incomplete_files(store, manifest, definition)
    if missing:
        raise ConfigError(f"{hypothesis} is not complete for {len(missing)} file(s) ({', '.join(missing[:3])}…)")
    repos = {w.repo for w in disk.weights(definition)}
    for other_id, other in registry.HYPOTHESES.items():
        if other_id == hypothesis or not repos & {w.repo for w in disk.weights(other)}:
            continue
        if incomplete_files(store, manifest, other):
            raise ConfigError(f"{other_id} shares weights with {hypothesis} and still has files to run; run it first")
    return {repo: delete(cache, repo) for repo in sorted(repos)}


def build_captions(store: Layout, manifest: Manifest) -> dict[str, int]:
    """`yt-auto` from original-language auto json3 tracks and `ref-manual` from manual zh json3 tracks."""
    auto: list[ChunkHypothesis] = []
    reference: list[ChunkHypothesis] = []
    for video in manifest.videos:
        for track in video.captions.obtained:
            if track.format != "json3":
                continue
            if track.kind == "auto" and track.lang.endswith("-orig"):
                auto += _caption_rows(store, video, track.path, CAPTION_HYPOTHESIS, CaptionKind.AUTO)
            elif track.kind == "manual" and track.lang.split("-")[0] == "zh":
                reference += _caption_rows(store, video, track.path, REFERENCE, CaptionKind.MANUAL)
    write_rows(store, CAPTION_HYPOTHESIS, auto)
    write_rows(store, REFERENCE, reference)
    return {
        CAPTION_HYPOTHESIS: len({r.file for r in auto}),
        REFERENCE: len({r.file for r in reference}),
    }


def _caption_rows(store: Layout, video: Video, path: str, name: str, kind: CaptionKind) -> list[ChunkHypothesis]:
    segments = parse_json3((store.root / path).read_text(encoding="utf-8"), kind=kind)
    return [
        ChunkHypothesis(
            hypothesis=name,
            file=video.id,
            chunk=index,
            start=segment.start,
            end=segment.end,
            text=segment.text,
            words=tuple(Word(text=w.text, start=w.start, end=w.end) for w in segment.words),
        )
        for index, segment in enumerate(segments)
    ]


def load_screen(store: Layout) -> ScreenResult:
    path = store.screen / SCREEN_FILE
    if not path.exists():
        raise ConfigError("no screening result; run `screen` first")
    return ScreenResult.model_validate_json(path.read_text(encoding="utf-8"))


def write_decision(report: Report, results_dir: Path, day: str) -> tuple[Path, str]:
    decision = decide(report)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"asr-{day}.md"
    atomic.write_text(path, render_markdown(report) + "\n" + render_decision(decision))
    return path, decision.summary


def _files(value: str | None) -> list[str]:
    return [part for part in (value or "").split(",") if part]


@click.group()
def main() -> None:
    """ASR selection eval (data under $YTVERBATIM_EVAL_DIR)."""


_files_option = click.option("--files", help="Comma-separated video ids (default: every file in the manifest).")
_cache_option = click.option(
    "--hf-cache",
    type=click.Path(path_type=Path),
    default=None,
    help="Candidate weight cache (default: $YTVERBATIM_HOME/hf-candidates). Never the default HF cache.",
)


@main.command()
@_files_option
@_cache_option
def vad(files: str | None, hf_cache: Path | None) -> None:
    """Speech regions plus the shared ≤30 s chunk plan and alignment windows per file."""
    store, manifest = layout(), load_manifest()
    os.environ["HF_HUB_CACHE"] = str(hf_cache or default_hf_cache())
    with worker_lock():
        for video in videos(manifest, _files(files)):
            if load_plan(store, video.id) is not None:
                click.echo(f"{video.id}: plan exists")
                continue
            started = time.perf_counter()
            plan = plan_file(store, video)
            save_plan(store, plan)
            speech = sum(e - s for s, e in plan.speech)
            click.echo(
                f"{video.id}: {len(plan.chunks)} chunks, {len(plan.windows)} windows, "
                f"{speech / plan.duration_s:.0%} speech, {time.perf_counter() - started:.1f}s"
            )


@main.command()
@click.option("--hyp", "hypothesis", required=True, type=click.Choice(sorted(registry.HYPOTHESES)))
@_files_option
@_cache_option
def asr(hypothesis: str, files: str | None, hf_cache: Path | None) -> None:
    """Decode the selected files with one hypothesis (resumes per chunk)."""
    store, manifest = layout(), load_manifest()
    cap = memory_cap_bytes(config.load_options().asr.memory_cap_gib)
    outcome = run_hypothesis(store, manifest, hypothesis, _files(files), hf_cache or default_hf_cache(), memory_cap=cap)
    if not outcome.results:
        click.echo(f"{hypothesis}: complete for all {len(outcome.skipped_files)} selected files")
        return
    metrics = [r.metrics for r in outcome.results]
    decode, audio = sum(m.decode_s for m in metrics), sum(m.audio_s for m in metrics)
    click.echo(
        f"{hypothesis}: {len(outcome.decoded_files)} files, {sum(m.chunks_decoded for m in metrics)} chunks decoded "
        f"({sum(m.chunks_skipped for m in metrics)} resumed), load {sum(m.load_s for m in metrics):.1f}s, "
        f"decode {decode:.1f}s for {audio:.0f}s of chunk audio (RTF {decode / audio if audio else 0:.3f}), "
        f"wall {sum(r.elapsed_s for r in outcome.results):.1f}s"
    )


@main.command()
@click.option("--hyp", "hypothesis", required=True, type=click.Choice(sorted(registry.HYPOTHESES)))
@_cache_option
def cleanup(hypothesis: str, hf_cache: Path | None) -> None:
    """Verify a hypothesis is complete, then delete its weights from the candidate cache."""
    store, manifest = layout(), load_manifest()
    for repo, freed in cleanup_hypothesis(store, manifest, hypothesis, hf_cache or default_hf_cache()).items():
        click.echo(f"deleted {repo}: {freed / 1e9:.2f} GB freed")


@main.command()
def captions() -> None:
    """Build the yt-auto pseudo-hypothesis and the ref-manual reference from the caption tracks."""
    for name, count in build_captions(layout(), load_manifest()).items():
        click.echo(f"{name}: {count} files")


@main.command()
@click.option("--hyps", help="Comma-separated hypotheses (default: every run on disk).")
@click.option("--judge-n", default=400, show_default=True, help="Disputed spans sampled for the judge.")
@click.option("--seed", default=0, show_default=True)
@click.option("--max-calls", type=int, default=None, help="Stop after this many judge calls (quota).")
@click.option("--no-judge", is_flag=True, help="Automatic metrics only; uses cached verdicts if any.")
def screen(hyps: str | None, judge_n: int, seed: int, max_calls: int | None, no_judge: bool) -> None:
    """Disputes, automatic metrics and the blind judge sample; recommends the control and the challenger."""
    store, manifest = layout(), load_manifest()
    hypotheses = _files(hyps) or hypotheses_on_disk(store)
    spans = collect_spans(store, manifest, hypotheses)
    dump_jsonl(store.screen / DISPUTES_FILE, spans)
    items = judge_items(spans, judge_n, seed)
    if no_judge:
        cached = load_verdicts(store.judge)
        verdicts = {i.span_id: cached[i.key(JUDGE_MODEL)] for i in items if i.key(JUDGE_MODEL) in cached}
    else:
        client = ClaudeCLIClient(model=JUDGE_MODEL)
        verdicts = run_judge(client, items, store.judge, model=JUDGE_MODEL, max_calls=max_calls, progress=click.echo)
    result = build_screen(
        store,
        manifest,
        hypotheses,
        spans,
        verdicts,
        judge_model=JUDGE_MODEL if verdicts else None,
        judge_prompt_version=PROMPT_VERSION if verdicts else None,
        judge_seed=seed,
        judge_sample=len(items),
        generated=now(),
    )
    write_json(store.screen / SCREEN_FILE, result)
    click.echo(screen_table(result))
    click.echo(recommendation_text(result))


@main.command()
@click.option("--control", required=True)
@click.option("--challenger", required=True)
@click.option("--n", "count", default=DEFAULT_N, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--name", default="main", show_default=True)
def session(control: str, challenger: str, count: int, seed: int, name: str) -> None:
    """Build the stratified, side-randomised A/B session between the control and the challenger."""
    store, manifest = layout(), load_manifest()
    shown, key = build_session(store, manifest, control, challenger, n=count, seed=seed)
    created = write_session(store, name, shown, key)
    numeric = sum(e.stratum == "numeric" for e in key.entries)
    by_language = {lang: sum(e.language == lang for e in key.entries) for lang in ("zh", "en")}
    click.echo(
        f"session {name!r} {'built' if created else 'already exists'}: {len(shown)} spans "
        f"({numeric} numeric/entity; zh {by_language['zh']}, en {by_language['en']}) "
        f"from {sum(key.population.values())} disputes"
    )


@main.command()
@click.option("--name", default="main", show_default=True)
def listen(name: str) -> None:
    """Blind A/B in the terminal: one keypress per clip."""
    listen_loop(layout(), load_manifest(), name, getchar=click.getchar, echo=click.echo)


@main.command()
@click.option("--session", "session_name", default="main", show_default=True)
def report(session_name: str) -> None:
    """Aggregate JSON and markdown report under $YTVERBATIM_EVAL_DIR/report."""
    store, manifest = layout(), load_manifest()
    result = build_report(store, manifest, load_screen(store), session_name, now())
    write_json(store.report / REPORT_JSON, result)
    atomic.write_text(store.report / REPORT_MD, render_markdown(result))
    click.echo(f"wrote {store.report / REPORT_MD}")


@main.command(name="decide")
@click.option("--date", "day", default=None, help="Date for the results file name (default: today).")
def decide_command(day: str | None) -> None:
    """Apply the decision rule and write evals/results/asr-<date>.md (aggregates only)."""
    store = layout()
    path = store.report / REPORT_JSON
    if not path.exists():
        raise ConfigError("no report; run `report` first")
    written, summary = write_decision(
        Report.model_validate_json(path.read_text(encoding="utf-8")), RESULTS_DIR, day or date.today().isoformat()
    )
    click.echo(f"{summary}\nwrote {written}")


def run() -> None:
    try:
        main(standalone_mode=False)
    except YtverbatimError as exc:
        raise SystemExit(f"error: {exc}") from None
    except click.exceptions.Abort:
        raise SystemExit(130) from None
    except click.ClickException as exc:
        exc.show()
        raise SystemExit(exc.exit_code) from None


if __name__ == "__main__":
    run()
