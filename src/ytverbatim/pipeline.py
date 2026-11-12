"""`ytverbatim run`: fetch, transcribe locally, segment and write `context.json` plus `context.md` per video.

This is the first usable slice and has no LLM stage. Acquisition is the fetch session's job; ASR runs in the worker
subprocess; everything between the worker's chunk rows and the contract document is `build_document`, a pure function.
"""

import collections
import contextlib
import fcntl
import json
import logging
import os
import shutil
import time
from collections.abc import Callable, Collection, Generator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import opencc
from pydantic import JsonValue

from ytverbatim import __version__, atomic, config, ids, paths
from ytverbatim.acquire.audio import WAV_NAME
from ytverbatim.acquire.metadata import VideoMetadata, info_path, load_metadata
from ytverbatim.acquire.session import (
    REPORT_FILE,
    BatchReport,
    FetchOptions,
    FetchSession,
    PlannedVideo,
    VideoOutcome,
    VideoStatus,
    check_cooldown,
    plan_videos,
    sleep_args,
    stop_fix,
)
from ytverbatim.asr import chunks, registry
from ytverbatim.asr import runner as asr_runner
from ytverbatim.asr.models import ChunkHypothesis, FileJob, WorkerRequest
from ytverbatim.asr.protocol import EngineDef
from ytverbatim.auth.cooldown import guarded
from ytverbatim.auth.source import CookieSource, resolve_source
from ytverbatim.auth.state import AuthState
from ytverbatim.contract import (
    Access,
    Chapter,
    ContextDocument,
    Flag,
    Paragraph,
    Quality,
    Segment,
    SourceRef,
    Transcript,
    TranscriptRun,
    Video,
    Word,
    dump_document,
    load_document,
)
from ytverbatim.contract import Generator as GeneratorInfo
from ytverbatim.contract.models import SCHEMA_VERSION
from ytverbatim.errors import (
    AuthError,
    BatchStopError,
    BusyError,
    ConfigError,
    ContractError,
    DependencyError,
    EngineError,
    MediaValidationError,
    YtverbatimError,
)
from ytverbatim.render import render_markdown
from ytverbatim.text import segment as seg
from ytverbatim.ytdlp.runner import SubprocessRunner, redact
from ytverbatim.ytdlp.types import YtDlpRunner

log = logging.getLogger(__name__)

CONTEXT_JSON = "context.json"
CONTEXT_MD = "context.md"
LOCK_NAME = ".lock"
STAGES_DIR = ".stages"
UNDETERMINED_LANGUAGE = "und"

# Chunk flags that mark a decode the pipeline cannot vouch for; the others only describe how the audio was cut.
_ANOMALY_FLAG = "decoder_anomaly"


@dataclass(frozen=True)
class ProgressEvent:
    video_id: str
    stage: str
    detail: str = ""


@dataclass(frozen=True)
class VideoResult:
    outcome: VideoOutcome
    document: ContextDocument | None = None
    markdown: Path | None = None
    elapsed_s: float = 0.0


def _default_runner(sleep: Sequence[str]) -> YtDlpRunner:
    return SubprocessRunner(sleep_args=sleep)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def process(
    url: str,
    options: config.Options | None = None,
    *,
    out: Path | None = None,
    auth: str | None = None,
    captions: bool = True,
    force: bool = False,
    progress: Callable[[ProgressEvent], None] | None = None,
    runner_factory: Callable[[Sequence[str]], YtDlpRunner] = _default_runner,
) -> ContextDocument:
    """Produce the context document for one video, reusing it when `context.json` already exists.

    Raises the video's own error (`AuthError`, `EngineError`, ...), or `BatchStopError` when YouTube asked
    us to back off. `auth` is an auth spec as for `--auth`; without it the configured source applies.
    """
    ref = ids.classify(url)
    if ref is None or ref.kind is not ids.RefKind.VIDEO:
        raise ValueError(f"process takes one YouTube video, not {url!r}; use process_many for playlists and channels")
    results: list[VideoResult] = []
    report = process_many(
        [url],
        options,
        out=out,
        source=resolve_source(auth),
        captions=captions,
        force=force,
        progress=progress,
        on_video=results.append,
        runner_factory=runner_factory,
    )
    if report.stopped is not None:
        raise report.stopped
    (result,) = results
    if result.document is None:
        raise _error_for(result.outcome)
    return result.document


def process_many(  # noqa: PLR0913
    targets: Sequence[str],
    options: config.Options | None = None,
    *,
    out: Path | None = None,
    source: CookieSource | None = None,
    captions: bool = True,
    force: bool = False,
    ignore_cooldown: bool = False,
    progress: Callable[[ProgressEvent], None] | None = None,
    on_video: Callable[[VideoResult], None] | None = None,
    runner_factory: Callable[[Sequence[str]], YtDlpRunner] = _default_runner,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = _utcnow,
) -> BatchReport:
    """Run every video behind `targets` into `<out>/<id>/` and write `<out>/batch_report.json`.

    Refuses to start during a cooldown (`BatchStopError`), like `fetch`. Per-video failures land in the report;
    a batch stop mid-run ends the batch and is recorded as well.
    """
    options = options or config.load_options()
    definition = _definition(options)
    out = out if out is not None else paths.default_output_root()
    started = clock()
    if not ignore_cooldown:
        check_cooldown(started)
    if options.auth.policy == "always" and source is None:
        raise AuthError(AuthState.NO_SOURCE, "--policy always needs an auth source")
    fetch_options = FetchOptions(
        policy=options.auth.policy,
        captions=captions,
        cookie_channels=options.auth.cookie_channels,
        max_videos=options.ytdlp.max_videos_per_run,
        ignore_cooldown=ignore_cooldown,
    )
    videos, capped = plan_videos(guarded(runner_factory(()), clock), targets, source=source, options=fetch_options)
    session = FetchSession(
        runner_factory(sleep_args(len(videos))), source=source, out=out, options=fetch_options, sleep=sleep, clock=clock
    )
    settings = _Settings(definition, out, force, options.text.chapters == "youtube", progress or _no_progress, clock)
    outcomes: list[VideoOutcome] = []
    stopped: BatchStopError | None = None
    try:
        for planned in videos:
            if stopped is not None:
                outcomes.append(VideoOutcome(planned.video_id, VideoStatus.NOT_ATTEMPTED))
                continue
            began = time.monotonic()
            try:
                result = _run_video(session, planned, settings)
            except BatchStopError as exc:
                stopped = exc
                result = VideoResult(
                    VideoOutcome(
                        planned.video_id, VideoStatus.ERROR, error=exc.signal.value, fix=stop_fix(exc.signal, exc.until)
                    )
                )
            result = replace(result, elapsed_s=time.monotonic() - began)
            outcomes.append(result.outcome)
            if on_video is not None:
                on_video(result)
            # A stop seen on a successful call lets that video finish but no later one start.
            if stopped is None:
                stopped = session.pending_stop()
    finally:
        report = BatchReport(tuple(outcomes), stopped, capped, started, clock())
        out.mkdir(parents=True, exist_ok=True)
        atomic.write_json(out / REPORT_FILE, report.to_json())
    return report


def _no_progress(_event: ProgressEvent) -> None:
    return None


@dataclass(frozen=True)
class _Settings:
    definition: EngineDef
    out: Path
    force: bool
    chapters: bool
    emit: Callable[[ProgressEvent], None]
    clock: Callable[[], datetime]


def _definition(options: config.Options) -> EngineDef:
    if options.asr.source != "asr":
        raise ConfigError("asr.source = 'captions' is not implemented yet; only ASR transcripts can be produced")
    return registry.get(options.asr.engine)


def _run_video(session: FetchSession, planned: PlannedVideo, settings: _Settings) -> VideoResult:
    video_id = planned.video_id
    vdir = paths.video_dir(video_id, settings.out)
    try:
        with video_lock(vdir):
            context = vdir / CONTEXT_JSON
            if not settings.force and context.is_file():
                settings.emit(ProgressEvent(video_id, "skip", f"{CONTEXT_JSON} exists"))
                document = load_document(context)
                outcome = VideoOutcome(
                    video_id,
                    VideoStatus.OK,
                    access=document.access,
                    chapters=len(document.chapters),
                    reused=("context",),
                )
                return VideoResult(outcome, document, vdir / CONTEXT_MD)
            settings.emit(ProgressEvent(video_id, "fetch"))
            outcome = session.fetch_video(video_id, cookies_first=planned.cookies_first)
            if outcome.status is not VideoStatus.OK or outcome.access is None:
                return VideoResult(outcome)
            return VideoResult(outcome, _transcribe(vdir, outcome.access, settings), vdir / CONTEXT_MD)
    except (BatchStopError, DependencyError):
        raise
    except YtverbatimError as exc:
        return VideoResult(VideoOutcome(video_id, VideoStatus.ERROR, error=_error_kind(exc), detail=redact(str(exc))))


def _error_kind(exc: YtverbatimError) -> str:
    if isinstance(exc, BusyError):
        return "busy"
    if isinstance(exc, EngineError):
        return "engine"
    if isinstance(exc, ContractError):
        return "contract"
    if isinstance(exc, MediaValidationError):
        return "media_invalid"
    return "error"


def _error_for(outcome: VideoOutcome) -> YtverbatimError:
    detail = outcome.detail or outcome.error or "failed"
    if outcome.auth_error and outcome.error is not None:
        return AuthError(AuthState(outcome.error), outcome.detail)
    if outcome.error == "busy":
        return BusyError(detail)
    if outcome.error == "engine":
        return EngineError(detail)
    return YtverbatimError(f"{outcome.video_id}: {outcome.error}: {detail}")


@contextlib.contextmanager
def video_lock(vdir: Path) -> Generator[None]:
    """Hold the video directory exclusively, or raise `BusyError` at once; waiting could mean hours of ASR."""
    vdir.mkdir(parents=True, exist_ok=True)
    fd = os.open(vdir / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BusyError(f"another ytverbatim run is processing {vdir.name}") from None
        yield
    finally:
        os.close(fd)


def _transcribe(vdir: Path, access: Access, settings: _Settings) -> ContextDocument:
    definition = settings.definition
    metadata = load_metadata(vdir)
    video_id = metadata.video_id
    asr_dir = vdir / STAGES_DIR / "asr" / definition.spec.id
    if settings.force:
        shutil.rmtree(asr_dir, ignore_errors=True)
    settings.emit(ProgressEvent(video_id, "asr", definition.spec.id))
    request = WorkerRequest(
        engine=definition,
        out_dir=str(asr_dir),
        files=(FileJob(id=video_id, wav=str(vdir / WAV_NAME), language=metadata.language),),
    )
    result = asr_runner.run_worker(request)
    rows = _read_rows(result.hypotheses, video_id)
    settings.emit(ProgressEvent(video_id, "write"))
    document = build_document(
        metadata=metadata,
        tags=_tags(vdir),
        access=access,
        definition=definition,
        rows=rows,
        chapters=settings.chapters,
        created_at=settings.clock(),
    )
    atomic.write_text(vdir / CONTEXT_MD, render_markdown(document))
    # The sentinel goes last: a context.json on disk means the markdown beside it is complete too.
    atomic.write_text(vdir / CONTEXT_JSON, dump_document(document))
    (vdir / WAV_NAME).unlink(missing_ok=True)
    return document


def _read_rows(path: Path, video_id: str) -> list[ChunkHypothesis]:
    rows = [
        ChunkHypothesis.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted((row for row in rows if row.file == video_id), key=lambda row: row.chunk)


def _tags(vdir: Path) -> tuple[str, ...]:
    # VideoMetadata does not keep tags, but the raw extraction does.
    try:
        raw: object = json.loads(info_path(vdir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    tags = raw.get("tags") if isinstance(raw, dict) else None
    return tuple(tag for tag in tags if isinstance(tag, str)) if isinstance(tags, list) else ()


def build_document(
    *,
    metadata: VideoMetadata,
    tags: Sequence[str],
    access: Access,
    definition: EngineDef,
    rows: Sequence[ChunkHypothesis],
    chapters: bool,
    created_at: datetime,
) -> ContextDocument:
    """Turn one engine's chunk rows into a validated document; no I/O."""
    spec = definition.spec
    language = _language(rows, metadata.language)
    tokens, warnings = _tokens(rows, language)
    sentences = seg.segment_tokens(tokens, language)
    text = " ".join(sentence.text for sentence in sentences)
    script = _script(language, text)
    if script == "zh-Hant":
        warnings.append("the transcript contains Traditional Chinese characters, which are not converted")
    if not sentences:
        warnings.append("no speech was transcribed")

    run = TranscriptRun(
        id=f"asr:{spec.id}",
        kind="asr",
        engine=spec.id,
        model_repo=spec.repo,
        model_revision=spec.revision,
        params={**definition.variant, "language_hint": metadata.language, "sample_rate": spec.sample_rate},
        vad=_vad_info() if spec.max_chunk_s is not None else None,
        chunking=_chunking_info(spec.max_chunk_s, rows),
        aligner=None,
    )
    chapter_of, chapter_list = _chapters(metadata, sentences, enabled=chapters)
    paragraph_ranges = seg.split_paragraphs([s.start for s in sentences], [s.end for s in sentences], chapter_of)
    paragraph_of = [index for index, (first, last) in enumerate(paragraph_ranges) for _ in range(first, last + 1)]
    anomalies = {row.chunk: row for row in rows if _ANOMALY_FLAG in row.flags}
    segments = tuple(
        _segment(index, sentence, run, paragraph_of[index], chapter_of[index], anomalies)
        for index, sentence in enumerate(sentences)
    )
    paragraphs = tuple(
        Paragraph(
            index=index,
            start=sentences[first].start,
            end=max(s.end for s in sentences[first : last + 1]),
            first_segment=first,
            last_segment=last,
        )
        for index, (first, last) in enumerate(paragraph_ranges)
    )
    return ContextDocument(
        schema_version=SCHEMA_VERSION,
        generator=GeneratorInfo(name="ytverbatim", version=__version__),
        created_at=created_at,
        video=_video(metadata, tags, rows),
        access=access,
        transcript=Transcript(
            language=language,
            script=script,
            runs=(run,),
            punctuation="engine" if seg.is_punctuated(text, language) else "none",
            correction_llm=None,
        ),
        chapters=chapter_list,
        paragraphs=paragraphs,
        segments=segments,
        quality=_quality(rows, sentences, warnings, chunked=spec.max_chunk_s is not None),
        llm_usage=(),
    )


def _language(rows: Sequence[ChunkHypothesis], declared: str | None) -> str:
    """Return the language the engine reports for most of the text, else the one YouTube declares."""
    weight: collections.Counter[str] = collections.Counter()
    for row in rows:
        if row.language and row.text.strip():
            weight[row.language] += len(row.text)
    if weight:
        return weight.most_common(1)[0][0]
    return declared or UNDETERMINED_LANGUAGE


def _squash(text: str) -> str:
    return "".join(text.split())


def _tokens(rows: Sequence[ChunkHypothesis], language: str) -> tuple[list[seg.Token], list[str]]:
    """Word tokens where the engine's word times reproduce its text, else chunk-level tokens."""
    tokens: list[seg.Token] = []
    interpolated = mismatched = 0
    for row in rows:
        text = row.text.strip()
        if not text:
            continue
        words = [word for word in row.words if word.text.strip()]
        if words and _squash("".join(word.text for word in words)) == _squash(text):
            piece = [seg.Token(w.text, w.start, w.end, w.conf, origin=row.chunk) for w in words]
        else:
            interpolated += 1
            mismatched += bool(words)
            piece = seg.pseudo_tokens(text, row.start, row.end, language=language, origin=row.chunk)
        # Chunk texts are stripped, so "end." and "Next" would otherwise fuse and hide the sentence break.
        if (
            tokens
            and piece
            and not seg.is_zh(language)
            and not tokens[-1].text[-1:].isspace()
            and not piece[0].text[:1].isspace()
        ):
            piece[0] = replace(piece[0], text=f" {piece[0].text}")
        tokens += piece
    warnings: list[str] = []
    if interpolated:
        warnings.append(
            f"timing is chunk-level for {interpolated} chunk(s): segment times are spread over each chunk "
            "in proportion to characters"
        )
    if mismatched:
        warnings.append(f"word timings did not match the text in {mismatched} chunk(s) and were not used")
    return _monotonic(tokens), warnings


def _monotonic(tokens: Sequence[seg.Token]) -> list[seg.Token]:
    """Segment starts must not go backwards, and engines occasionally emit a word that does."""
    out: list[seg.Token] = []
    floor = 0.0
    for token in tokens:
        start = max(token.start, floor)
        out.append(replace(token, start=start, end=max(token.end, start)))
        floor = start
    return out


def _script(language: str, text: str) -> Literal["zh-Hans", "zh-Hant", "n/a"]:
    if not seg.is_zh(language):
        return "n/a"
    return "zh-Hans" if opencc.OpenCC("t2s").convert(text) == text else "zh-Hant"


def _chapters(
    metadata: VideoMetadata, sentences: Sequence[seg.Sentence], *, enabled: bool
) -> tuple[list[int | None], tuple[Chapter, ...]]:
    """Map sentences to YouTube chapters by start time, dropping chapters that no sentence starts in."""
    if not enabled or not metadata.chapters or not sentences:
        return [None] * len(sentences), ()
    youtube = sorted(metadata.chapters, key=lambda c: c.start)
    raw = seg.chapter_indexes([s.start for s in sentences], [c.start for c in youtube])
    order = list(dict.fromkeys(raw))
    compact = {old: new for new, old in enumerate(order)}
    chapters = tuple(
        Chapter(
            index=compact[old],
            title=youtube[old].title,
            start=youtube[old].start,
            end=youtube[old].end,
            source="youtube",
            first_segment=raw.index(old),
            last_segment=len(raw) - 1 - raw[::-1].index(old),
        )
        for old in order
    )
    return [compact[old] for old in raw], chapters


def _segment(
    index: int,
    sentence: seg.Sentence,
    run: TranscriptRun,
    paragraph: int,
    chapter: int | None,
    anomalies: Mapping[int, ChunkHypothesis],
) -> Segment:
    timed = all(token.timed for token in sentence.tokens)
    words = (
        tuple(
            Word(text=t.text.strip(), start=t.start, end=t.end, conf=t.conf) for t in sentence.tokens if t.text.strip()
        )
        if timed
        else None
    )
    flags: list[Flag] = []
    for chunk, row in anomalies.items():
        span = sentence.span_of(chunk)
        if span is not None:
            flags.append(
                Flag(
                    kind="decoder_anomaly",
                    char_start=span[0],
                    char_end=span[1],
                    note=f"the decoder looped or stalled in the chunk at {_clock(row.start)}-{_clock(row.end)}",
                )
            )
    return Segment(
        id=index,
        start=sentence.start,
        end=sentence.end,
        text=sentence.text,
        words=words,
        source=SourceRef(kind=run.kind, run=run.id),
        flags=tuple(sorted(flags, key=lambda f: f.char_start)),
        paragraph=paragraph,
        chapter=chapter,
    )


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def _video(metadata: VideoMetadata, tags: Sequence[str], rows: Sequence[ChunkHypothesis]) -> Video:
    duration = metadata.duration if metadata.duration is not None else max((row.end for row in rows), default=0.0)
    return Video(
        id=metadata.video_id,
        url=ids.video_url(metadata.video_id),
        title=metadata.title,
        channel=metadata.channel,
        channel_id=metadata.channel_id or "",
        upload_date=_upload_date(metadata.upload_date),
        duration_s=duration,
        description=metadata.description,
        language=metadata.language,
        tags=tuple(tags),
    )


def _upload_date(raw: str | None) -> date | None:
    try:
        return datetime.strptime(raw, "%Y%m%d").date() if raw else None
    except ValueError:
        return None


def _vad_info() -> dict[str, JsonValue]:
    # Imported here: asr.vad needs numpy (the asr extra), which a whole-file run never does.
    from ytverbatim.asr import vad

    return {"repo": vad.SILERO_REPO, "revision": vad.SILERO_REVISION, **asdict(vad.DEFAULT_VAD_PARAMS)}


def _chunking_info(max_chunk_s: float | None, rows: Sequence[ChunkHypothesis]) -> dict[str, JsonValue] | None:
    if max_chunk_s is None:
        return None
    return {
        "max_chunk_s": max_chunk_s,
        "tail_fraction": chunks.DEFAULT_TAIL_FRACTION,
        "min_pause_s": chunks.DEFAULT_MIN_PAUSE_S,
        "chunks": len(rows),
    }


def _quality(
    rows: Sequence[ChunkHypothesis], sentences: Sequence[seg.Sentence], warnings: Sequence[str], *, chunked: bool
) -> Quality:
    counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        counts.update(row.flags)
        if not row.text.strip() and "empty_output" not in row.flags:
            counts["empty_output"] += 1
    transcribed = _covered_seconds([(s.start, s.end) for s in sentences])
    # Without a VAD (whole-file engines) there is no separate speech estimate; the transcribed span stands in.
    speech = sum(row.end - row.start for row in rows) if chunked else transcribed
    return Quality(
        speech_seconds=speech,
        transcribed_seconds=transcribed,
        decoder_anomalies=counts[_ANOMALY_FLAG],
        flags_by_kind=dict(counts),
        warnings=tuple(warnings),
    )


def _covered_seconds(spans: Collection[tuple[float, float]]) -> float:
    total = 0.0
    reach = 0.0
    for start, end in sorted(spans):
        start = max(start, reach)
        if end > start:
            total += end - start
            reach = end
    return total
