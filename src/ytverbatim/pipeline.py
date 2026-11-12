"""`ytverbatim run`: fetch, transcribe locally, segment and write `context.json` plus `context.md` per video.

This is the first usable slice and has no LLM stage. Acquisition is the fetch session's job; ASR runs in the worker
subprocess; everything between the worker's chunk rows and the contract document is `build_document`, a pure function.
"""

import collections
import json
import logging
import shutil
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import opencc
from pydantic import JsonValue, ValidationError

from ytverbatim import __version__, atomic, config, ids, paths
from ytverbatim.acquire.audio import WAV_NAME
from ytverbatim.acquire.metadata import VideoMetadata, info_path, load_metadata
from ytverbatim.acquire.session import (
    BatchReport,
    FetchOptions,
    FetchSession,
    PlannedVideo,
    VideoOutcome,
    VideoStatus,
    default_runner,
    fetch,
    video_lock,
)
from ytverbatim.asr import chunks, registry
from ytverbatim.asr import runner as asr_runner
from ytverbatim.asr.models import ChunkHypothesis, FileJob, WorkerRequest
from ytverbatim.asr.protocol import EngineDef
from ytverbatim.auth.cooldown import utcnow
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
    EngineMemoryError,
    MediaValidationError,
    YtverbatimError,
)
from ytverbatim.render import render_markdown
from ytverbatim.text import segment as seg
from ytverbatim.ytdlp.runner import redact
from ytverbatim.ytdlp.types import YtDlpRunner

log = logging.getLogger(__name__)

CONTEXT_JSON = "context.json"
CONTEXT_MD = "context.md"
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
    # The exception behind a failed outcome when the pipeline caught one, so `process` can raise it unchanged.
    error: Exception | None = None


def process(
    url: str,
    options: config.Options | None = None,
    *,
    out: Path | None = None,
    auth: str | None = None,
    captions: bool = True,
    force: bool = False,
    progress: Callable[[ProgressEvent], None] | None = None,
    runner_factory: Callable[[Sequence[str]], YtDlpRunner] = default_runner,
) -> ContextDocument:
    """Produce the context document for one video, reusing it when `context.json` already exists from this engine.

    Raises the video's own error (`AuthError`, `EngineError`, `ContractError`, ...), or `BatchStopError` when
    YouTube asked us to back off before the video was done. `auth` is an auth spec as for `--auth`; without it
    the configured source applies.
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
    (result,) = results
    if result.document is not None:
        if report.stopped is not None:
            # The cooldown file is already written, so later runs wait; this video itself finished.
            log.warning("%s is done, but YouTube asked to back off: %s", ref.id, report.stopped)
        return result.document
    if report.stopped is not None:
        raise report.stopped
    if result.error is not None:
        raise result.error
    raise _error_for(result.outcome)


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
    runner_factory: Callable[[Sequence[str]], YtDlpRunner] = default_runner,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = utcnow,
) -> BatchReport:
    """Run every video behind `targets` into `<out>/<id>/` and write `<out>/batch_report.json`.

    Raises `DependencyError` when the engine's extra is missing and refuses a cooldown (`BatchStopError`), both
    before any request; the batch itself is `fetch`'s, with this pipeline as its per-video hook. Per-video
    failures land in the report; a batch stop mid-run ends the batch and is recorded as well. An ASR worker killed
    by the memory watch raises `EngineMemoryError`, after the report of the videos before it is written: every
    later video would run the same engine into the same limit.
    """
    options = options or config.load_options()
    definition = _definition(options)
    registry.check_dependencies(definition)
    out = out if out is not None else paths.default_output_root()
    fetch_options = FetchOptions(
        policy=options.auth.policy,
        captions=captions,
        cookie_channels=options.auth.cookie_channels,
        max_videos=options.ytdlp.max_videos_per_run,
        ignore_cooldown=ignore_cooldown,
    )
    settings = _Settings(
        definition,
        out,
        force,
        options.text.chapters == "youtube",
        progress or _no_progress,
        clock,
        asr_runner.memory_cap_bytes(options.asr.memory_cap_gib),
    )
    hook = _RunHook(settings)

    def emit(outcome: VideoOutcome) -> None:
        result = hook.results.pop(outcome.video_id, None)
        if on_video is not None:
            on_video(result if result is not None else VideoResult(outcome))

    return fetch(
        targets,
        source=source,
        out=out,
        options=fetch_options,
        runner_factory=runner_factory,
        sleep=sleep,
        clock=clock,
        on_video=emit,
        hook=hook,
    )


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
    memory_cap_bytes: int | None


def _definition(options: config.Options) -> EngineDef:
    if options.asr.source != "asr":
        raise ConfigError("asr.source = 'captions' is not implemented yet; only ASR transcripts can be produced")
    return registry.get(options.asr.engine)


@dataclass
class _RunHook:
    """`fetch`'s per-video hook for `run`; keeps each video's full result for `process_many`'s `on_video`."""

    settings: _Settings
    results: dict[str, VideoResult] = field(default_factory=dict[str, VideoResult])

    def done(self, video_id: str) -> bool:
        try:
            return _reusable(paths.video_dir(video_id, self.settings.out), self.settings) is not None
        except (ContractError, OSError):
            return False

    def __call__(self, session: FetchSession, planned: PlannedVideo) -> VideoOutcome:
        began = time.monotonic()
        result = _run_video(session, planned, self.settings)
        self.results[planned.video_id] = replace(result, elapsed_s=time.monotonic() - began)
        return result.outcome


def _reusable(vdir: Path, settings: _Settings) -> ContextDocument | None:
    """Return the finished document, unless `--force` is set or it holds no run of the selected engine."""
    context = vdir / CONTEXT_JSON
    if settings.force or not context.is_file():
        return None
    document = load_document(context)
    wanted = _run_id(settings.definition)
    if all(run.id != wanted for run in document.transcript.runs):
        log.info("%s holds no %s run; transcribing it again", context, wanted)
        return None
    return document


def _run_id(definition: EngineDef) -> str:
    return f"asr:{definition.spec.id}"


def _run_video(session: FetchSession, planned: PlannedVideo, settings: _Settings) -> VideoResult:
    video_id = planned.video_id
    vdir = paths.video_dir(video_id, settings.out)
    try:
        with video_lock(vdir):
            document = _reusable(vdir, settings)
            if document is not None:
                settings.emit(ProgressEvent(video_id, "skip", f"{CONTEXT_JSON} exists"))
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
    except (BatchStopError, DependencyError, EngineMemoryError):
        raise
    except (YtverbatimError, ValueError, OSError) as exc:
        detail = str(exc) if isinstance(exc, YtverbatimError) else f"{type(exc).__name__}: {exc}"
        outcome = VideoOutcome(video_id, VideoStatus.ERROR, error=_error_kind(exc), detail=redact(detail))
        return VideoResult(outcome, error=exc)


def _error_kind(exc: Exception) -> str:
    match exc:
        case BusyError():
            return "busy"
        case EngineError():
            return "engine"
        case ContractError():
            return "contract"
        case MediaValidationError():
            return "media_invalid"
        case OSError():
            return "os_error"
        case ValueError():
            return "invalid_value"
        case _:
            return "error"


def _error_for(outcome: VideoOutcome) -> YtverbatimError:
    """Rebuild the error of a failure the fetch session recorded only as an outcome."""
    detail = outcome.detail or outcome.error or "failed"
    if outcome.auth_error and outcome.error is not None:
        return AuthError(AuthState(outcome.error), outcome.detail)
    if outcome.error == "media_invalid":
        return MediaValidationError(detail)
    return YtverbatimError(f"{outcome.video_id}: {outcome.error}: {detail}")


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
        memory_cap_bytes=settings.memory_cap_bytes,
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
    rows: list[ChunkHypothesis] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(ChunkHypothesis.model_validate_json(line))
        except ValidationError as exc:
            raise ContractError(f"{path.name} line {number} is not a valid hypothesis row:\n{exc}") from exc
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
    """Turn one engine's chunk rows into a validated document; no I/O.

    Raises `ContractError` when the inputs cannot make a valid document, such as a chapter that ends before it starts.
    """
    try:
        return _build(metadata, tags, access, definition, rows, chapters=chapters, created_at=created_at)
    except ValidationError as exc:
        raise ContractError(f"{metadata.video_id}: the transcript does not fit the contract:\n{exc}") from exc


def _build(
    metadata: VideoMetadata,
    tags: Sequence[str],
    access: Access,
    definition: EngineDef,
    rows: Sequence[ChunkHypothesis],
    *,
    chapters: bool,
    created_at: datetime,
) -> ContextDocument:
    spec = definition.spec
    language = _language(rows, metadata.language)
    tokens, interpolated, warnings = _tokens(rows, language)
    sentences = seg.segment_tokens(tokens, language)
    text = " ".join(sentence.text for sentence in sentences)
    script = _script(language, text)
    if script == "zh-Hant":
        warnings.append("the transcript contains Traditional Chinese characters, which are not converted")
    if not sentences:
        warnings.append("no speech was transcribed")

    run = TranscriptRun(
        id=_run_id(definition),
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
        _segment(index, sentence, run, paragraph_of[index], chapter_of[index], anomalies, interpolated)
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


def _tokens(rows: Sequence[ChunkHypothesis], language: str) -> tuple[list[seg.Token], set[int], list[str]]:
    """Return the tokens, the chunks whose word timings had to be partly interpolated, and timing warnings.

    Tokens carry the engine's word times wherever its words line up with its text; the rest is interpolated. A chunk
    without any word timings is spread over the chunk, which the warning covers rather than a flag on every segment.
    """
    zh = seg.uses_zh_rules(" ".join(row.text for row in rows), language)
    tokens: list[seg.Token] = []
    interpolated: set[int] = set()
    chunk_level = 0
    for row in rows:
        text = row.text.strip()
        if not text:
            continue
        words = [word for word in row.words if word.text.strip()]
        if words:
            piece, complete = seg.aligned_tokens(text, words, row.start, row.end, language=language, origin=row.chunk)
            if not complete:
                interpolated.add(row.chunk)
        else:
            chunk_level += 1
            piece = seg.pseudo_tokens(text, row.start, row.end, language=language, origin=row.chunk)
        # Chunk texts are stripped, so "end." and "Next" would otherwise fuse and hide the sentence break.
        if tokens and piece and not zh and not tokens[-1].text[-1:].isspace() and not piece[0].text[:1].isspace():
            piece[0] = replace(piece[0], text=f" {piece[0].text}")
        tokens += piece
    warnings: list[str] = []
    if chunk_level:
        warnings.append(
            f"timing is chunk-level for {chunk_level} chunk(s): segment times are spread over each chunk "
            "in proportion to characters"
        )
    if interpolated:
        warnings.append(
            f"word timings did not cover the text in {len(interpolated)} chunk(s); the uncovered words are timed "
            "between their timed neighbours and their segments are flagged timing_interpolated"
        )
    return _monotonic(tokens), interpolated, warnings


def _monotonic(tokens: Sequence[seg.Token]) -> list[seg.Token]:
    """Segment starts must not go backwards, and engines occasionally emit a word that does.

    Word times are already clamped to their chunk, so a bogus late start can hold later ones back only to its
    chunk's end.
    """
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
    interpolated: Collection[int],
) -> Segment:
    words = tuple(
        Word(text=t.text.strip(), start=t.start, end=t.end, conf=t.conf)
        for t in sentence.tokens
        if t.timed and t.text.strip()
    )
    flags: list[Flag] = []
    if any(not t.timed and t.origin in interpolated for t in sentence.tokens):
        flags.append(
            Flag(
                kind="timing_interpolated",
                char_start=0,
                char_end=len(sentence.text),
                note="some words here had no engine timing; their times are interpolated from the words around them",
            )
        )
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
        words=words or None,
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
    return {"repo": chunks.SILERO_REPO, "revision": chunks.SILERO_REVISION, **asdict(chunks.DEFAULT_VAD_PARAMS)}


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
