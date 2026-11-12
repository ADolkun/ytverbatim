"""`ytverbatim run`: fetch, transcribe, refine the text and write `context.json` plus `context.md` per video.

Stages, per video under its directory lock: metadata, captions and audio (the fetch session's) → asr (the worker
subprocess, which also plans the VAD chunks) or manual captions → normalize → punctuate → correct → numeric →
segment, chapters and assemble (`stages.build_document`) → render. Every stage from asr to numeric is checkpointed
in the video's `StageStore`, keyed by what it read, so a rerun with changed options recomputes only the stages those
options reach, and an interrupted run resumes where it stopped. The 16 kHz WAV is deleted once the ASR output is
stored; the compressed audio stays.
"""

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from ytverbatim import atomic, config, ids, paths, stages, store
from ytverbatim.acquire.audio import WAV_NAME
from ytverbatim.acquire.captions import CaptionKind, CaptionTrack
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
    saved_captions,
)
from ytverbatim.asr import registry
from ytverbatim.asr import runner as asr_runner
from ytverbatim.asr.models import ChunkHypothesis, FileJob, WorkerRequest
from ytverbatim.asr.protocol import EngineDef
from ytverbatim.auth.cooldown import utcnow
from ytverbatim.auth.source import CookieSource, resolve_source
from ytverbatim.auth.state import AuthState
from ytverbatim.contract import ContextDocument, LLMUsage, dump_document, load_document
from ytverbatim.errors import (
    AuthError,
    BatchStopError,
    BusyError,
    ConfigError,
    ContractError,
    DependencyError,
    EngineError,
    EngineMemoryError,
    LLMError,
    MediaValidationError,
    YtDlpError,
    YtverbatimError,
)
from ytverbatim.llm import client_from_options
from ytverbatim.llm.protocol import LLMClient
from ytverbatim.render import render_markdown
from ytverbatim.stages import Stretch, Transcribed, build_document, stretch_flag, with_results
from ytverbatim.store import StageStore, input_hash
from ytverbatim.store import lock as video_lock
from ytverbatim.text import numeric
from ytverbatim.text.correct import CorrectionContext, correct_segments
from ytverbatim.text.punctuate import needs_punctuation, punctuate
from ytverbatim.ytdlp.runner import redact
from ytverbatim.ytdlp.types import YtDlpRunner

log = logging.getLogger(__name__)

CONTEXT_JSON = "context.json"
CONTEXT_MD = "context.md"
STAGES_DIR = store.STAGES_DIR

# Bump a stage's version when its code changes what it outputs for the same input; only that stage and the stages
# its output feeds are recomputed.
STAGE_VERSIONS: dict[str, int] = {
    "asr": 1,
    "captions": 1,
    "second_opinion": 1,
    "normalize": 1,
    "punctuate": 1,
    "correct": 1,
    "numeric": 1,
    "context": 1,
}
_GLOSSARY_MAX_TERMS = 20

__all__ = [
    "CONTEXT_JSON",
    "CONTEXT_MD",
    "ProgressEvent",
    "StageTiming",
    "VideoResult",
    "build_document",
    "process",
    "process_many",
    "video_lock",
]


@dataclass(frozen=True)
class ProgressEvent:
    video_id: str
    stage: str
    detail: str = ""


@dataclass(frozen=True)
class StageTiming:
    stage: str
    seconds: float
    reused: bool


@dataclass(frozen=True)
class VideoResult:
    outcome: VideoOutcome
    document: ContextDocument | None = None
    markdown: Path | None = None
    elapsed_s: float = 0.0
    # The exception behind a failed outcome when the pipeline caught one, so `process` can raise it unchanged.
    error: Exception | None = None
    # Stages that could not run as asked (no LLM for punctuation, no manual captions); also in `quality.warnings`.
    notices: tuple[str, ...] = ()
    timings: tuple[StageTiming, ...] = ()


def process(
    url: str,
    options: config.Options | None = None,
    *,
    llm: LLMClient | None = None,
    out: Path | None = None,
    auth: str | None = None,
    captions: bool = True,
    force: bool = False,
    progress: Callable[[ProgressEvent], None] | None = None,
    runner_factory: Callable[[Sequence[str]], YtDlpRunner] = default_runner,
) -> ContextDocument:
    """Produce the context document for one video, reusing it when nothing it depends on has changed.

    `llm` is the client for punctuation, correction and numeric verdicts; without it the configured `[llm]`
    provider applies, and with none configured those stages degrade as `quality.warnings` records. Raises the
    video's own error (`AuthError`, `EngineError`, `LLMError`, `ContractError`, ...), or `BatchStopError` when
    YouTube asked us to back off before the video was done. `auth` is an auth spec as for `--auth`; without it the
    configured source applies.
    """
    ref = ids.classify(url)
    if ref is None or ref.kind is not ids.RefKind.VIDEO:
        raise ValueError(f"process takes one YouTube video, not {url!r}; use process_many for playlists and channels")
    results: list[VideoResult] = []
    report = process_many(
        [url],
        options,
        llm=llm,
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
    llm: LLMClient | None = None,
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

    Raises `ConfigError` for options that cannot run (an unknown engine, correction without an LLM),
    `DependencyError` when an engine's extra is missing, and refuses a cooldown (`BatchStopError`), all before any
    request; the batch itself is `fetch`'s, with this pipeline as its per-video hook. Per-video failures land in the
    report; a batch stop mid-run ends the batch and is recorded as well. An ASR worker killed by the memory watch
    raises `EngineMemoryError`, after the report of the videos before it is written: every later video would run the
    same engine into the same limit.
    """
    options = options or config.load_options()
    out = out if out is not None else paths.default_output_root()
    settings = _settings(options, llm, out, force, progress or _no_progress, clock)
    fetch_options = FetchOptions(
        policy=options.auth.policy,
        captions=captions or options.asr.source == "captions",
        cookie_channels=options.auth.cookie_channels,
        max_videos=options.ytdlp.max_videos_per_run,
        ignore_cooldown=ignore_cooldown,
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
    text: config.TextOptions
    source: Literal["asr", "captions"]
    definition: EngineDef
    second_opinion: EngineDef | None
    llm: LLMClient | None
    memory_cap_bytes: int | None
    out: Path
    force: bool
    emit: Callable[[ProgressEvent], None]
    clock: Callable[[], datetime]

    @property
    def llm_id(self) -> str | None:
        return self.llm.id if self.llm is not None else None

    def run_key(self) -> str:
        """Everything a finished document depends on besides the video itself; a change means a rerun."""
        return input_hash(
            "context",
            STAGE_VERSIONS["context"],
            {
                "versions": dict(STAGE_VERSIONS),
                "source": self.source,
                "engine": _engine_key(self.definition),
                "second_opinion": _engine_key(self.second_opinion) if self.second_opinion else None,
                "text": _plain(asdict(self.text)),
                "llm": self.llm_id,
            },
        )


def _settings(
    options: config.Options,
    llm: LLMClient | None,
    out: Path,
    force: bool,
    emit: Callable[[ProgressEvent], None],
    clock: Callable[[], datetime],
) -> _Settings:
    definition = registry.get(options.asr.engine)
    registry.check_dependencies(definition)
    second: EngineDef | None = None
    if options.text.numeric_second_opinion != "off":
        second = registry.get(options.text.numeric_second_opinion)
        if second.spec.max_chunk_s is None:
            raise ConfigError(
                f"text.numeric_second_opinion = {second.spec.id!r} decodes whole files; it needs a chunked engine "
                "that can re-decode just the numeric spans"
            )
        registry.check_dependencies(second)
    if llm is None and options.llm.provider != "none":
        llm = client_from_options(options.llm)
    if options.text.correct and llm is None:
        raise ConfigError("correction needs an LLM; pass --llm PROVIDER:MODEL or set [llm], or use --no-correct")
    memory_cap = asr_runner.memory_cap_bytes(options.asr.memory_cap_gib)
    return _Settings(options.text, options.asr.source, definition, second, llm, memory_cap, out, force, emit, clock)


def _engine_key(definition: EngineDef) -> JsonValue:
    return _plain(asdict(definition))


def _plain(value: object) -> JsonValue:
    return json.loads(json.dumps(value))


class _ContextMark(BaseModel):
    """Written after `context.json`: the document there was made with these settings and is this exact file."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sha256: str


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
    """Return the finished document, unless `--force` is set or it was made with other settings."""
    context = vdir / CONTEXT_JSON
    if settings.force or not context.is_file():
        return None
    mark = StageStore(vdir).load("context", STAGE_VERSIONS["context"], settings.run_key(), _ContextMark)
    if mark is None:
        return None
    data = context.read_bytes()
    if hashlib.sha256(data).hexdigest() != mark.sha256:
        return None
    return load_document(data.decode("utf-8"))


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
            if settings.force:
                StageStore(vdir).clear()
            return _VideoRun(session, planned, vdir, settings).run()
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
        case LLMError():
            return "llm"
        case ContractError():
            return "contract"
        case MediaValidationError():
            return "media_invalid"
        case ConfigError():
            return "config"
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


class _Stretches(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    stretches: tuple[Stretch, ...]


class _TextResult(BaseModel):
    """One text stage's output; `notice` says why the stage could not do its job (and is also a quality warning)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stretches: tuple[Stretch, ...]
    usage: tuple[LLMUsage, ...] = ()
    notice: str | None = None


class _Piece(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str
    start: float
    end: float


class _SecondOpinion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    engine: str | None
    pieces: tuple[_Piece, ...]


@dataclass
class _VideoRun:
    session: FetchSession
    planned: PlannedVideo
    vdir: Path
    settings: _Settings
    timings: list[StageTiming] = field(default_factory=list[StageTiming])
    notices: list[str] = field(default_factory=list[str])

    @property
    def video_id(self) -> str:
        return self.planned.video_id

    @property
    def store(self) -> StageStore:
        return StageStore(self.vdir)

    def run(self) -> VideoResult:
        settings = self.settings
        cached = self._local_metadata()
        need_audio = settings.source == "asr" and (cached is None or not self._asr_cached(cached))
        outcome = self._fetch(audio=need_audio)
        if outcome.status is not VideoStatus.OK or outcome.access is None:
            return VideoResult(outcome)
        metadata = load_metadata(self.vdir)
        tracks = saved_captions(self.vdir)

        source: Transcribed | None = None
        if settings.source == "captions":
            source = self._from_captions(tracks, metadata)
        if source is None:
            if not self._asr_cached(metadata) and not (self.vdir / WAV_NAME).is_file():
                outcome = self._fetch(audio=True)
                if outcome.status is not VideoStatus.OK or outcome.access is None:
                    return VideoResult(outcome)
            source = self._asr(metadata)
        second = self._second_opinion(source, metadata) if source.run.kind == "asr" else None
        (self.vdir / WAV_NAME).unlink(missing_ok=True)

        language = source.language
        normalized = self._normalize(source)
        punctuated = self._punctuate(normalized, language)
        corrected, correction_llm = self._correct(punctuated, language, metadata)
        verified = self._numeric(corrected, source, tracks, second)
        for result in (punctuated, verified):
            if result.notice is not None:
                self.notices.append(result.notice)

        self._emit("write")
        began = time.monotonic()
        document = build_document(
            metadata=metadata,
            tags=_tags(self.vdir),
            access=outcome.access,
            source=source,
            stretches=verified.stretches,
            chapters=settings.text.chapters != "none",
            created_at=settings.clock(),
            correction_llm=correction_llm,
            warnings=self.notices,
            llm_usage=(*punctuated.usage, *corrected.usage, *verified.usage),
        )
        markdown = render_markdown(document)
        dumped = dump_document(document)
        atomic.write_text(self.vdir / CONTEXT_MD, markdown)
        # context.json and then its mark go last: a marked context.json means the markdown beside it is complete.
        atomic.write_text(self.vdir / CONTEXT_JSON, dumped)
        sha = hashlib.sha256(dumped.encode("utf-8")).hexdigest()
        self.store.save("context", STAGE_VERSIONS["context"], settings.run_key(), _ContextMark(sha256=sha))
        self.timings.append(StageTiming("write", time.monotonic() - began, reused=False))
        return VideoResult(
            outcome,
            document,
            self.vdir / CONTEXT_MD,
            notices=tuple(self.notices),
            timings=tuple(self.timings),
        )

    def _emit(self, stage: str, detail: str = "") -> None:
        self.settings.emit(ProgressEvent(self.video_id, stage, detail))

    def _stage[M: BaseModel](
        self, stage: str, inputs: dict[str, JsonValue], model: type[M], compute: Callable[[], M], detail: str = ""
    ) -> M:
        key = input_hash(stage, STAGE_VERSIONS[stage], inputs)
        began = time.monotonic()

        def announced() -> M:
            self._emit(stage, detail)
            return compute()

        output, reused = self.store.run(stage, STAGE_VERSIONS[stage], key, model, announced)
        self.timings.append(StageTiming(stage, time.monotonic() - began, reused))
        return output

    def _fetch(self, *, audio: bool) -> VideoOutcome:
        self._emit("fetch")
        began = time.monotonic()
        outcome = self.session.fetch_video(self.video_id, cookies_first=self.planned.cookies_first, audio=audio)
        self.timings.append(StageTiming("fetch", time.monotonic() - began, reused=False))
        return outcome

    def _local_metadata(self) -> VideoMetadata | None:
        try:
            return load_metadata(self.vdir)
        except (OSError, ValueError, YtDlpError):
            return None

    def _asr_request(self, metadata: VideoMetadata) -> WorkerRequest:
        definition = self.settings.definition
        return WorkerRequest(
            engine=definition,
            out_dir=str(self.store.work_dir("asr") / definition.spec.id),
            files=(FileJob(id=self.video_id, wav=str(self.vdir / WAV_NAME), language=metadata.language),),
            memory_cap_bytes=self.settings.memory_cap_bytes,
        )

    def _asr_inputs(self, metadata: VideoMetadata) -> dict[str, JsonValue]:
        # Where the WAV and the output live does not change the decode; the video id names the audio.
        request = self._asr_request(metadata)
        fingerprint = request.model_dump(
            mode="json", exclude={"out_dir": True, "hf_hub_cache": True, "files": {"__all__": {"wav"}}}
        )
        return {"request": fingerprint}

    def _asr_cached(self, metadata: VideoMetadata) -> bool:
        key = input_hash("asr", STAGE_VERSIONS["asr"], self._asr_inputs(metadata))
        source = self.store.load("asr", STAGE_VERSIONS["asr"], key, Transcribed)
        if source is None:
            return False
        if self.settings.second_opinion is None:
            return True
        key = input_hash("second_opinion", STAGE_VERSIONS["second_opinion"], self._second_inputs(source))
        return self.store.load("second_opinion", STAGE_VERSIONS["second_opinion"], key, _SecondOpinion) is not None

    def _asr(self, metadata: VideoMetadata) -> Transcribed:
        definition = self.settings.definition

        def compute() -> Transcribed:
            result = asr_runner.run_worker(self._asr_request(metadata))
            return stages.from_asr_rows(_read_rows(result.hypotheses, self.video_id), definition, metadata)

        return self._stage("asr", self._asr_inputs(metadata), Transcribed, compute, definition.spec.id)

    def _from_captions(self, tracks: Sequence[CaptionTrack], metadata: VideoMetadata) -> Transcribed | None:
        track = _manual_track(tracks, metadata.language)
        if track is None:
            language = metadata.language or "the video's language"
            self.notices.append(f"no manual caption track in {language}; transcribed with ASR instead")
            return None
        inputs: dict[str, JsonValue] = {"track": _track_key(track), "language": metadata.language}
        return self._stage(
            "captions", inputs, Transcribed, lambda: stages.from_captions(track, metadata), track.language
        )

    def _second_inputs(self, source: Transcribed) -> dict[str, JsonValue]:
        second = self.settings.second_opinion
        return {"source": store.digest(source), "engine": _engine_key(second) if second else None}

    def _second_opinion(self, source: Transcribed, metadata: VideoMetadata) -> numeric.SecondOpinion | None:
        second = self.settings.second_opinion
        if second is None:
            return None

        def compute() -> _SecondOpinion:
            words = [word for stretch in source.stretches for word in stretch.words]
            request = numeric.second_opinion_request(
                _timed(source.stretches),
                engine=second,
                file_id=self.video_id,
                wav=self.vdir / WAV_NAME,
                out_dir=self.store.work_dir("second_opinion") / second.spec.id,
                language=metadata.language,
                words=words,
            )
            if request is None:
                return _SecondOpinion(engine=None, pieces=())
            result = asr_runner.run_worker(
                request.model_copy(update={"memory_cap_bytes": self.settings.memory_cap_bytes})
            )
            found = numeric.SecondOpinion.from_hypotheses(_read_rows(result.hypotheses, self.video_id))
            if found is None:
                return _SecondOpinion(engine=second.spec.id, pieces=())
            pieces = tuple(_Piece(text=p.text, start=p.start, end=p.end) for p in found.pieces)
            return _SecondOpinion(engine=found.engine, pieces=pieces)

        stored = self._stage("second_opinion", self._second_inputs(source), _SecondOpinion, compute, second.spec.id)
        if stored.engine is None or not stored.pieces:
            return None
        pieces = tuple(numeric.TimedText(p.text, p.start, p.end) for p in stored.pieces)
        return numeric.SecondOpinion(stored.engine, pieces)

    def _normalize(self, source: Transcribed) -> _Stretches:
        text = self.settings.text
        inputs: dict[str, JsonValue] = {"source": store.digest(source), "script": text.script, "fillers": text.fillers}

        def compute() -> _Stretches:
            normalized = stages.normalize(source.stretches, source.language, script=text.script, fillers=text.fillers)
            return _Stretches(stretches=normalized)

        return self._stage("normalize", inputs, _Stretches, compute)

    def _punctuate(self, previous: _Stretches, language: str) -> _TextResult:
        llm = self.settings.llm
        inputs: dict[str, JsonValue] = {"text": _text_key(previous.stretches), "llm": self.settings.llm_id}

        def compute() -> _TextResult:
            stretches = previous.stretches
            texts = [stretch.text for stretch in stretches]
            if not needs_punctuation(" ".join(texts), language):
                return _TextResult(stretches=stretches)
            if llm is None:
                notice = "punctuation not restored: the transcript has few sentence marks and no LLM is configured"
                return _TextResult(stretches=stretches, notice=notice)
            result = punctuate(texts, language=language, client=llm, checkpoint_dir=self.store.work_dir("punctuate"))
            failed = set(result.failed)
            note = "the LLM's punctuation changed the words twice, so this stretch was kept as transcribed"
            flags = [
                [stretch_flag(stretch, "punctuation_failed", note)] if index in failed else []
                for index, stretch in enumerate(stretches)
            ]
            return _TextResult(stretches=with_results(stretches, result.texts, result.edits, flags), usage=result.usage)

        return self._stage("punctuate", inputs, _TextResult, compute, self.settings.llm_id or "")

    def _correct(self, previous: _TextResult, language: str, metadata: VideoMetadata) -> tuple[_TextResult, str | None]:
        llm = self.settings.llm if self.settings.text.correct else None
        inputs: dict[str, JsonValue] = {"text": _text_key(previous.stretches), "llm": llm.id if llm else None}
        if llm is None:
            return _TextResult(stretches=previous.stretches), None

        def compute() -> _TextResult:
            stretches = previous.stretches
            note = metadata.creator_note.text if metadata.creator_note is not None else None
            context = CorrectionContext(glossary=_tags(self.vdir)[:_GLOSSARY_MAX_TERMS], creator_note=note)
            result = correct_segments(
                [stretch.text for stretch in stretches],
                client=llm,
                language=language,
                context=context,
                checkpoint_dir=self.store.work_dir("correct"),
                confidence=_confidence(stretches),
            )
            return _TextResult(
                stretches=with_results(stretches, result.texts, result.edits, result.flags), usage=result.usage
            )

        return self._stage("correct", inputs, _TextResult, compute, llm.id), llm.id

    def _numeric(
        self,
        previous: _TextResult,
        source: Transcribed,
        tracks: Sequence[CaptionTrack],
        second: numeric.SecondOpinion | None,
    ) -> _TextResult:
        llm = self.settings.llm
        evidence_tracks = _evidence_tracks(tracks, source.language)
        inputs: dict[str, JsonValue] = {
            "text": _text_key(previous.stretches),
            "captions": [_track_key(track) for track in evidence_tracks],
            "second_opinion": store.digest(_second_key(second)),
            "llm": self.settings.llm_id,
        }

        def compute() -> _TextResult:
            stretches = previous.stretches
            primary = source.run.id if source.run.kind == "asr" else stages.CAPTION_MANUAL
            evidence = numeric.Evidence(
                primary=primary,
                words=tuple(word for stretch in stretches for word in stretch.words),
                captions=tuple(evidence_tracks),
                second_opinion=second,
            )
            result = numeric.verify_numbers(
                _timed(stretches),
                language=source.language,
                evidence=evidence,
                client=llm,
                checkpoint_dir=self.store.work_dir("numeric"),
            )
            notice = None
            if llm is None and result.stats.disputed:
                notice = (
                    f"{result.stats.disputed} disputed number(s) left as transcribed and flagged: "
                    "no LLM is configured to choose among the readings"
                )
            return _TextResult(
                stretches=with_results(stretches, result.texts, result.edits, result.flags),
                usage=result.usage,
                notice=notice,
            )

        return self._stage("numeric", inputs, _TextResult, compute, self.settings.llm_id or "")


def _text_key(stretches: Sequence[Stretch]) -> str:
    # Only the text a stage reads; what the previous stage cost or noticed does not change this stage's output.
    return store.digest(_Stretches(stretches=tuple(stretches)))


def _timed(stretches: Sequence[Stretch]) -> list[numeric.TimedText]:
    return [numeric.TimedText(stretch.text, stretch.start, stretch.end) for stretch in stretches]


def _confidence(stretches: Sequence[Stretch]) -> list[list[float] | None] | None:
    confidence = [[word.conf for word in stretch.words if word.conf is not None] or None for stretch in stretches]
    return confidence if any(confidence) else None


def _manual_track(tracks: Sequence[CaptionTrack], language: str | None) -> CaptionTrack | None:
    manual = [track for track in tracks if track.kind is CaptionKind.MANUAL]
    if language is None:
        return manual[0] if len(manual) == 1 else None
    return next((track for track in manual if _base(track.language) == _base(language)), None)


def _evidence_tracks(tracks: Sequence[CaptionTrack], language: str) -> list[CaptionTrack]:
    # A track in another language is a translation; it would only make every number look unheard.
    return [track for track in tracks if _base(track.language) == _base(language)]


def _base(code: str) -> str:
    return code.split("-")[0].lower()


def _track_key(track: CaptionTrack) -> str:
    return store.digest(_plain(asdict(track)))


def _second_key(second: numeric.SecondOpinion | None) -> JsonValue:
    if second is None:
        return None
    return {"engine": second.engine, "pieces": [[p.text, p.start, p.end] for p in second.pieces]}


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
