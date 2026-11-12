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
import shutil
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
from ytverbatim.asr.models import FINGERPRINT_FILE, ChunkHypothesis, FileJob, WorkerRequest
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
from ytverbatim.llm import check_ready, client_from_options
from ytverbatim.llm.protocol import LLMClient
from ytverbatim.render import render_markdown
from ytverbatim.stages import Stretch, Transcribed, build_document, stretch_flag, with_results
from ytverbatim.store import StageStore, input_hash
from ytverbatim.store import lock as video_lock
from ytverbatim.text import clean, numeric
from ytverbatim.text import correct as correct_module
from ytverbatim.text import punctuate as punctuate_module
from ytverbatim.text.correct import CorrectionContext, correct_segments
from ytverbatim.text.punctuate import needs_punctuation, punctuate
from ytverbatim.ytdlp.runner import redact
from ytverbatim.ytdlp.types import YtDlpRunner

log = logging.getLogger(__name__)

CONTEXT_JSON = "context.json"
CONTEXT_MD = "context.md"

# The pipeline's own part of each stage's version: bump it when this module changes what a stage outputs for the same
# input. The text modules' versions are folded in by `stage_versions`, so a prompt, guard or rule change there redoes
# that stage, everything after it, and the context record. Worker folders are keyed by the decode request alone, so an
# `asr` or `second_opinion` bump re-reads the decoded rows they keep instead of decoding the audio again.
_PIPELINE_VERSIONS: dict[str, int] = {
    "asr": 2,
    "captions": 1,
    "second_opinion": 1,
    "normalize": 1,
    "punctuate": 2,
    "correct": 1,
    "numeric": 1,
    "context": 3,
}
_GLOSSARY_MAX_TERMS = 20


def stage_versions() -> dict[str, str]:
    """Each stored stage's version: the pipeline's own, plus the versions of the modules it runs."""
    modules = {
        "asr": [stages.VERSION],
        "captions": [stages.VERSION],
        "second_opinion": [numeric.PROMPT_VERSION],
        "normalize": [clean.VERSION, stages.VERSION],
        "punctuate": [punctuate_module.PROMPT_VERSION, stages.VERSION],
        "correct": [correct_module.PROMPT_VERSION, stages.VERSION],
        "numeric": [numeric.PROMPT_VERSION, stages.VERSION],
        "context": [stages.VERSION],
    }
    return {stage: "+".join([str(version), *modules[stage]]) for stage, version in _PIPELINE_VERSIONS.items()}


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
    captions = captions or options.asr.source == "captions"
    settings = _settings(options, llm, out, force=force, captions=captions, emit=progress or _no_progress, clock=clock)
    fetch_options = FetchOptions(
        policy=options.auth.policy,
        captions=captions,
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
    captions: bool
    emit: Callable[[ProgressEvent], None]
    clock: Callable[[], datetime]

    @property
    def llm_id(self) -> str | None:
        return self.llm.id if self.llm is not None else None

    def run_key(self) -> str:
        """Everything a finished document depends on besides the video itself; a change means a rerun."""
        versions = stage_versions()
        return input_hash(
            "context",
            versions["context"],
            {
                "versions": dict[str, JsonValue](versions),
                "source": self.source,
                "captions": self.captions,
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
    *,
    force: bool,
    captions: bool,
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
        # A missing `claude` CLI or base URL must fail now, not after an hour of ASR.
        check_ready(options.llm)
        llm = client_from_options(options.llm)
    if options.text.correct and llm is None:
        raise ConfigError("correction needs an LLM; pass --llm PROVIDER:MODEL or set [llm], or use --no-correct")
    memory_cap = asr_runner.memory_cap_bytes(options.asr.memory_cap_gib)
    return _Settings(
        options.text, options.asr.source, definition, second, llm, memory_cap, out, force, captions, emit, clock
    )


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
        except OSError:
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
    mark = StageStore(vdir).load("context", stage_versions()["context"], settings.run_key(), _ContextMark)
    if mark is None:
        return None
    data = context.read_bytes()
    if hashlib.sha256(data).hexdigest() != mark.sha256:
        return None
    try:
        return load_document(data.decode("utf-8"))
    except ContractError:
        # A document this reader can no longer load (a contract change) is rebuilt from the stored stages.
        return None


def _run_video(session: FetchSession, planned: PlannedVideo, settings: _Settings) -> VideoResult:
    video_id = planned.video_id
    vdir = paths.video_dir(video_id, settings.out)
    try:
        with video_lock(vdir):
            document = _reusable(vdir, settings)
            if document is not None:
                settings.emit(ProgressEvent(video_id, "skip", f"{CONTEXT_JSON} exists"))
                if not (vdir / CONTEXT_MD).is_file():
                    atomic.write_text(vdir / CONTEXT_MD, render_markdown(document))
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
    """One text stage's output; `notices` say what the stage could not do (and are also quality warnings).

    `usage` is what producing this output cost, batches reused from checkpoints included.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    stretches: tuple[Stretch, ...]
    usage: tuple[LLMUsage, ...] = ()
    notices: tuple[str, ...] = ()
    llm_punctuated: bool = False


class _Opinion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    opinion: numeric.SecondOpinion | None


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
        need_audio = settings.source == "asr" and (cached is None or self._needs_audio(cached))
        outcome = self._fetch(audio=need_audio)
        if outcome.status is not VideoStatus.OK or outcome.access is None:
            return VideoResult(outcome)
        metadata = load_metadata(self.vdir)
        tracks = saved_captions(self.vdir)

        source: Transcribed | None = None
        if settings.source == "captions":
            source = self._from_captions(tracks, metadata)
        if source is None:
            if self._needs_audio(metadata) and not (self.vdir / WAV_NAME).is_file():
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
        self.notices += [*punctuated.notices, *verified.notices]

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
            llm_punctuated=punctuated.llm_punctuated,
            warnings=self.notices,
            llm_usage=(*punctuated.usage, *corrected.usage, *verified.usage),
        )
        dumped = dump_document(document)
        # The record goes first and comes back last, so it never vouches for a half-written pair of files.
        self.store.discard("context")
        atomic.write_text(self.vdir / CONTEXT_MD, render_markdown(document))
        atomic.write_text(self.vdir / CONTEXT_JSON, dumped)
        sha = hashlib.sha256(dumped.encode("utf-8")).hexdigest()
        self.store.save("context", stage_versions()["context"], settings.run_key(), _ContextMark(sha256=sha))
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

    def _key(self, stage: str, inputs: dict[str, JsonValue]) -> str:
        return input_hash(stage, stage_versions()[stage], inputs)

    def _stage[M: BaseModel](
        self,
        stage: str,
        key: str,
        model: type[M],
        compute: Callable[[], M],
        detail: str = "",
        *,
        keep: Path | None = None,
    ) -> M:
        """Run or reuse one stage; `keep` is its worker folder, and the stage's other folders go once it is stored."""
        began = time.monotonic()

        def announced() -> M:
            self._emit(stage, detail)
            return compute()

        output, reused = self.store.run(stage, stage_versions()[stage], key, model, announced)
        if keep is not None and not reused:
            _prune(self.store.work_dir(stage), keep)
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

    def _placed(self, stage: str, request: WorkerRequest) -> WorkerRequest:
        """Point `request` at its worker folder in `stage`'s working directory.

        The folder is keyed by what the decode depends on and never by a stage version: a change to what the pipeline
        makes of the rows must not decode the audio again. The worker refuses a folder holding another request's
        output, and an interrupted run of the same request resumes in the same folder.
        """
        fingerprint = request.fingerprint()
        out_dir = self.store.work_dir(stage) / request.engine.spec.id / store.digest(_without_wav(fingerprint))[:16]
        _adopt(out_dir, fingerprint)
        return request.model_copy(update={"out_dir": str(out_dir)})

    def _asr_request(self, metadata: VideoMetadata) -> WorkerRequest:
        request = WorkerRequest(
            engine=self.settings.definition,
            out_dir=str(self.store.work_dir("asr")),
            files=(FileJob(id=self.video_id, wav=str(self.vdir / WAV_NAME), language=metadata.language),),
            memory_cap_bytes=self.settings.memory_cap_bytes,
        )
        return self._placed("asr", request)

    def _asr_key(self, request: WorkerRequest) -> str:
        # The worker's own fingerprint, minus the WAV path: the video id already names the audio.
        return self._key("asr", {"request": _without_wav(request.fingerprint())})

    def _transcribed(self, hypotheses: Path, metadata: VideoMetadata) -> Transcribed:
        return stages.from_asr_rows(_read_rows(hypotheses, self.video_id), self.settings.definition, metadata)

    def _needs_audio(self, metadata: VideoMetadata) -> bool:
        """Whether an ASR stage still has audio to decode; stored stages and finished worker folders need none."""
        request = self._asr_request(metadata)
        source = self.store.load("asr", stage_versions()["asr"], self._asr_key(request), Transcribed)
        if source is None:
            hypotheses = asr_runner.finished(request)
            if hypotheses is None:
                return True
            source = self._transcribed(hypotheses, metadata)
        if self.settings.second_opinion is None:
            return False
        key = self._key("second_opinion", self._second_inputs(source, metadata))
        if self.store.load("second_opinion", stage_versions()["second_opinion"], key, _Opinion) is not None:
            return False
        second = self._second_request(source, metadata)
        return second is not None and asr_runner.finished(second) is None

    def _asr(self, metadata: VideoMetadata) -> Transcribed:
        request = self._asr_request(metadata)

        def compute() -> Transcribed:
            hypotheses = asr_runner.finished(request) or asr_runner.run_worker(request).hypotheses
            return self._transcribed(hypotheses, metadata)

        engine = request.engine.spec.id
        return self._stage("asr", self._asr_key(request), Transcribed, compute, engine, keep=Path(request.out_dir))

    def _from_captions(self, tracks: Sequence[CaptionTrack], metadata: VideoMetadata) -> Transcribed | None:
        track, missing = _manual_track(tracks, metadata.language)
        if track is None:
            self.notices.append(f"{missing}; transcribed with ASR instead")
            return None
        key = self._key("captions", {"track": _track_key(track), "language": metadata.language})
        return self._stage("captions", key, Transcribed, lambda: stages.from_captions(track, metadata), track.language)

    def _second_inputs(self, source: Transcribed, metadata: VideoMetadata) -> dict[str, JsonValue]:
        second = self.settings.second_opinion
        return {
            "source": store.digest(source),
            "engine": _engine_key(second) if second else None,
            "language": metadata.language,
        }

    def _second_request(self, source: Transcribed, metadata: VideoMetadata) -> WorkerRequest | None:
        second = self.settings.second_opinion
        if second is None:
            return None
        request = numeric.second_opinion_request(
            _timed(source.stretches),
            engine=second,
            file_id=self.video_id,
            wav=self.vdir / WAV_NAME,
            out_dir=self.store.work_dir("second_opinion"),
            language=metadata.language,
            words=[word for stretch in source.stretches for word in stretch.words],
        )
        if request is None:
            return None
        request = request.model_copy(update={"memory_cap_bytes": self.settings.memory_cap_bytes})
        return self._placed("second_opinion", request)

    def _second_opinion(self, source: Transcribed, metadata: VideoMetadata) -> numeric.SecondOpinion | None:
        second = self.settings.second_opinion
        if second is None:
            return None
        key = self._key("second_opinion", self._second_inputs(source, metadata))
        request = self._second_request(source, metadata)

        def compute() -> _Opinion:
            if request is None:
                return _Opinion(opinion=None)
            hypotheses = asr_runner.finished(request) or asr_runner.run_worker(request).hypotheses
            return _Opinion(opinion=numeric.SecondOpinion.from_hypotheses(_read_rows(hypotheses, self.video_id)))

        keep = Path(request.out_dir) if request is not None else None
        return self._stage("second_opinion", key, _Opinion, compute, second.spec.id, keep=keep).opinion

    def _normalize(self, source: Transcribed) -> _Stretches:
        text = self.settings.text
        key = self._key("normalize", {"source": store.digest(source), "script": text.script, "fillers": text.fillers})

        def compute() -> _Stretches:
            normalized = stages.normalize(source.stretches, source.language, script=text.script, fillers=text.fillers)
            return _Stretches(stretches=normalized)

        return self._stage("normalize", key, _Stretches, compute)

    def _punctuate(self, previous: _Stretches, language: str) -> _TextResult:
        llm = self.settings.llm
        key = self._key("punctuate", {"text": _text_key(previous.stretches), "llm": self.settings.llm_id})

        def compute() -> _TextResult:
            stretches = previous.stretches
            texts = [stretch.text for stretch in stretches]
            # Per stretch: Whisper often leaves only the opening minute unpunctuated in an otherwise punctuated video.
            if not any(needs_punctuation(text, language) for text in texts):
                return _TextResult(stretches=stretches)
            if llm is None:
                notice = "punctuation not restored: some stretches have few sentence marks and no LLM is configured"
                return _TextResult(stretches=stretches, notices=(notice,))
            result = punctuate(texts, language=language, client=llm, checkpoint_dir=self.store.work_dir("punctuate"))
            failed = set(result.failed)
            note = "the LLM's punctuation changed the words twice, so this stretch was kept as transcribed"
            flags = [
                [stretch_flag(stretch, "punctuation_failed", note)] if index in failed else []
                for index, stretch in enumerate(stretches)
            ]
            return _TextResult(
                stretches=with_results(stretches, result.texts, result.edits, flags, record=False),
                usage=result.usage,
                llm_punctuated=any(result.edits),
            )

        return self._stage("punctuate", key, _TextResult, compute, self.settings.llm_id or "")

    def _correct(self, previous: _TextResult, language: str, metadata: VideoMetadata) -> tuple[_TextResult, str | None]:
        llm = self.settings.llm if self.settings.text.correct else None
        if llm is None:
            return _TextResult(stretches=previous.stretches), None
        context = correction_context(metadata, _tags(self.vdir))
        inputs: dict[str, JsonValue] = {
            "text": _text_key(previous.stretches),
            "llm": self.settings.llm_id,
            "glossary": list(context.glossary),
            "creator_note": context.creator_note,
        }

        def compute() -> _TextResult:
            stretches = previous.stretches
            result = correct_segments(
                [stretch.text for stretch in stretches],
                client=llm,
                language=language,
                context=context,
                checkpoint_dir=self.store.work_dir("correct"),
                confidence=stages.word_confidence(stretches),
            )
            return _TextResult(
                stretches=with_results(stretches, result.texts, result.edits, result.flags), usage=result.usage
            )

        return self._stage("correct", self._key("correct", inputs), _TextResult, compute, llm.id), llm.id

    def _numeric(
        self,
        previous: _TextResult,
        source: Transcribed,
        tracks: Sequence[CaptionTrack],
        second: numeric.SecondOpinion | None,
    ) -> _TextResult:
        llm = self.settings.llm
        evidence_tracks = _evidence_tracks(tracks, source)
        inputs: dict[str, JsonValue] = {
            "text": _text_key(previous.stretches),
            "captions": [_track_key(track) for track in evidence_tracks],
            "second_opinion": store.digest(_Opinion(opinion=second)),
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
            notices: list[str] = []
            if not evidence_tracks and second is None and result.stats.single_source:
                notices.append(
                    f"numbers not cross-checked: no captions in {source.language} and no second opinion; "
                    f"{result.stats.single_source} number(s) rest on the transcript alone"
                )
            if llm is None and result.stats.disputed:
                notices.append(
                    f"{result.stats.disputed} disputed number(s) left as transcribed and flagged: "
                    "no LLM is configured to choose among the readings"
                )
            return _TextResult(
                stretches=with_results(stretches, result.texts, result.edits, result.flags),
                usage=result.usage,
                notices=tuple(notices),
            )

        return self._stage("numeric", self._key("numeric", inputs), _TextResult, compute, self.settings.llm_id or "")


def _adopt(out_dir: Path, fingerprint: dict[str, JsonValue]) -> None:
    """Move a folder holding this exact request's output, left under another name, to `out_dir`.

    Folders were once keyed by the stage version, so a version bump would otherwise decode the audio again.
    """
    if out_dir.exists() or not out_dir.parent.is_dir():
        return
    for entry in out_dir.parent.iterdir():
        try:
            stored: object = json.loads((entry / FINGERPRINT_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if stored == fingerprint:
            entry.rename(out_dir)
            return


def _prune(root: Path, keep: Path) -> None:
    """Remove the stale requests of `keep`'s engine; its other entries can never be resumed again.

    Other engines' folders stay: switching back to one resumes its finished chunks instead of decoding again.
    """
    for entry in keep.parent.iterdir():
        if entry != keep:
            _remove(entry)
    for entry in root.iterdir():
        if not entry.is_dir():
            _remove(entry)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _without_wav(fingerprint: dict[str, JsonValue]) -> dict[str, JsonValue]:
    files = fingerprint.get("files")
    if not isinstance(files, list):
        return fingerprint
    kept: list[JsonValue] = [
        {k: v for k, v in job.items() if k != "wav"} if isinstance(job, dict) else job for job in files
    ]
    return {**fingerprint, "files": kept}


def _text_key(stretches: Sequence[Stretch]) -> str:
    # Only the text a stage reads; what the previous stage cost or noticed does not change this stage's output.
    return store.digest(_Stretches(stretches=tuple(stretches)))


def _timed(stretches: Sequence[Stretch]) -> list[numeric.TimedText]:
    return [numeric.TimedText(stretch.text, stretch.start, stretch.end) for stretch in stretches]


def _manual_track(tracks: Sequence[CaptionTrack], language: str | None) -> tuple[CaptionTrack | None, str]:
    """Pick the manual track to transcribe from; without one, say why there is none."""
    manual = [track for track in tracks if track.kind is CaptionKind.MANUAL]
    if language is None:
        if len(manual) == 1:
            return manual[0], ""
        if manual:
            return None, f"{len(manual)} manual caption tracks and no declared video language to choose one"
        return None, "no manual caption track"
    track = next((track for track in manual if _base(track.language) == _base(language)), None)
    return track, "" if track is not None else f"no manual caption track in {language}"


def _evidence_tracks(tracks: Sequence[CaptionTrack], source: Transcribed) -> list[CaptionTrack]:
    """Caption tracks that are independent readings of the transcript's numbers."""
    # A track in another language is a translation; it would only make every number look unheard. The manual track a
    # captions-mode transcript was built from is the transcript itself.
    return [
        track
        for track in tracks
        if _base(track.language) == _base(source.language)
        and not (source.run.kind != "asr" and track.kind is CaptionKind.MANUAL)
    ]


def _base(code: str) -> str:
    return code.split("-")[0].lower()


def _track_key(track: CaptionTrack) -> str:
    return store.digest(_plain(asdict(track)))


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


def correction_context(metadata: VideoMetadata, tags: Sequence[str]) -> CorrectionContext:
    note = metadata.creator_note.text if metadata.creator_note is not None else None
    return CorrectionContext(glossary=tuple(tags[:_GLOSSARY_MAX_TERMS]), creator_note=note)


def info_tags(raw: object) -> tuple[str, ...]:
    tags = raw.get("tags") if isinstance(raw, dict) else None
    return tuple(tag for tag in tags if isinstance(tag, str)) if isinstance(tags, list) else ()


def _tags(vdir: Path) -> tuple[str, ...]:
    # VideoMetadata does not keep tags, but the raw extraction does.
    try:
        raw: object = json.loads(info_path(vdir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    return info_tags(raw)
