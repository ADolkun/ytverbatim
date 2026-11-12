"""Correction-LLM eval runs: one shared base per file, then one run of the production correction per candidate.

The base is the input hypothesis after the same normalize and punctuate stages `ytverbatim run` applies. Every
candidate corrects exactly that text through `correct_segments`, with the context and word confidences the pipeline
would pass. Outputs record the versions and the base they were made from, and a mismatch refuses instead of mixing
configurations. Everything is written under ``$YTVERBATIM_EVAL_DIR/correction``.
"""

import contextlib
import importlib
import importlib.util
import json
import os
import random
import socket
import sys
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, ValidationError
from rapidfuzz.distance import Levenshtein

from evals import disk
from evals.consensus import TimedUnit
from evals.manifest import Manifest, Video
from evals.normalize import Language, to_units
from evals.screen import DISPUTE_VIEW, Span, window_spans
from evals.session import SessionKey, ShownSpan, session_from_spans
from evals.store import (
    Layout,
    Plan,
    file_complete,
    hypothesis_units,
    now,
    require_plan,
    row_units,
    rows_by_file,
    videos,
    write_json,
)
from ytverbatim import config, pipeline, proc, stages
from ytverbatim.acquire.metadata import VideoMetadata, trim_info
from ytverbatim.asr import registry
from ytverbatim.asr.models import ChunkHypothesis
from ytverbatim.asr.runner import default_memory_cap_bytes, worker_lock
from ytverbatim.contract.models import Edit, Flag, LLMUsage, Word
from ytverbatim.errors import (
    ConfigError,
    DependencyError,
    EngineError,
    EngineMemoryError,
    LLMOutputError,
    ProcMemoryExceeded,
)
from ytverbatim.llm import OpenAICompatClient, openai_compat
from ytverbatim.llm.protocol import LLMClient, LLMRequest, LLMResponse, Readiness
from ytverbatim.stages import Stretch
from ytverbatim.store import digest
from ytverbatim.text import clean
from ytverbatim.text import correct as correct_module
from ytverbatim.text import punctuate as punctuate_module
from ytverbatim.text.correct import Rejection, correct_segments
from ytverbatim.text.punctuate import needs_punctuation, punctuate

REPO_ROOT = Path(__file__).resolve().parent.parent
CALLS_FILE = "calls.jsonl"
FAILURE_FILE = "failure.json"
SERVER_LOG = "server.log"
SERVER_RECORD = "server.json"

# Reading names inside the eval's own disputes; a span never reaches the judge or the listener under these.
BASE = "base"
BEFORE = "before"
PROPOSED = "proposed"
# Supplies word times when the input hypothesis has none (Qwen3 rows); a production-control run always exists.
TIMING_DONOR = "w-prod"
SESSION_NAME = "correction"
SESSION_N = 80
REJECTIONS_PER_GUARD = 30
REJECTIONS_TOTAL = 150

# mlx_lm.server loads whichever model a request names, so any other string would trigger a reload; this alias always
# means the one the server was started with.
SERVER_MODEL = "default_model"
READY_TIMEOUT_S = 600.0
CLIENT_TIMEOUT_S = 900.0
_LOG_TAIL_LINES = 15
_SERVER_ENV_KEYS = ("PATH", "HOME", "TMPDIR", "LANG")
_SERVER_FLAGS = (
    "--chat-template-args", '{"enable_thinking":false}',
    "--prompt-cache-size", "1",
    "--decode-concurrency", "1",
    "--prompt-concurrency", "1",
    "--max-tokens", "8192",
    "--log-level", "WARNING",
)  # fmt: skip


@dataclass(frozen=True)
class Candidate:
    """A correction model: a Claude CLI model id, or (with ``revision``) a Hugging Face repo served locally."""

    id: str
    model: str
    revision: str | None = None

    @property
    def local(self) -> bool:
        return self.revision is not None

    @property
    def client_id(self) -> str:
        """The id checkpoints and `Edit.llm` carry; for a local model it names the weights, not the server's host."""
        if self.revision is None:
            return f"claude-cli:{self.model}"
        return f"mlx:{self.model}@{self.revision[:12]}"

    @property
    def weights(self) -> disk.Weights:
        if self.revision is None:
            raise ConfigError(f"{self.id} runs through the Claude CLI and has no local weights")
        return disk.Weights(self.model, self.revision)


CANDIDATES: dict[str, Candidate] = {
    c.id: c
    for c in (
        Candidate("c-sonnet", "claude-sonnet-5-5"),
        Candidate("c-haiku", "claude-haiku-5-5"),
        Candidate("l-qwen35-9b", "mlx-community/Qwen3.5-9B-8bit", "16daa4818c54ce5f5436f929d52542eb65bbed9d"),
        Candidate("l-qwen3-8b", "mlx-community/Qwen3-8B-8bit", "48a0b75b1ae72503e21e1558d040bc227510ff06"),
    )
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BaseFile(_Frozen):
    """One file's stretches after normalize and punctuate: the text every candidate corrects."""

    file: str
    hypothesis: str
    punctuation_llm: str
    versions: dict[str, str]
    language: str
    # Digest of the normalized stretches punctuation started from; it catches an input hypothesis that changed.
    input_digest: str
    stretches: tuple[Stretch, ...]
    usage: tuple[LLMUsage, ...]

    @property
    def text_digest(self) -> str:
        return digest({"language": self.language, "stretches": [s.model_dump(mode="json") for s in self.stretches]})


class CandidateFile(_Frozen):
    """One candidate's correction of one file's base, aligned with the base stretches."""

    file: str
    candidate: str
    client: str
    base_digest: str
    versions: dict[str, str]
    texts: tuple[str, ...]
    edits: tuple[tuple[Edit, ...], ...]
    flags: tuple[tuple[Flag, ...], ...]
    rejections: tuple[Rejection, ...]
    usage: tuple[LLMUsage, ...]
    batches: int

    @property
    def calls(self) -> int:
        return sum(u.calls for u in self.usage)


class Failure(_Frozen):
    file: str
    error: str
    at: str


class ServerRecord(_Frozen):
    candidate: str
    model: str
    revision: str
    cap_bytes: int
    peak_footprint_bytes: int | None


def first_try_validity(files: Sequence[CandidateFile]) -> float | None:
    """Share of batches whose first reply was valid JSON; ``complete_json`` calls twice only after an invalid one."""
    batches = sum(f.batches for f in files)
    if not batches:
        return None
    return 1 - (sum(f.calls for f in files) - batches) / batches


def base_dir(store: Layout) -> Path:
    return store.correction / "base"


def base_path(store: Layout, file: str) -> Path:
    return base_dir(store) / f"{file}.json"


def run_dir(store: Layout, candidate: Candidate) -> Path:
    return store.correction / "runs" / candidate.id


def load_failure(store: Layout, candidate: Candidate) -> Failure | None:
    directory = run_dir(store, candidate)
    return _load(directory / FAILURE_FILE, Failure, directory)


def candidate_path(store: Layout, candidate: Candidate, file: str) -> Path:
    return run_dir(store, candidate) / f"{file}.json"


def base_versions() -> dict[str, str]:
    return {"stages": stages.VERSION, "clean": clean.VERSION, "punctuate": punctuate_module.PROMPT_VERSION}


def candidate_versions() -> dict[str, str]:
    return {**base_versions(), "correct": correct_module.PROMPT_VERSION}


def _metadata(store: Layout, video: Video) -> tuple[dict[str, object], VideoMetadata]:
    raw: dict[str, object] = json.loads((store.root / video.meta).read_text(encoding="utf-8"))
    return raw, trim_info(raw)


def _load[M: BaseModel](path: Path, model: type[M], directory: Path) -> M | None:
    if not path.exists():
        return None
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ConfigError(
            f"{path.name} does not match the current {model.__name__} format ({exc.error_count()} errors); "
            f"move {directory} aside to redo it"
        ) from exc


def _refuse(directory: Path, file: str, what: str, stored: object, current: object) -> ConfigError:
    return ConfigError(
        f"{file} in {directory} was made with {what} {stored!r}, but this run uses {current!r}; "
        f"move {directory} aside to redo it, or rerun with the original setting"
    )


@dataclass(frozen=True)
class BaseOutcome:
    built: tuple[BaseFile, ...]
    reused: tuple[str, ...]


def build_base(
    store: Layout,
    manifest: Manifest,
    hypothesis: str,
    files: Sequence[str],
    client: LLMClient,
) -> BaseOutcome:
    """Write ``base/<file>.json`` for each file that has none; a rerun of a finished file makes no call."""
    definition = registry.get(hypothesis)
    text = config.TextOptions()
    built: list[BaseFile] = []
    reused: list[str] = []
    done = rows_by_file(store, hypothesis)
    for video in videos(manifest, files):
        rows = done.get(video.id, [])
        if not file_complete(definition, require_plan(store, video.id), rows):
            raise ConfigError(
                f"{hypothesis} has no complete output for {video.id}; "
                f"run `python -m evals.run_asr asr --hyp {hypothesis}` first"
            )
        _, metadata = _metadata(store, video)
        transcribed = stages.from_asr_rows(rows, definition, metadata)
        normalized = stages.normalize(
            transcribed.stretches, transcribed.language, script=text.script, fillers=text.fillers
        )
        input_digest = digest([s.model_dump(mode="json") for s in normalized])
        existing = _load(base_path(store, video.id), BaseFile, base_dir(store))
        if existing is not None:
            _check_base(store, existing, hypothesis, client.id, input_digest)
            reused.append(video.id)
            continue
        built.append(_write_base(store, video.id, hypothesis, client, transcribed.language, normalized, input_digest))
    return BaseOutcome(tuple(built), tuple(reused))


def _check_base(store: Layout, base: BaseFile, hypothesis: str, punctuation_llm: str, input_digest: str) -> None:
    directory = base_dir(store)
    for what, stored, current in (
        ("input hypothesis", base.hypothesis, hypothesis),
        ("punctuation LLM", base.punctuation_llm, punctuation_llm),
        ("stage versions", base.versions, base_versions()),
        ("input text", base.input_digest, input_digest),
    ):
        if stored != current:
            raise _refuse(directory, base.file, what, stored, current)


def _write_base(
    store: Layout,
    file: str,
    hypothesis: str,
    client: LLMClient,
    language: str,
    normalized: Sequence[Stretch],
    input_digest: str,
) -> BaseFile:
    stretches = tuple(normalized)
    usage: tuple[LLMUsage, ...] = ()
    texts = [s.text for s in stretches]
    if any(needs_punctuation(t, language) for t in texts):
        result = punctuate(texts, language=language, client=client, checkpoint_dir=base_dir(store) / "ckpt" / file)
        stretches = stages.with_results(stretches, result.texts, result.edits, record=False)
        usage = result.usage
    base = BaseFile(
        file=file,
        hypothesis=hypothesis,
        punctuation_llm=client.id,
        versions=base_versions(),
        language=language,
        input_digest=input_digest,
        stretches=stretches,
        usage=usage,
    )
    write_json(base_path(store, file), base)
    return base


def load_base(store: Layout, file: str) -> BaseFile:
    base = _load(base_path(store, file), BaseFile, base_dir(store))
    if base is None:
        raise ConfigError(f"no base for {file}; run `python -m evals.run_correction base --hyp H` first")
    return base


class RecordingClient:
    """Reports ``id`` instead of the wrapped client's and appends every reply's time and tokens to ``calls.jsonl``."""

    def __init__(self, inner: LLMClient, *, client_id: str, calls: Path, file: str) -> None:
        self._inner = inner
        self._id = client_id
        self._calls = calls
        self._file = file

    @property
    def id(self) -> str:
        return self._id

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._inner.complete(request)
        usage = response.usage
        line = {
            "file": self._file,
            "elapsed_s": round(response.elapsed_s, 3),
            "input_tokens": None if usage is None else usage.input_tokens,
            "output_tokens": None if usage is None else usage.output_tokens,
        }
        self._calls.parent.mkdir(parents=True, exist_ok=True)
        with self._calls.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        return response


@dataclass(frozen=True)
class CandidateOutcome:
    corrected: tuple[CandidateFile, ...]
    reused: tuple[str, ...]


def load_candidate_file(store: Layout, candidate: Candidate, file: str) -> CandidateFile | None:
    """Return the stored run of one file; refuse if it was made from another base, version or model."""
    stored = _load(candidate_path(store, candidate, file), CandidateFile, run_dir(store, candidate))
    if stored is None:
        return None
    directory = run_dir(store, candidate)
    for what, was, now_ in (
        ("client", stored.client, candidate.client_id),
        ("stage versions", stored.versions, candidate_versions()),
        ("base text", stored.base_digest, load_base(store, file).text_digest),
    ):
        if was != now_:
            raise _refuse(directory, file, what, was, now_)
    return stored


def pending_files(store: Layout, manifest: Manifest, candidate: Candidate, files: Sequence[str]) -> list[str]:
    return [v.id for v in videos(manifest, files) if load_candidate_file(store, candidate, v.id) is None]


def run_candidate(
    store: Layout, manifest: Manifest, candidate: Candidate, files: Sequence[str], client: LLMClient
) -> CandidateOutcome:
    """Correct each file's base with ``client`` through the pipeline's own call; finished files are skipped.

    A batch whose reply is invalid twice raises `LLMOutputError`; it is recorded in ``failure.json`` first, because
    that candidate would fail the JSON gate in production too.
    """
    corrected: list[CandidateFile] = []
    reused: list[str] = []
    for video in videos(manifest, files):
        if load_candidate_file(store, candidate, video.id) is not None:
            reused.append(video.id)
            continue
        base = load_base(store, video.id)
        raw, metadata = _metadata(store, video)
        recording = RecordingClient(
            client,
            client_id=candidate.client_id,
            calls=run_dir(store, candidate) / CALLS_FILE,
            file=video.id,
        )
        try:
            result = correct_segments(
                [s.text for s in base.stretches],
                client=recording,
                language=base.language,
                context=pipeline.correction_context(metadata, pipeline.info_tags(raw)),
                checkpoint_dir=run_dir(store, candidate) / "ckpt" / video.id,
                confidence=stages.word_confidence(base.stretches),
            )
        except LLMOutputError as exc:
            write_json(run_dir(store, candidate) / FAILURE_FILE, Failure(file=video.id, error=str(exc), at=now()))
            raise
        done = CandidateFile(
            file=video.id,
            candidate=candidate.id,
            client=candidate.client_id,
            base_digest=base.text_digest,
            versions=candidate_versions(),
            texts=result.texts,
            edits=result.edits,
            flags=result.flags,
            rejections=result.rejections,
            usage=result.usage,
            batches=result.batches,
        )
        write_json(candidate_path(store, candidate, video.id), done)
        corrected.append(done)
    return CandidateOutcome(tuple(corrected), tuple(reused))


def cleanup_candidate(
    store: Layout,
    manifest: Manifest,
    candidate: Candidate,
    cache: Path,
    *,
    delete: Callable[[Path, str], int] = disk.delete_repo,
) -> int:
    """Delete a finished local candidate's weights from the candidate cache; return the bytes freed."""
    weights = candidate.weights
    disk.require_candidate_cache(cache)
    missing = pending_files(store, manifest, candidate, [])
    if missing:
        raise ConfigError(
            f"{candidate.id} has no output for {len(missing)} file(s) ({', '.join(missing[:3])}…); "
            f"finish it with `python -m evals.run_correction correct --candidate {candidate.id}` first"
        )
    return delete(cache, weights.repo)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _require_mlx_lm() -> None:
    if importlib.util.find_spec("mlx_lm") is None:
        raise DependencyError("the local candidate needs mlx-lm; run `uv sync --group local-llm`")


def _snapshot_download(repo: str, revision: str, cache: Path) -> Path:
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise DependencyError("downloading weights needs huggingface_hub; run `uv sync --group local-llm`") from exc
    return Path(cast(str, hub.snapshot_download(repo_id=repo, revision=revision, cache_dir=str(cache))))


@dataclass(frozen=True)
class ServerSeams:
    """Everything `local_server` reaches outside this process through, so tests can replace each piece."""

    free: Callable[[Path], int] = disk.free_bytes
    download: Callable[[str, str, Path], Path] = _snapshot_download
    runtime: Callable[[], None] = _require_mlx_lm
    lock: Callable[[], AbstractContextManager[None]] = worker_lock
    background: Callable[..., AbstractContextManager[proc.Background]] = proc.background
    ready: Callable[[str], Readiness] = openai_compat.readiness
    port: Callable[[], int] = _free_port
    poll_s: float = 1.0
    ready_timeout_s: float = READY_TIMEOUT_S


DEFAULT_SEAMS = ServerSeams()


@contextlib.contextmanager
def local_server(
    candidate: Candidate,
    cache: Path,
    directory: Path,
    *,
    memory_cap: int | None = None,
    seams: ServerSeams = DEFAULT_SEAMS,
) -> Generator[OpenAICompatClient]:
    """Serve a local candidate on a loopback port for the block and yield a client for it.

    The MLX worker lock is held from before the server starts until it is gone, so no ASR worker shares unified
    memory with it. The server's footprint is watched against ``memory_cap`` (default: half of memory); a breach
    becomes `EngineMemoryError` and the batches already checkpointed stay.
    """
    weights = candidate.weights
    seams.runtime()
    disk.require_candidate_cache(cache)
    disk.check_weights([weights], cache, candidate.id, free=seams.free)
    snapshot = (
        disk.repo_dir(cache, weights.repo) / "snapshots" / weights.revision
        if disk.present(cache, weights)
        else seams.download(weights.repo, weights.revision, cache)
    )
    cap = memory_cap or default_memory_cap_bytes()
    port = seams.port()
    base_url = f"http://127.0.0.1:{port}/v1"
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / SERVER_LOG
    argv = [
        sys.executable, "-m", "evals.mlx_server", "--memory-cap-bytes", str(cap), "--",
        "--model", str(snapshot), "--host", "127.0.0.1", "--port", str(port), *_SERVER_FLAGS,
    ]  # fmt: skip
    try:
        with (
            seams.lock(),
            seams.background(
                argv, log=log, memory_watch=proc.MemoryWatch(cap), cwd=REPO_ROOT, env=_server_env(cache)
            ) as server,
        ):
            try:
                _await_ready(server, base_url, log, seams)
                yield OpenAICompatClient(base_url=base_url, model=SERVER_MODEL, timeout_s=CLIENT_TIMEOUT_S)
            finally:
                write_json(
                    directory / SERVER_RECORD,
                    ServerRecord(
                        candidate=candidate.id,
                        model=weights.repo,
                        revision=weights.revision,
                        cap_bytes=cap,
                        peak_footprint_bytes=server.peak_footprint_bytes,
                    ),
                )
    except ProcMemoryExceeded as exc:
        raise EngineMemoryError(
            f"{candidate.id} server stopped to protect the machine: {exc}", _memory_fix(exc), exc
        ) from exc


def _await_ready(server: proc.Background, base_url: str, log: Path, seams: ServerSeams) -> None:
    deadline = time.monotonic() + seams.ready_timeout_s
    while True:
        if not server.alive():
            raise EngineError(f"the local server exited with code {server.returncode} before it was ready{_tail(log)}")
        if seams.ready(base_url).ready:
            return
        if time.monotonic() > deadline:
            raise EngineError(
                f"the local server did not answer within {seams.ready_timeout_s:g}s; "
                f"see {log}, or try the smaller l-qwen3-8b{_tail(log)}"
            )
        time.sleep(seams.poll_s)


def _tail(log: Path) -> str:
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-_LOG_TAIL_LINES:]
    except OSError:
        return ""
    return f"\n--- {log} (last lines) ---\n" + "\n".join(lines) if lines else ""


def _memory_fix(exc: ProcMemoryExceeded) -> str:
    resume = "finished batches are kept and the next run resumes from them"
    if exc.trigger == "available":
        return f"close memory-heavy apps, then run again ({resume})"
    return (
        "raise asr.memory_cap_gib in the config file (14 suits this model) only if this Mac has that much memory to "
        f"spare, or run the smaller l-qwen3-8b ({resume})"
    )


def _server_env(cache: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in _SERVER_ENV_KEYS if k in os.environ}
    env.update({k: v for k, v in os.environ.items() if k.startswith("LC_")})
    env.update({"HF_HUB_CACHE": str(cache), "HF_HUB_OFFLINE": "1"})
    return env


class RejectionSpan(_Frozen):
    """One dispute inside a rejected proposal: readings ``before`` and ``proposed``, from one candidate's guard."""

    candidate: str
    guard: str
    # Position in that candidate file's ``rejections``; the disputes of one proposal share it.
    rejection: int
    span: Span


def complete_candidates(store: Layout, manifest: Manifest) -> list[Candidate]:
    """Candidates with a finished run for every manifest file."""
    return [c for c in CANDIDATES.values() if not pending_files(store, manifest, c, [])]


def _require_run(store: Layout, candidate: Candidate, file: str) -> CandidateFile:
    run = load_candidate_file(store, candidate, file)
    if run is None:
        raise ConfigError(
            f"{candidate.id} has no output for {file}; "
            f"run `python -m evals.run_correction correct --candidate {candidate.id}` first"
        )
    return run


@dataclass(frozen=True)
class _Timing:
    """A file's base with the word times every pseudo-row of its stretches carries."""

    video: Video
    plan: Plan
    language: Language
    base: BaseFile
    words: list[tuple[Word, ...]]

    def units(self, name: str, index: int, text: str) -> list[TimedUnit]:
        """Units of ``text`` as one hypothesis's reading of base stretch ``index``, timed by that stretch's words."""
        stretch = self.base.stretches[index]
        row = ChunkHypothesis(
            hypothesis=name,
            file=self.video.id,
            chunk=index,
            start=stretch.start,
            end=stretch.end,
            text=text,
            words=self.words[index],
        )
        return row_units(row, self.language, DISPUTE_VIEW, self.plan.speech)


def _timing(store: Layout, video: Video, base: BaseFile) -> _Timing:
    """Word times per stretch; stretches whose source had none borrow `TIMING_DONOR`'s by alignment.

    The donor's units are aligned to the base's across the whole file rather than per stretch, because a stretch
    cut from untimed rows has only approximate bounds to select donor words by.
    """
    plan, language = require_plan(store, video.id), video.view_language
    stretches = base.stretches
    words = [s.words for s in stretches]
    lacking = {i for i, s in enumerate(stretches) if not s.word_timed}
    if not lacking:
        return _Timing(video, plan, language, base, words)
    rows = rows_by_file(store, TIMING_DONOR).get(video.id, [])
    if not rows or not file_complete(registry.get(TIMING_DONOR), plan, rows):
        raise ConfigError(
            f"{base.hypothesis} has no word timings for {video.id}, so its edits are timed from {TIMING_DONOR}; "
            f"run `python -m evals.run_asr asr --hyp {TIMING_DONOR}` first"
        )
    donor = hypothesis_units(rows, language, DISPUTE_VIEW, plan.speech)
    units = [(i, unit) for i, s in enumerate(stretches) for unit in to_units(s.text, language, DISPUTE_VIEW)]
    timed: dict[int, TimedUnit] = {}
    for op in Levenshtein.opcodes([u.text for u in donor], [unit for _, unit in units]):
        if op.tag in ("equal", "replace"):
            for offset in range(min(op.src_end - op.src_start, op.dest_end - op.dest_start)):
                timed[op.dest_start + offset] = donor[op.src_start + offset]
    borrowed: dict[int, list[Word]] = {i: [] for i in lacking}
    for position, (i, unit) in enumerate(units):
        if i in borrowed and position in timed:
            borrowed[i].append(Word(text=unit, start=timed[position].start_s, end=timed[position].end_s))
    for i, found in borrowed.items():
        words[i] = tuple(found)
    return _Timing(video, plan, language, base, words)


def edit_spans(
    store: Layout, manifest: Manifest, candidates: Sequence[Candidate], files: Sequence[str] = ()
) -> list[Span]:
    """Disputes between the base and every candidate's text, with the base as pivot.

    Only stretches where some candidate's text differs are aligned, so an unchanged stretch gives none. Every span
    carries a reading for `BASE` and for each candidate id, and is numbered ``<file>#<stretch>#<n>``.
    """
    if not candidates:
        raise ConfigError("no candidate to compare with the base")
    spans: list[Span] = []
    for video in videos(manifest, files):
        base = load_base(store, video.id)
        runs = {c.id: _require_run(store, c, video.id) for c in candidates}
        changed = [i for i, s in enumerate(base.stretches) if any(run.texts[i] != s.text for run in runs.values())]
        if not changed:
            continue
        timing = _timing(store, video, base)
        for i in changed:
            units = {BASE: timing.units(BASE, i, base.stretches[i].text)}
            units.update({cid: timing.units(cid, i, run.texts[i]) for cid, run in runs.items()})
            spans += window_spans(video, timing.language, i, units, pivot=BASE)
    return spans


def audible_edits(spans: Sequence[Span], candidate: str) -> list[Span]:
    """Return the disputes where ``candidate`` changed the base's words in a way a listener can hear."""
    return [s for s in spans if not s.format_only and s.reading(candidate).text != s.reading(BASE).text]


def rejection_spans(
    store: Layout,
    manifest: Manifest,
    candidates: Sequence[Candidate],
    *,
    seed: int = 0,
    per_guard: int = REJECTIONS_PER_GUARD,
    total: int = REJECTIONS_TOTAL,
    files: Sequence[str] = (),
) -> list[RejectionSpan]:
    """Draw a seeded sample of rejected proposals, as disputes between the text before and the proposed text.

    Proposals with no audible difference, and bad-index replies (nothing was proposed against any text), are not
    candidates. At most ``per_guard`` proposals are drawn per guard, pooled over candidates, and the per-guard
    limit shrinks until the draw fits in ``total``.
    """
    proposals: dict[str, list[list[RejectionSpan]]] = {}
    for video in videos(manifest, files):
        runs = [(c, _require_run(store, c, video.id)) for c in candidates]
        if not any(run.rejections for _, run in runs):
            continue
        timing = _timing(store, video, load_base(store, video.id))
        for candidate, run in runs:
            for ordinal, rejection in enumerate(run.rejections):
                if rejection.reason == "index":
                    continue
                found = _rejection_disputes(timing, candidate, ordinal, rejection)
                if found:
                    proposals.setdefault(rejection.reason, []).append(found)
    limit = per_guard
    while sum(min(len(p), limit) for p in proposals.values()) > total:
        limit -= 1
    rng = random.Random(seed)
    drawn = [
        group
        for guard in sorted(proposals)
        for group in rng.sample(proposals[guard], min(len(proposals[guard]), limit))
    ]
    return sorted((s for group in drawn for s in group), key=lambda r: (r.candidate, r.span.id))


def _rejection_disputes(
    timing: _Timing, candidate: Candidate, ordinal: int, rejection: Rejection
) -> list[RejectionSpan]:
    units = {
        BEFORE: timing.units(BEFORE, rejection.segment, rejection.before),
        PROPOSED: timing.units(PROPOSED, rejection.segment, rejection.proposed),
    }
    spans = window_spans(timing.video, timing.language, ordinal, units, pivot=BEFORE)
    # The prefix keeps these ids apart from the edit spans', which are numbered by stretch.
    return [
        RejectionSpan(
            candidate=candidate.id,
            guard=rejection.reason,
            rejection=ordinal,
            span=s.model_copy(update={"id": f"{candidate.id}:{s.id}"}),
        )
        for s in spans
        if not s.format_only
    ]


def session_for(
    store: Layout,
    manifest: Manifest,
    candidates: Sequence[Candidate],
    target: Candidate,
    *,
    n: int = SESSION_N,
    seed: int = 0,
) -> tuple[list[ShownSpan], SessionKey]:
    """Build the A/B session of ``target``'s audible edits against the base, from the disputes among ``candidates``.

    ``candidates`` must be the set the judge screened, so each session span is a judged dispute under the same id.
    """
    if target.id not in {c.id for c in candidates}:
        raise ConfigError(f"{target.id} is not among the candidates whose edits were compared")
    spans = audible_edits(edit_spans(store, manifest, candidates), target.id)
    return session_from_spans(spans, manifest, BASE, target.id, n=n, seed=seed)
