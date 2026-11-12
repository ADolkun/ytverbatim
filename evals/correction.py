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
import socket
import sys
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, ValidationError

from evals import disk
from evals.manifest import Manifest, Video
from evals.store import Layout, file_complete, now, require_plan, rows_by_file, videos, write_json
from ytverbatim import config, pipeline, proc, stages
from ytverbatim.acquire.metadata import VideoMetadata, trim_info
from ytverbatim.asr import registry
from ytverbatim.asr.runner import default_memory_cap_bytes, worker_lock
from ytverbatim.contract.models import Edit, Flag, LLMUsage
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
