"""Child process that loads one engine and decodes a request's files; run as `python -P -m ytverbatim.asr.worker`.

Everything that needs MLX happens here, never in the parent. `-P` keeps the working directory off `sys.path`, so a
checkout in the current directory cannot shadow the installed package.
"""

import json
import math
import os
import resource
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from ytverbatim import atomic
from ytverbatim.asr.audio import Audio, frame_energy, load_wav
from ytverbatim.asr.chunks import Chunk, plan_chunks
from ytverbatim.asr.guard import QualityGuard
from ytverbatim.asr.models import (
    ERROR_FILE,
    FINGERPRINT_FILE,
    HYPOTHESES_FILE,
    METRICS_FILE,
    ChunkHypothesis,
    FileJob,
    RunMetrics,
    WorkerError,
    WorkerRequest,
)
from ytverbatim.asr.protocol import ChunkFlag, ChunkInput, Engine, EngineDef, Guard, GuardedChunk, RawChunk
from ytverbatim.asr.registry import build_engine
from ytverbatim.asr.vad import speech_regions
from ytverbatim.contract.models import Word
from ytverbatim.errors import EngineError

EXIT_FAILED = 1
EXIT_BAD_REQUEST = 2
_PLAN_TOLERANCE_S = 1e-3


class _MlxCore(Protocol):
    def get_peak_memory(self) -> int: ...


def run_request(request: WorkerRequest, *, engine: Engine | None = None, guard: Guard | None = None) -> RunMetrics:
    """Decode every file, appending one flushed JSONL line per chunk and skipping chunks already on disk.

    `engine` replaces the one built from the request, for in-process tests. `guard` is the hook for the quality
    checks: it sees every decoded chunk and may decode again before the line is written.
    """
    spec = request.engine.spec
    out = Path(request.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / ERROR_FILE).unlink(missing_ok=True)
    sink_path = out / HYPOTHESES_FILE
    _claim_output(out, request)
    done = _read_done(sink_path, spec.id)
    session = _Session(request.engine, engine, guard)
    decoded = skipped = 0
    audio_s = 0.0
    with sink_path.open("a", encoding="utf-8") as sink:
        for job in request.files:
            audio = load_wav(Path(job.wav))
            for index, chunk in enumerate(_plan(job, request.engine, audio)):
                existing = done.get((job.id, index))
                if existing is not None:
                    _require_same_chunk(existing, chunk)
                    skipped += 1
                    continue
                row = session.decode(spec.id, job, audio, index, chunk)
                sink.write(row.model_dump_json() + "\n")
                sink.flush()
                decoded += 1
                audio_s += chunk.duration
    metrics = RunMetrics(
        hypothesis=spec.id,
        files=len(request.files),
        chunks_decoded=decoded,
        chunks_skipped=skipped,
        load_s=session.load_s,
        decode_s=session.decode_s,
        audio_s=audio_s,
        peak_memory_bytes=_peak_memory_bytes(),
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    atomic.write_json(out / METRICS_FILE, metrics.model_dump(mode="json"))
    return metrics


class _Session:
    """Builds and loads the engine on the first chunk that needs decoding, so a fully resumed run loads nothing."""

    def __init__(self, definition: EngineDef, engine: Engine | None, guard: Guard | None) -> None:
        self._definition = definition
        self._engine = engine
        self._guard = guard
        self._loaded = False
        self.load_s = 0.0
        self.decode_s = 0.0

    def decode(self, hypothesis: str, job: FileJob, audio: Audio, index: int, chunk: Chunk) -> ChunkHypothesis:
        engine = self._ready()
        samples = audio.slice(chunk.start, chunk.end)
        language = self._definition.engine_language(job.language)
        started = time.perf_counter()
        try:
            raw = engine.transcribe(samples, language=language, context=job.context)
            guarded = (
                GuardedChunk(raw)
                if self._guard is None
                else self._guard(engine, ChunkInput(samples, chunk.start, chunk.end, language, job.context), raw)
            )
        except Exception as exc:
            where = f"{job.id} chunk {index} ({chunk.start:.1f}-{chunk.end:.1f}s)"
            raise EngineError(f"{hypothesis} failed on {where}: {exc}") from exc
        finally:
            self.decode_s += time.perf_counter() - started
        return _row(hypothesis, job.id, index, chunk, guarded)

    def _ready(self) -> Engine:
        if self._engine is None:
            self._engine = build_engine(self._definition)
        if not self._loaded:
            started = time.perf_counter()
            self._engine.load()
            self.load_s = time.perf_counter() - started
            self._loaded = True
        return self._engine


def _row(hypothesis: str, file: str, index: int, chunk: Chunk, guarded: GuardedChunk) -> ChunkHypothesis:
    raw = guarded.raw
    flags: list[ChunkFlag] = ["forced_cut"] if chunk.forced_cut else []
    if raw.generated_tokens is not None and raw.max_tokens is not None and raw.generated_tokens >= raw.max_tokens:
        flags.append("token_cap_hit")
    flags.extend(guarded.flags)
    return ChunkHypothesis(
        hypothesis=hypothesis,
        file=file,
        chunk=index,
        start=chunk.start,
        end=chunk.end,
        text=raw.text,
        language=raw.language,
        words=_absolute(raw, chunk.start),
        generated_tokens=raw.generated_tokens,
        max_tokens=raw.max_tokens,
        confidence=raw.confidence,
        flags=tuple(dict.fromkeys(flags)),
    )


def _absolute(raw: RawChunk, offset: float) -> tuple[Word, ...]:
    return tuple(Word(text=w.text, start=w.start + offset, end=w.end + offset, conf=w.conf) for w in raw.words)


def _plan(job: FileJob, definition: EngineDef, audio: Audio) -> list[Chunk]:
    if audio.duration == 0:
        return []
    max_chunk_s = definition.spec.max_chunk_s
    if max_chunk_s is None:
        return [Chunk(0.0, audio.duration)]
    if job.chunks is not None:
        return list(job.chunks)
    return plan_chunks(speech_regions(audio), audio.duration, max_chunk_s=max_chunk_s, frame_energy=frame_energy(audio))


def default_guard(request: WorkerRequest) -> Guard | None:
    """Return the guard `main` runs: none when the request opts out, flag-only for a whole-file hypothesis.

    A whole-file hypothesis is the production control, so re-decoding part of it would stop it being one.
    """
    if not request.quality_guard:
        return None
    return QualityGuard(redecode=request.engine.spec.max_chunk_s is not None)


def _claim_output(out: Path, request: WorkerRequest) -> None:
    """Record which request this output directory belongs to, or refuse to resume one that belongs to another."""
    path = out / FINGERPRINT_FILE
    current = request.fingerprint()
    if not path.exists():
        hypotheses = out / HYPOTHESES_FILE
        if hypotheses.exists() and hypotheses.stat().st_size > 0:
            raise EngineError(
                f"{out} holds {HYPOTHESES_FILE} but no {FINGERPRINT_FILE}, so what produced it is unknown; use a fresh "
                "output directory"
            )
        atomic.write_json(path, current)
        return
    try:
        stored: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EngineError(f"{path} is unreadable: {exc}") from exc
    if stored != current:
        known = stored if isinstance(stored, dict) else {}
        changed = sorted(key for key in current.keys() | known.keys() if current.get(key) != known.get(key))
        raise EngineError(
            f"{out} holds output of a different request ({', '.join(changed) or 'format'} changed); resuming would "
            "mix the two, so use a fresh output directory"
        )


def _read_done(path: Path, hypothesis: str) -> dict[tuple[str, int], ChunkHypothesis]:
    if not path.exists():
        return {}
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    if len(complete) != len(data):
        # A kill between write and flush leaves a partial last line; that chunk is simply decoded again.
        with path.open("r+b") as handle:
            handle.truncate(len(complete))
    done: dict[tuple[str, int], ChunkHypothesis] = {}
    for number, line in enumerate(complete.decode("utf-8").splitlines(), start=1):
        try:
            row = ChunkHypothesis.model_validate_json(line)
        except ValueError as exc:
            raise EngineError(f"{path.name} line {number} is corrupt: {exc}") from exc
        if row.hypothesis != hypothesis:
            raise EngineError(f"{path.name} holds hypothesis {row.hypothesis!r}, not {hypothesis!r}")
        done[(row.file, row.chunk)] = row
    return done


def _require_same_chunk(existing: ChunkHypothesis, chunk: Chunk) -> None:
    if not (
        math.isclose(existing.start, chunk.start, abs_tol=_PLAN_TOLERANCE_S)
        and math.isclose(existing.end, chunk.end, abs_tol=_PLAN_TOLERANCE_S)
    ):
        raise EngineError(
            f"{existing.file} chunk {existing.chunk} was decoded for {existing.start:.3f}-{existing.end:.3f}s but the "
            f"plan now says {chunk.start:.3f}-{chunk.end:.3f}s; use a fresh output directory for a different plan"
        )


def _peak_memory_bytes() -> int | None:
    # Only a process that already ran an MLX engine or the VAD has the module; importing it here would defeat a
    # fake-engine run, and there is no peak to report without it.
    core = sys.modules.get("mlx.core")
    return None if core is None else cast(_MlxCore, core).get_peak_memory()


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -P -m ytverbatim.asr.worker REQUEST.json", file=sys.stderr)
        return EXIT_BAD_REQUEST
    try:
        request = WorkerRequest.model_validate_json(Path(args[0]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"invalid worker request: {exc}", file=sys.stderr)
        return EXIT_BAD_REQUEST
    if request.hf_hub_cache is not None:
        # huggingface_hub reads this when it is first imported, which only happens inside an adapter.
        os.environ["HF_HUB_CACHE"] = request.hf_hub_cache
    try:
        run_request(request, guard=default_guard(request))
    except Exception as exc:
        error = WorkerError(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        out = Path(request.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        atomic.write_json(out / ERROR_FILE, error.model_dump(mode="json"))
        print(f"{error.type}: {error.message}", file=sys.stderr)
        return EXIT_FAILED
    return 0


if __name__ == "__main__":
    sys.exit(main())
