import json
import os
import sys
from pathlib import Path

import pytest
from fake_engine import RATE, definition, write_silence
from pydantic import JsonValue, ValidationError

from ytverbatim.asr import worker
from ytverbatim.asr.chunks import Chunk, Region
from ytverbatim.asr.models import (
    ERROR_FILE,
    FINGERPRINT_FILE,
    HYPOTHESES_FILE,
    METRICS_FILE,
    ChunkHypothesis,
    FileJob,
    RunMetrics,
    WorkerRequest,
)
from ytverbatim.asr.protocol import (
    AsrContext,
    ChunkInput,
    Engine,
    EngineDef,
    EngineSpec,
    GuardedChunk,
    RawChunk,
)
from ytverbatim.contract.models import Word
from ytverbatim.errors import EngineError


def request(tmp_path: Path, *jobs: FileJob, engine: EngineDef | None = None) -> WorkerRequest:
    return WorkerRequest(engine=engine or definition(), out_dir=str(tmp_path / "out"), files=jobs)


def rows(tmp_path: Path) -> list[ChunkHypothesis]:
    lines = (tmp_path / "out" / HYPOTHESES_FILE).read_text(encoding="utf-8").splitlines()
    return [ChunkHypothesis.model_validate_json(line) for line in lines]


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    return write_silence(tmp_path / "-dash.wav", 10.0)


class FakeMlx:
    """Records allocator calls into the fake engine's call log, so their order against load and decode shows."""

    def __init__(self, log: Path | None = None) -> None:
        self.log = log
        self.calls: list[str] = []

    def _record(self, call: str) -> None:
        self.calls.append(call)
        if self.log is not None:
            with self.log.open("a", encoding="utf-8") as handle:
                handle.write(call + "\n")

    def set_cache_limit(self, limit: int) -> int:
        self._record(f"cache_limit {limit}")
        return 0

    def set_memory_limit(self, limit: int) -> int:
        self._record(f"memory_limit {limit}")
        return 0

    def clear_cache(self) -> None:
        self._record("clear_cache")

    def get_peak_memory(self) -> int:
        return 7


@pytest.fixture(autouse=True)
def fake_mlx(monkeypatch: pytest.MonkeyPatch) -> FakeMlx:
    """Stands in for `mlx.core`, so `main` never changes the real allocator of the test process."""
    fake = FakeMlx()
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    return fake


def test_the_allocator_is_bounded_before_load_and_emptied_after_every_chunk(tmp_path: Path, wav: Path) -> None:
    log = tmp_path / "calls.log"
    job = FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0), Chunk(3.0, 5.0)))
    req = request(tmp_path, job, engine=definition(calls_log=str(log))).model_copy(update={"memory_cap_bytes": 8 << 30})
    worker.run_request(req, mlx=FakeMlx(log))
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"cache_limit {worker.CACHE_LIMIT_BYTES}",
        f"memory_limit {int((8 << 30) * worker.MLX_LIMIT_FRACTION)}",
        "load",
        f"transcribe {2 * RATE}",
        "clear_cache",
        f"transcribe {2 * RATE}",
        "clear_cache",
    ]


def test_without_a_cap_only_the_cache_is_bounded(tmp_path: Path, wav: Path) -> None:
    mlx = FakeMlx()
    worker.run_request(request(tmp_path, FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0),))), mlx=mlx)
    assert mlx.calls == [f"cache_limit {worker.CACHE_LIMIT_BYTES}", "clear_cache"]


def test_main_bounds_the_mlx_allocator(tmp_path: Path, wav: Path, fake_mlx: FakeMlx) -> None:
    req = request(tmp_path, FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0),))).model_copy(
        update={"memory_cap_bytes": 1 << 30}
    )
    req_path = tmp_path / "req.json"
    req_path.write_text(req.model_dump_json(), encoding="utf-8")
    assert worker.main([str(req_path)]) == 0
    assert fake_mlx.calls == [
        f"cache_limit {worker.CACHE_LIMIT_BYTES}",
        f"memory_limit {int((1 << 30) * worker.MLX_LIMIT_FRACTION)}",
        "clear_cache",
    ]
    metrics = RunMetrics.model_validate_json((tmp_path / "out" / METRICS_FILE).read_text(encoding="utf-8"))
    assert metrics.peak_memory_bytes == 7
    assert metrics.peak_footprint_bytes is not None
    assert metrics.peak_footprint_bytes > 0


def test_the_memory_cap_does_not_change_the_fingerprint(tmp_path: Path, wav: Path) -> None:
    req = request(tmp_path, FileJob(id="f1", wav=str(wav)))
    assert req.model_copy(update={"memory_cap_bytes": 1 << 30}).fingerprint() == req.fingerprint()


def test_chunks_become_flushed_lines_with_absolute_times(tmp_path: Path, wav: Path) -> None:
    job = FileJob(
        id="f1",
        wav=str(wav),
        language="en",
        context=AsrContext(system_prompt="ctx"),
        chunks=(Chunk(1.0, 3.0), Chunk(4.0, 9.0, forced_cut=True)),
    )
    metrics = worker.run_request(request(tmp_path, job))
    first, second = rows(tmp_path)
    assert (first.hypothesis, first.file, first.chunk, first.start, first.end) == ("fake", "f1", 0, 1.0, 3.0)
    assert first.text == f"en:ctx:{2 * RATE}"
    assert [(w.start, w.end) for w in first.words] == [(1.0, 1.5), (1.5, 2.0)]
    assert second.words[1].conf == 0.8
    assert [(w.start, w.end) for w in second.words] == [(4.0, 4.5), (4.5, 5.0)]
    assert first.flags == ()
    assert second.flags == ("forced_cut",)
    assert metrics.chunks_decoded == 2
    assert metrics.audio_s == pytest.approx(7.0)
    saved = RunMetrics.model_validate_json((tmp_path / "out" / METRICS_FILE).read_text(encoding="utf-8"))
    assert saved == metrics
    assert saved.max_rss_bytes > 0


def test_whole_file_hypothesis_is_one_chunk(tmp_path: Path, wav: Path) -> None:
    worker.run_request(request(tmp_path, FileJob(id="f1", wav=str(wav)), engine=definition(max_chunk_s=None)))
    (only,) = rows(tmp_path)
    assert (only.chunk, only.start, only.end) == (0, 0.0, pytest.approx(10.0))


def test_whole_file_hypothesis_refuses_a_chunk_plan(tmp_path: Path, wav: Path) -> None:
    job = FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 1.0),))
    with pytest.raises(ValidationError, match="whole files"):
        request(tmp_path, job, engine=definition(max_chunk_s=None))


def test_file_ids_must_be_unique(tmp_path: Path, wav: Path) -> None:
    with pytest.raises(ValidationError, match="unique"):
        request(tmp_path, FileJob(id="f", wav=str(wav)), FileJob(id="f", wav=str(wav)))


def test_missing_plan_comes_from_the_vad(tmp_path: Path, wav: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []

    def fake_regions(audio: object) -> list[Region]:
        seen.append(audio.duration)  # type: ignore[attr-defined]
        return [Region(0.5, 2.0), Region(5.0, 7.0)]

    monkeypatch.setattr(worker, "speech_regions", fake_regions)
    worker.run_request(request(tmp_path, FileJob(id="f1", wav=str(wav))))
    assert seen == [pytest.approx(10.0)]
    assert [(r.start, r.end) for r in rows(tmp_path)] == [(0.5, 7.0)]


def test_resume_skips_finished_chunks_and_loads_nothing(tmp_path: Path, wav: Path) -> None:
    log = tmp_path / "calls.log"
    engine = definition(calls_log=str(log))
    job = FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0), Chunk(3.0, 5.0)))
    worker.run_request(request(tmp_path, job, engine=engine))
    before = (tmp_path / "out" / HYPOTHESES_FILE).read_bytes()
    log.unlink()
    metrics = worker.run_request(request(tmp_path, job, engine=engine))
    assert (metrics.chunks_decoded, metrics.chunks_skipped, metrics.load_s) == (0, 2, 0.0)
    assert not log.exists()
    assert (tmp_path / "out" / HYPOTHESES_FILE).read_bytes() == before


def test_resume_decodes_only_what_is_missing_after_a_torn_write(tmp_path: Path, wav: Path) -> None:
    job = FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0), Chunk(3.0, 5.0), Chunk(6.0, 8.0)))
    worker.run_request(request(tmp_path, job))
    path = tmp_path / "out" / HYPOTHESES_FILE
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 40])
    metrics = worker.run_request(request(tmp_path, job))
    assert (metrics.chunks_decoded, metrics.chunks_skipped) == (1, 2)
    assert [r.chunk for r in rows(tmp_path)] == [0, 1, 2]


def test_resume_refuses_a_different_chunk_plan(tmp_path: Path, wav: Path) -> None:
    worker.run_request(request(tmp_path, FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0),))))
    with pytest.raises(EngineError, match="fresh output directory"):
        worker.run_request(request(tmp_path, FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 3.0),))))


def test_resume_refuses_another_hypothesis_and_corrupt_lines(tmp_path: Path, wav: Path) -> None:
    job = FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0),))
    worker.run_request(request(tmp_path, job))
    path = tmp_path / "out" / HYPOTHESES_FILE
    other = EngineDef(_renamed(definition().spec), "fake_engine.create", {"language": "pass"})
    with pytest.raises(EngineError, match=r"different request \(engine changed\)"):
        worker.run_request(request(tmp_path, job, engine=other))
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(EngineError, match="line 1 is corrupt"):
        worker.run_request(request(tmp_path, job))


@pytest.mark.parametrize(
    ("update", "job_update", "field"),
    [
        ({"engine": definition(calls_log="elsewhere")}, {}, "engine"),
        ({"quality_guard": False}, {}, "quality_guard"),
        ({}, {"language": "zh"}, "files"),
        ({}, {"context": AsrContext(system_prompt="other")}, "files"),
    ],
)
def test_resume_refuses_a_request_that_differs_beyond_the_output_location(
    tmp_path: Path, wav: Path, update: dict[str, object], job_update: dict[str, object], field: str
) -> None:
    job = FileJob(id="f1", wav=str(wav), language="en", chunks=(Chunk(0.0, 2.0),))
    worker.run_request(request(tmp_path, job))
    before = (tmp_path / "out" / HYPOTHESES_FILE).read_bytes()
    changed = request(tmp_path, job.model_copy(update=job_update)).model_copy(update=update)
    with pytest.raises(EngineError, match=rf"\({field} changed\).*fresh output directory"):
        worker.run_request(changed)
    assert (tmp_path / "out" / HYPOTHESES_FILE).read_bytes() == before


def test_resume_ignores_where_the_weights_are_cached(tmp_path: Path, wav: Path) -> None:
    job = FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0),))
    worker.run_request(request(tmp_path, job))
    moved = request(tmp_path, job).model_copy(update={"hf_hub_cache": "/example/hub"})
    assert worker.run_request(moved).chunks_skipped == 1
    stored = json.loads((tmp_path / "out" / FINGERPRINT_FILE).read_text(encoding="utf-8"))
    assert "out_dir" not in stored
    assert "hf_hub_cache" not in stored


def test_hypotheses_without_a_fingerprint_are_not_resumed(tmp_path: Path, wav: Path) -> None:
    job = FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0),))
    worker.run_request(request(tmp_path, job))
    (tmp_path / "out" / FINGERPRINT_FILE).unlink()
    with pytest.raises(EngineError, match=r"no fingerprint\.json"):
        worker.run_request(request(tmp_path, job))


@pytest.mark.parametrize(("policy", "heard"), [("pass", "zh+en"), ("auto", "auto")])
def test_the_language_policy_decides_what_the_engine_hears(tmp_path: Path, wav: Path, policy: str, heard: str) -> None:
    job = FileJob(id="f1", wav=str(wav), language="zh+en", chunks=(Chunk(0.0, 1.0),))
    seen: list[str | None] = []

    def guard(engine: Engine, chunk: ChunkInput, raw: RawChunk) -> GuardedChunk:
        seen.append(chunk.language)
        return GuardedChunk(raw)

    worker.run_request(request(tmp_path, job, engine=definition(language=policy)), guard=guard)
    assert rows(tmp_path)[0].text.startswith(f"{heard}:")
    assert seen == [None if policy == "auto" else "zh+en"]


@pytest.mark.parametrize("policy", [None, "detect", True])
def test_a_request_without_a_valid_language_policy_is_refused(tmp_path: Path, wav: Path, policy: JsonValue) -> None:
    variant: dict[str, JsonValue] = {} if policy is None else {"language": policy}
    engine = EngineDef(definition().spec, "fake_engine.create", variant)
    with pytest.raises(ValidationError, match="'auto' or 'pass'"):
        request(tmp_path, FileJob(id="f1", wav=str(wav)), engine=engine)


def _renamed(spec: EngineSpec) -> EngineSpec:
    return EngineSpec(
        "other",
        spec.family,
        spec.repo,
        spec.revision,
        spec.max_chunk_s,
        spec.native_timestamps,
        spec.emits_punctuation,
        spec.supports_context,
        spec.sample_rate,
    )


def test_token_cap_is_flagged_by_the_worker(tmp_path: Path, wav: Path) -> None:
    job = FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0, forced_cut=True),))
    worker.run_request(request(tmp_path, job, engine=definition(hit_cap=True)))
    assert rows(tmp_path)[0].flags == ("forced_cut", "token_cap_hit")


def test_guard_sees_every_chunk_and_can_replace_it(tmp_path: Path, wav: Path) -> None:
    seen: list[tuple[float, float, int]] = []

    def guard(engine: Engine, chunk: ChunkInput, raw: RawChunk) -> GuardedChunk:
        seen.append((chunk.start, chunk.end, len(chunk.audio)))
        again = engine.transcribe(chunk.audio[:RATE], language=chunk.language, context=chunk.context)
        return GuardedChunk(
            RawChunk(text=f"fixed {again.text}", words=(Word(text="x", start=0.0, end=0.1),)), ("decoder_anomaly",)
        )

    job = FileJob(id="f1", wav=str(wav), chunks=(Chunk(1.0, 4.0, forced_cut=True),))
    worker.run_request(request(tmp_path, job), guard=guard)
    assert seen == [(1.0, 4.0, 3 * RATE)]
    (row,) = rows(tmp_path)
    assert row.text == f"fixed auto::{RATE}"
    assert row.flags == ("forced_cut", "decoder_anomaly")
    assert row.words == (Word(text="x", start=1.0, end=1.1),)
    assert row.generated_tokens is None


def test_engine_failure_names_the_chunk_and_keeps_earlier_lines(tmp_path: Path, wav: Path) -> None:
    job = FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0), Chunk(3.0, 5.0)))
    with pytest.raises(EngineError, match=r"f1 chunk 1 \(3\.0-5\.0s\): fake decoder exploded"):
        worker.run_request(request(tmp_path, job, engine=definition(fail_on_call=2)))
    assert [r.chunk for r in rows(tmp_path)] == [0]


def test_a_run_that_does_not_finish_leaves_no_metrics_to_mark_it_complete(tmp_path: Path, wav: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / METRICS_FILE).write_text("{}", encoding="utf-8")
    job = FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0),))
    with pytest.raises(EngineError, match="fake decoder exploded"):
        worker.run_request(request(tmp_path, job, engine=definition(fail_on_call=1)))
    assert not (out / METRICS_FILE).exists()


def test_main_writes_a_structured_error_and_exits_nonzero(
    tmp_path: Path, wav: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job = FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0),))
    req = request(tmp_path, job, engine=definition(fail_on_call=1))
    req_path = tmp_path / "req.json"
    req_path.write_text(req.model_dump_json(), encoding="utf-8")
    assert worker.main([str(req_path)]) == worker.EXIT_FAILED
    error = json.loads((tmp_path / "out" / ERROR_FILE).read_text(encoding="utf-8"))
    assert error["type"] == "EngineError"
    assert "fake decoder exploded" in error["message"]
    assert "Traceback" in error["traceback"]
    assert "EngineError" in capsys.readouterr().err


def test_main_clears_a_stale_error_on_success(tmp_path: Path, wav: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / ERROR_FILE).write_text("{}", encoding="utf-8")
    req = request(tmp_path, FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 2.0),)))
    req_path = tmp_path / "req.json"
    req_path.write_text(req.model_dump_json(), encoding="utf-8")
    assert worker.main([str(req_path)]) == 0
    assert not (out / ERROR_FILE).exists()


def test_main_rejects_bad_invocations(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert worker.main([]) == worker.EXIT_BAD_REQUEST
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert worker.main([str(bad)]) == worker.EXIT_BAD_REQUEST
    assert "invalid worker request" in capsys.readouterr().err


def test_main_applies_the_hub_cache_from_the_request(
    tmp_path: Path, wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", "/example/before")
    req = WorkerRequest(
        engine=definition(),
        out_dir=str(tmp_path / "out"),
        files=(FileJob(id="f1", wav=str(wav), chunks=(Chunk(0.0, 1.0),)),),
        hf_hub_cache="/example/hub",
    )
    req_path = tmp_path / "req.json"
    req_path.write_text(req.model_dump_json(), encoding="utf-8")
    assert worker.main([str(req_path)]) == 0
    assert os.environ["HF_HUB_CACHE"] == "/example/hub"


def test_empty_audio_has_nothing_to_decode(tmp_path: Path) -> None:
    empty = write_silence(tmp_path / "empty.wav", 0.0)
    metrics = worker.run_request(
        request(tmp_path, FileJob(id="f1", wav=str(empty)), engine=definition(max_chunk_s=None))
    )
    assert metrics.chunks_decoded == 0
    assert rows(tmp_path) == []


def test_chunk_hypothesis_rejects_a_backwards_span() -> None:
    with pytest.raises(ValidationError, match="before it starts"):
        ChunkHypothesis(hypothesis="h", file="f", chunk=0, start=2.0, end=1.0, text="")
