import json
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner
from evals_fixture import CHUNKS, EN, SECONDS, ZH, chunk_rows, make_eval

from evals import disk, run_asr
from evals.store import CAPTION_HYPOTHESIS, REFERENCE, Layout, read_history, rows_by_file, write_rows
from ytverbatim.asr import registry
from ytverbatim.asr.chunks import Region
from ytverbatim.asr.models import RunMetrics, WorkerRequest
from ytverbatim.asr.protocol import EngineDef, EngineSpec
from ytverbatim.asr.runner import WorkerResult
from ytverbatim.errors import ConfigError

HERE = Path(__file__).parent
GB = 1_000_000_000


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("YTVERBATIM_HOME", str(home / "share"))
    monkeypatch.setenv("PYTHONPATH", str(HERE / "asr"))


def fake_definition(hypothesis: str = "q-base") -> EngineDef:
    real = registry.get(hypothesis).spec
    spec = EngineSpec(
        hypothesis, "fake", real.repo, real.revision, real.max_chunk_s, False, False, real.supports_context, 16_000
    )
    return EngineDef(spec, "fake_engine.create", {"language": registry.get(hypothesis).variant["language"]})


def plenty(_: Path) -> int:
    return 100 * GB


def test_asr_runs_the_real_worker_with_the_shared_plan_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    cache = tmp_path / "cache"
    first = run_asr.run_hypothesis(layout, manifest, "q-base", [ZH], cache, definition=fake_definition(), free=plenty)
    assert first.decoded_files == (ZH,)
    rows = rows_by_file(layout, "q-base")[ZH]
    assert [(r.start, r.end) for r in rows] == [(c.start, c.end) for c in CHUNKS]
    # The fake engine echoes language and context: the job carries the metadata language, the registry's policy
    # makes Qwen3 auto-detect, and a non-context engine gets no context.
    assert rows[0].text.startswith("auto::")
    request_job = WorkerRequest.model_validate_json(
        (layout.file_run_dir("q-base", ZH) / "request.json").read_text()
    ).files[0]
    assert request_job.language == "zh"
    request = WorkerRequest.model_validate_json((layout.file_run_dir("q-base", ZH) / "request.json").read_text())
    assert request.hf_hub_cache == str(cache)
    (history,) = read_history(layout, "q-base")
    assert history.files == (ZH,)
    assert history.file_seconds == SECONDS
    assert history.metrics.peak_footprint_bytes is not None
    assert history.metrics.peak_footprint_bytes > 0

    again = run_asr.run_hypothesis(layout, manifest, "q-base", [ZH], cache, definition=fake_definition(), free=plenty)
    assert again.results == ()
    assert again.skipped_files == (ZH,)


def test_context_goes_only_to_context_engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    job = run_asr.file_job(layout, manifest.videos[0], registry.get("q-ctx"), run_asr.require_plan(layout, ZH))
    assert job.context is not None
    assert "NVDA" in job.context.hotwords
    assert job.language == "zh"
    whole = run_asr.file_job(layout, manifest.videos[0], registry.get("w-prod"), run_asr.require_plan(layout, ZH))
    assert whole.context is None
    assert whole.chunks is None


def test_asr_cli_with_a_fake_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, _ = make_eval(tmp_path / "eval", monkeypatch)
    seen: list[WorkerRequest] = []

    def worker(request: WorkerRequest) -> WorkerResult:
        seen.append(request)
        hypothesis = request.engine.spec.id
        write_rows(layout, hypothesis, [r for job in request.files for r in chunk_rows(hypothesis, job.id, "abcd")])
        metrics = RunMetrics(
            hypothesis=hypothesis, files=len(request.files), chunks_decoded=8, chunks_skipped=0, load_s=1.0,
            decode_s=4.0, audio_s=30.0, peak_memory_bytes=GB, max_rss_bytes=GB,
        )  # fmt: skip
        return WorkerResult(layout.rows_path(hypothesis, request.files[0].id), metrics, GB, 6.0)

    def run_hypothesis(*args: object, **kwargs: object) -> run_asr.AsrOutcome:
        return original(*args, worker=worker, free=plenty, **kwargs)  # type: ignore[arg-type]

    original = run_asr.run_hypothesis
    monkeypatch.setattr(run_asr, "run_hypothesis", run_hypothesis)
    result = CliRunner().invoke(run_asr.main, ["asr", "--hyp", "q-base"])
    assert result.exit_code == 0, result.output
    assert "2 files, 16 chunks decoded" in result.output
    assert [[job.id for job in r.files] for r in seen] == [[ZH], [EN]]
    assert {Path(r.out_dir).name for r in seen} == {ZH, EN}
    again = CliRunner().invoke(run_asr.main, ["asr", "--hyp", "q-base"])
    assert "complete for all 2 selected files" in again.output
    assert len(seen) == 2


def test_asr_needs_a_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    layout.plan_path(EN).unlink()
    with pytest.raises(ConfigError, match="run `vad` first"):
        run_asr.run_hypothesis(layout, manifest, "q-base", [EN], tmp_path, definition=fake_definition(), free=plenty)


def test_vad_writes_a_plan_from_the_speech_regions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)

    def regions(_: Path) -> tuple[list[Region], float, list[float]]:
        return [Region(1.0, 12.0), Region(13.0, 50.0)], 60.0, [0.1] * 600

    plan = run_asr.plan_file(layout, manifest.videos[0], regions)
    assert plan.speech == ((1.0, 12.0), (13.0, 50.0))
    assert all(c.duration <= 30.0 for c in plan.chunks)
    assert plan.windows[0][0] == 0.0
    assert plan.windows[-1][1] == 60.0
    assert all(a[1] == b[0] for a, b in zip(plan.windows, plan.windows[1:], strict=False))


# --- disk gate ---


def snapshot(cache: Path, repo: str, revision: str) -> None:
    path = disk.repo_dir(cache, repo) / "snapshots" / revision
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")


def test_disk_gate_refuses_when_the_weights_would_not_fit(tmp_path: Path) -> None:
    definition = registry.get("f-aed")
    with pytest.raises(ConfigError, match=r"needs 4\.57 GB of weights plus 1\.50 GB headroom, but only 5\.00 GB"):
        disk.check_disk(definition, tmp_path, free=lambda _: 5 * GB)
    assert disk.check_disk(definition, tmp_path, free=lambda _: 7 * GB) == 4_570_000_000


def test_disk_gate_counts_every_missing_repo_and_skips_present_ones(tmp_path: Path) -> None:
    definition = registry.get("m-4bit")
    assert disk.check_disk(definition, tmp_path, free=lambda _: 100 * GB) == 4_530_000_000 + 2_580_000_000
    snapshot(tmp_path, definition.spec.repo, definition.spec.revision)
    assert disk.check_disk(definition, tmp_path, free=lambda _: 100 * GB) == 2_580_000_000


def test_disk_gate_passes_when_the_weights_are_present_even_with_no_space(tmp_path: Path) -> None:
    definition = registry.get("q-base")
    snapshot(tmp_path, definition.spec.repo, definition.spec.revision)
    assert disk.check_disk(definition, tmp_path, free=lambda _: 0) == 0
    assert disk.check_disk(registry.get("w-prod"), tmp_path, free=lambda _: 0) == 0


def test_asr_stops_at_the_disk_gate_before_starting_a_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)

    def worker(_: WorkerRequest) -> WorkerResult:
        raise AssertionError("the worker must not start")

    with pytest.raises(ConfigError, match="headroom"):
        run_asr.run_hypothesis(layout, manifest, "f-aed", [], tmp_path / "cache", worker=worker, free=lambda _: GB)


# --- cleanup ---


def complete(layout: Layout, hypothesis: str, files: tuple[str, ...] = (ZH, EN)) -> None:
    rows = [r for f in files for r in chunk_rows(hypothesis, f, "abcd")]
    if registry.get(hypothesis).spec.max_chunk_s is None:
        rows = [r for r in rows if r.chunk == 0]
    write_rows(layout, hypothesis, rows)


def recorder() -> tuple[list[str], Callable[[Path, str], int]]:
    deleted: list[str] = []

    def delete(_: Path, repo: str) -> int:
        deleted.append(repo)
        return GB

    return deleted, delete


def test_cleanup_refuses_incomplete_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    complete(layout, "f-aed", (ZH,))
    deleted, delete = recorder()
    with pytest.raises(ConfigError, match="not complete for 1 file"):
        run_asr.cleanup_hypothesis(layout, manifest, "f-aed", tmp_path / "cache", delete=delete)
    assert deleted == []
    complete(layout, "f-aed")
    run_asr.cleanup_hypothesis(layout, manifest, "f-aed", tmp_path / "cache", delete=delete)
    assert deleted == ["mlx-community/FireRedASR2-AED-mlx"]


def test_cleanup_refuses_while_a_hypothesis_sharing_the_repo_still_has_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    complete(layout, "q-base")
    deleted, delete = recorder()
    with pytest.raises(ConfigError, match="q-ctx shares weights with q-base"):
        run_asr.cleanup_hypothesis(layout, manifest, "q-base", tmp_path / "cache", delete=delete)
    complete(layout, "q-ctx")
    run_asr.cleanup_hypothesis(layout, manifest, "q-base", tmp_path / "cache", delete=delete)
    assert deleted == ["mlx-community/Qwen3-ASR-1.7B-8bit"]


def test_cleanup_deletes_the_tokenizer_repo_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    complete(layout, "m-4bit")
    deleted, delete = recorder()
    run_asr.cleanup_hypothesis(layout, manifest, "m-4bit", tmp_path / "cache", delete=delete)
    assert deleted == ["mlx-community/MiMo-Audio-Tokenizer", "mlx-community/MiMo-V2.5-ASR-MLX"]


def test_cleanup_never_touches_whisper_or_the_default_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    complete(layout, "w-prod")
    complete(layout, "f-aed")
    deleted, delete = recorder()
    with pytest.raises(ConfigError, match="default Hugging Face cache"):
        run_asr.cleanup_hypothesis(layout, manifest, "w-prod", tmp_path / "cache", delete=delete)
    default = Path.home() / ".cache" / "huggingface" / "hub"
    with pytest.raises(ConfigError, match="default Hugging Face cache"):
        run_asr.cleanup_hypothesis(layout, manifest, "f-aed", default, delete=delete)
    assert deleted == []


def test_delete_repo_also_frees_shared_blobs(tmp_path: Path) -> None:
    pytest.importorskip("huggingface_hub")
    cache = tmp_path / "hub"
    repo = disk.repo_dir(cache, "example/model")
    blob = repo / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * 100)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("1" * 40)
    snap = repo / "snapshots" / ("1" * 40)
    snap.mkdir(parents=True)
    (snap / "weights.bin").symlink_to("../../blobs/abc")
    keep = disk.repo_dir(cache, "example/other")
    (keep / "snapshots" / ("2" * 40)).mkdir(parents=True)
    freed = disk.delete_repo(cache, "example/model")
    assert freed == 100
    assert not repo.exists()
    assert keep.exists()
    assert disk.delete_repo(cache, "example/missing") == 0


# --- captions ---


def test_captions_build_the_pseudo_hypothesis_and_the_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    assert run_asr.build_captions(layout, manifest) == {CAPTION_HYPOTHESIS: 1, REFERENCE: 1}
    auto = rows_by_file(layout, CAPTION_HYPOTHESIS)
    assert list(auto) == [EN]
    assert [w.text for w in auto[EN][0].words] == ["we", "bought", "stock"]
    reference = rows_by_file(layout, REFERENCE)
    assert [r.text for r in reference[ZH]] == ["我们今天买了股票", "然后卖出了一百股"]
    assert REFERENCE not in run_asr.hypotheses_on_disk(layout)
    first_line = layout.rows_path(CAPTION_HYPOTHESIS, EN).read_text().splitlines()[0]
    assert json.loads(first_line)["hypothesis"] == "yt-auto"
