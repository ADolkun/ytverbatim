import fcntl
import logging
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fake_engine import definition, write_silence

from ytverbatim import paths, proc
from ytverbatim.asr import runner
from ytverbatim.asr.chunks import Chunk
from ytverbatim.asr.models import ERROR_FILE, HYPOTHESES_FILE, REQUEST_FILE, ChunkHypothesis, FileJob, WorkerRequest
from ytverbatim.errors import EngineError, MediaValidationError

HERE = Path(__file__).parent


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("PYTHONPATH", str(HERE))
    return home


def make_request(tmp_path: Path, *, seconds: float = 6.0, **variant: object) -> WorkerRequest:
    wav = write_silence(tmp_path / "-dash.wav", seconds)
    job = FileJob(id="f1", wav=str(wav), language="en", chunks=(Chunk(0.5, 2.5), Chunk(3.0, 5.0)))
    return WorkerRequest(engine=definition(**variant), out_dir=str(tmp_path / "out"), files=(job,))


def test_worker_argv_isolates_the_working_directory() -> None:
    argv = runner.worker_argv(Path("/tmp/req.json"))
    assert argv == [sys.executable, "-P", "-m", "ytverbatim.asr.worker", "/tmp/req.json"]


def test_a_real_worker_subprocess_produces_hypotheses_and_resumes(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    first = runner.run_worker(request)
    lines = first.hypotheses.read_text(encoding="utf-8").splitlines()
    assert [ChunkHypothesis.model_validate_json(line).chunk for line in lines] == [0, 1]
    assert first.hypotheses == tmp_path / "out" / HYPOTHESES_FILE
    assert (tmp_path / "out" / REQUEST_FILE).is_file()
    assert (first.metrics.chunks_decoded, first.metrics.chunks_skipped) == (2, 0)
    assert first.metrics.audio_s == pytest.approx(4.0)
    assert first.metrics.max_rss_bytes > 0
    assert first.children_max_rss_bytes > 0

    again = runner.run_worker(request)
    assert (again.metrics.chunks_decoded, again.metrics.chunks_skipped) == (0, 2)
    assert first.hypotheses.read_text(encoding="utf-8").splitlines() == lines


def test_the_error_report_travels_to_the_parent(tmp_path: Path) -> None:
    request = make_request(tmp_path, fail_on_call=2)
    with pytest.raises(EngineError, match=r"fake worker failed \(EngineError\):.*f1 chunk 1.*fake decoder exploded"):
        runner.run_worker(request)
    assert (tmp_path / "out" / ERROR_FILE).is_file()


def test_a_worker_that_dies_without_a_report_still_raises(tmp_path: Path) -> None:
    stub = tmp_path / "stub.sh"
    stub.write_text("#!/bin/sh\necho oom killed >&2\nexit 3\n", encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(EngineError, match=r"exited with 3 and left no error report\noom killed"):
        runner.run_worker(make_request(tmp_path), python=str(stub))


def test_a_clean_exit_without_metrics_raises(tmp_path: Path) -> None:
    stub = tmp_path / "stub.sh"
    stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(EngineError, match=r"no readable metrics\.json"):
        runner.run_worker(make_request(tmp_path), python=str(stub))


def test_timeout_is_three_times_the_audio_plus_half_an_hour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert runner.timeout_s(100.0) == 2100.0
    seen: list[float] = []

    def fake_run(argv: list[str], *, timeout_s: float) -> proc.ProcResult:
        seen.append(timeout_s)
        return proc.ProcResult(1, "", "", 0.0)

    monkeypatch.setattr(runner.proc, "run", fake_run)
    with pytest.raises(EngineError):
        runner.run_worker(make_request(tmp_path, seconds=10.0))
    assert seen == [pytest.approx(3 * 10.0 + 1800.0)]


def test_an_unreadable_wav_is_refused_before_spawning(tmp_path: Path) -> None:
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not audio")
    request = WorkerRequest(
        engine=definition(), out_dir=str(tmp_path / "out"), files=(FileJob(id="f1", wav=str(junk)),)
    )
    with pytest.raises(MediaValidationError):
        runner.run_worker(request)


def test_the_lock_is_exclusive_while_held() -> None:
    with runner.worker_lock():
        other = os.open(paths.locks_dir() / runner.LOCK_NAME, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(other)
    with runner.worker_lock():
        pass


def test_a_waiting_worker_says_so_and_proceeds_when_the_lock_frees(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=runner.log.name)
    entered = threading.Event()

    def contend() -> None:
        with runner.worker_lock():
            entered.set()

    with runner.worker_lock():
        thread = threading.Thread(target=contend)
        thread.start()
        assert not entered.wait(0.3)
    thread.join(timeout=10)
    assert entered.is_set()
    assert "waiting for another ASR worker" in caplog.text


def test_the_parent_imports_neither_mlx_nor_numpy_nor_an_engine_adapter() -> None:
    code = (
        "import sys, ytverbatim.asr.runner, ytverbatim.asr.registry\n"
        "heavy = {'mlx', 'mlx_audio', 'mlx_whisper', 'numpy'}\n"
        "loaded = {name.split('.')[0] for name in sys.modules} & heavy\n"
        "loaded |= {name for name in sys.modules if name.startswith('ytverbatim.asr.engines')}\n"
        "print(sorted(loaded))\n"
        "sys.exit(bool(loaded))"
    )
    result = subprocess.run([sys.executable, "-P", "-c", code], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
