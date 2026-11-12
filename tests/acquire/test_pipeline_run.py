import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from ytverbatim import config, pipeline
from ytverbatim.acquire import session
from ytverbatim.asr import runner as asr_runner
from ytverbatim.asr.models import HYPOTHESES_FILE, ChunkHypothesis, RunMetrics, WorkerRequest
from ytverbatim.asr.runner import WorkerResult
from ytverbatim.cli import main
from ytverbatim.contract import ContextDocument, Word, load_document
from ytverbatim.errors import BusyError, ConfigError, EngineError
from ytverbatim.pipeline import ProgressEvent, VideoResult, process, process_many, video_lock
from ytverbatim.ytdlp.types import Signal

PUBLIC = "pubVideo001"
NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)

# One sentence per YouTube chapter of the fake extraction (0-50 s and 50-100 s).
SENTENCES = (
    (" Markets", " rose", " today.", 5.0),
    (" Then", " they", " fell", " sharply.", 60.0),
)


def whole_file_rows(request: WorkerRequest) -> list[ChunkHypothesis]:
    words: list[Word] = []
    for sentence in SENTENCES:
        clock = sentence[-1]
        assert isinstance(clock, float)
        for piece in sentence[:-1]:
            assert isinstance(piece, str)
            words.append(Word(text=piece, start=clock, end=clock + 0.4))
            clock += 0.5
    return [
        ChunkHypothesis(
            hypothesis=request.engine.spec.id,
            file=request.files[0].id,
            chunk=0,
            start=0.0,
            end=100.0,
            text="".join(w.text for w in words),
            language="en",
            words=tuple(words),
        )
    ]


def chunked_rows(request: WorkerRequest) -> list[ChunkHypothesis]:
    texts = ("Markets rose today.", "Then they fell sharply.")
    return [
        ChunkHypothesis(
            hypothesis=request.engine.spec.id,
            file=request.files[0].id,
            chunk=index,
            start=5.0 + 55.0 * index,
            end=9.0 + 55.0 * index,
            text=text,
            language="en",
        )
        for index, text in enumerate(texts)
    ]


@dataclass
class FakeWorker:
    """Stands in for `asr.runner.run_worker`: writes the hypotheses file a real worker would leave."""

    calls: list[WorkerRequest] = field(default_factory=list[WorkerRequest])
    seen_wav: list[bool] = field(default_factory=list[bool])
    stale_output: list[bool] = field(default_factory=list[bool])
    error: EngineError | None = None

    def __call__(self, request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
        self.calls.append(request)
        out = Path(request.out_dir)
        self.seen_wav.append(Path(request.files[0].wav).is_file())
        self.stale_output.append((out / HYPOTHESES_FILE).exists())
        if self.error is not None:
            raise self.error
        out.mkdir(parents=True, exist_ok=True)
        rows = whole_file_rows(request) if request.engine.spec.max_chunk_s is None else chunked_rows(request)
        (out / HYPOTHESES_FILE).write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
        metrics = RunMetrics(
            hypothesis=request.engine.spec.id,
            files=1,
            chunks_decoded=len(rows),
            chunks_skipped=0,
            load_s=0.0,
            decode_s=0.0,
            audio_s=100.0,
            peak_memory_bytes=None,
            max_rss_bytes=0,
        )
        return WorkerResult(out / HYPOTHESES_FILE, metrics, 0, 0.0)


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> FakeWorker:
    fake = FakeWorker()
    monkeypatch.setattr(asr_runner, "run_worker", fake)
    return fake


def run(kit, yt, targets: Sequence[str] = (PUBLIC,), *, options: config.Options | None = None, **kwargs: object):
    results: list[VideoResult] = []
    events: list[ProgressEvent] = []
    report = process_many(
        targets,
        options,
        out=kit.out,
        runner_factory=lambda _sleep: yt,
        sleep=kit.slept.append,
        clock=lambda: NOW,
        progress=events.append,
        on_video=results.append,
        **kwargs,  # type: ignore[arg-type]
    )
    return report, results, events


def engine_options(engine: str) -> config.Options:
    return config.load_options(overrides={"asr": {"engine": engine}})


def test_end_to_end_writes_a_valid_document_and_markdown(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    report, results, events = run(session_kit, yt)

    assert report.exit_code == 0
    vdir = session_kit.out / PUBLIC
    doc = load_document(vdir / "context.json")
    assert [s.text for s in doc.segments] == ["Markets rose today.", "Then they fell sharply."]
    assert doc.video.id == PUBLIC and doc.access.cookies_used is False
    assert len(doc.chapters) == 2 and [s.chapter for s in doc.segments] == [0, 1]
    assert doc.transcript.runs[0].engine == "w-prod" and doc.transcript.punctuation == "engine"
    markdown = (vdir / "context.md").read_text(encoding="utf-8")
    assert markdown.startswith("# Synthetic title\n") and "Markets rose today." in markdown
    assert "Transcript: w-prod" in markdown

    assert not (vdir / "audio.wav").exists(), "the 16 kHz WAV is deleted after a successful run"
    assert (vdir / "audio.webm").is_file(), "the compressed audio is kept"
    assert (vdir / "metadata.json").is_file() and (vdir / "access.json").is_file()
    batch = json.loads((session_kit.out / "batch_report.json").read_text(encoding="utf-8"))
    assert batch["videos"][0]["status"] == "ok"

    (request,) = worker.calls
    assert request.engine.spec.id == "w-prod"
    assert request.files[0].language == "en" and request.files[0].id == PUBLIC
    assert request.files[0].wav == str(vdir / "audio.wav") and worker.seen_wav == [True]
    assert Path(request.out_dir) == vdir / ".stages" / "asr" / "w-prod"
    assert [e.stage for e in events] == ["fetch", "asr", "write"]
    assert results[0].document == doc and results[0].markdown == vdir / "context.md"
    assert len(yt.labels("metadata")) == 1


def test_process_returns_the_document(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    doc = process(PUBLIC, out=session_kit.out, runner_factory=lambda _sleep: yt, progress=lambda _event: None)
    assert isinstance(doc, ContextDocument) and len(doc.segments) == 2
    assert load_document(session_kit.out / PUBLIC / "context.json") == doc


def test_process_takes_a_single_video(session_kit, worker: FakeWorker) -> None:
    with pytest.raises(ValueError, match="one YouTube video"):
        process("https://www.youtube.com/playlist?list=PLsynthetic", out=session_kit.out)
    with pytest.raises(ValueError, match="one YouTube video"):
        process("not a url", out=session_kit.out)


def test_a_finished_video_is_skipped_without_fetching_or_decoding(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    _, first, _ = run(session_kit, yt)
    calls = len(yt.calls)

    report, second, events = run(session_kit, yt)
    assert report.exit_code == 0
    assert len(yt.calls) == calls and len(worker.calls) == 1
    assert second[0].outcome.reused == ("context",)
    assert second[0].document == first[0].document
    assert [e.stage for e in events] == ["skip"]


def test_force_redoes_the_video_from_a_clean_asr_directory(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    run(session_kit, yt)
    run(session_kit, yt, force=True)
    assert len(worker.calls) == 2
    assert worker.stale_output == [False, False], "--force must not resume from the earlier hypotheses"
    assert worker.seen_wav == [True, True], "the deleted WAV is rebuilt from the kept audio"
    assert not (session_kit.out / PUBLIC / "audio.wav").exists()


def test_an_interrupted_asr_run_resumes_in_the_same_directory(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    worker.error = EngineError("killed mid-decode")
    report, _, _ = run(session_kit, yt)
    assert report.exit_code == 1
    assert not (session_kit.out / PUBLIC / "context.json").exists()
    assert (session_kit.out / PUBLIC / "audio.wav").is_file(), "a failed run keeps the WAV for the retry"

    worker.error = None
    report, _, _ = run(session_kit, yt)
    assert report.exit_code == 0
    assert Path(worker.calls[0].out_dir) == Path(worker.calls[1].out_dir)


def test_a_worker_failure_is_a_per_video_error_and_the_batch_continues(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    worker.error = EngineError("w-prod worker failed (RuntimeError): out of memory")
    report, results, _ = run(session_kit, yt, ["pubVideo001", "pubVideo002"])
    first, second = (r.outcome for r in results)
    assert (first.error, second.error) == ("engine", "engine") and "out of memory" in first.detail
    assert report.exit_code == 1 and len(worker.calls) == 2
    assert all(r.document is None and r.markdown is None for r in results)


def test_process_raises_the_engine_error(session_kit, worker: FakeWorker) -> None:
    worker.error = EngineError("boom")
    yt = session_kit.yt()
    with pytest.raises(EngineError, match="boom"):
        process(PUBLIC, out=session_kit.out, runner_factory=lambda _sleep: yt)


def test_a_fetch_failure_skips_asr(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.GEO)]
    report, results, _ = run(session_kit, yt)
    assert report.exit_code == 1 and results[0].outcome.error == "geo"
    assert worker.calls == []


def test_a_bot_check_stops_the_batch(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.BOT_CHECK)]
    report, results, _ = run(session_kit, yt, [PUBLIC, "pubVideo002"])
    assert report.exit_code == 3 and report.stopped is not None
    assert [v.status for v in report.videos] == [session.VideoStatus.ERROR, session.VideoStatus.NOT_ATTEMPTED]
    assert [r.outcome.error for r in results] == ["bot_check"]
    assert worker.calls == []


def test_a_stop_seen_on_a_successful_call_finishes_that_video_and_stops_the_rest(
    session_kit, worker: FakeWorker
) -> None:
    yt = session_kit.yt()
    yt.secondary[("metadata", False)] = (Signal.BOT_CHECK,)
    report, _, _ = run(session_kit, yt, [PUBLIC, "pubVideo002"])
    assert report.exit_code == 3 and report.stopped is not None and report.stopped.signal is Signal.BOT_CHECK
    assert [v.status for v in report.videos] == [session.VideoStatus.OK, session.VideoStatus.NOT_ATTEMPTED]
    assert (session_kit.out / PUBLIC / "context.md").is_file()


def test_process_reraises_a_batch_stop(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.BOT_CHECK)]
    with pytest.raises(session.BatchStopError) as stopped:
        process(PUBLIC, out=session_kit.out, runner_factory=lambda _sleep: yt)
    assert stopped.value.signal is Signal.BOT_CHECK


def test_a_held_lock_raises_busy_without_touching_the_network(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    vdir = session_kit.out / PUBLIC
    with video_lock(vdir):
        report, results, _ = run(session_kit, yt)
        assert report.exit_code == 1 and results[0].outcome.error == "busy"
        with pytest.raises(BusyError, match=PUBLIC):
            process(PUBLIC, out=session_kit.out, runner_factory=lambda _sleep: yt)
    assert yt.calls == [] and worker.calls == []
    report, _, _ = run(session_kit, yt)
    assert report.exit_code == 0, "the lock is released when the holder is done"


def test_the_lock_is_released_after_a_failure(session_kit, worker: FakeWorker) -> None:
    worker.error = EngineError("boom")
    run(session_kit, session_kit.yt())
    with video_lock(session_kit.out / PUBLIC):
        pass


def test_the_engine_option_selects_a_chunked_hypothesis(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    _, results, _ = run(session_kit, yt, options=engine_options("q-base"))
    (request,) = worker.calls
    assert request.engine.spec.id == "q-base" and request.files[0].chunks is None, "the worker plans the chunks"
    doc = results[0].document
    assert doc is not None
    assert doc.transcript.runs[0].engine == "q-base" and doc.transcript.runs[0].vad is not None
    assert all(s.words is None for s in doc.segments)
    assert any("chunk-level" in w for w in doc.quality.warnings)
    assert Path(request.out_dir).name == "q-base"


def test_an_unknown_engine_fails_before_any_request(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    with pytest.raises(ConfigError, match="unknown ASR hypothesis"):
        run(session_kit, yt, options=engine_options("nope"))
    assert yt.calls == []


def test_captions_as_the_source_are_refused_until_they_exist(session_kit, worker: FakeWorker) -> None:
    options = config.load_options(overrides={"asr": {"source": "captions"}})
    with pytest.raises(ConfigError, match="captions"):
        run(session_kit, session_kit.yt(), options=options)


def test_no_captions_flag_reaches_the_fetch(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    run(session_kit, yt, captions=False)
    assert yt.labels("captions") == []


def test_a_cooldown_refuses_the_run(session_kit, worker: FakeWorker) -> None:
    session.start_cooldown(Signal.RATE_LIMITED, NOW)
    yt = session_kit.yt()
    with pytest.raises(session.BatchStopError):
        run(session_kit, yt)
    assert yt.calls == []
    report, _, _ = run(session_kit, yt, ignore_cooldown=True)
    assert report.exit_code == 0


def test_a_playlist_runs_every_video(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    yt.playlist = ["plVideo0001", "plVideo0002"]
    report, results, _ = run(session_kit, yt, ["https://www.youtube.com/playlist?list=PLsynthetic"])
    assert report.exit_code == 0 and [r.outcome.video_id for r in results] == yt.playlist
    assert all((session_kit.out / v / "context.md").is_file() for v in yt.playlist)


def invoke(kit, yt, monkeypatch: pytest.MonkeyPatch, *args: str):
    monkeypatch.setattr(pipeline, "SubprocessRunner", lambda *, sleep_args=(): yt)
    return CliRunner().invoke(main, ["run", "--out", str(kit.out), *args])


def test_run_is_registered() -> None:
    result = CliRunner().invoke(main, ["--help"])
    assert "run" in result.output
    help_text = CliRunner().invoke(main, ["run", "--help"]).output
    assert all(flag in help_text for flag in ("--out", "--auth", "--engine", "--captions", "--force"))


def test_cli_prints_the_output_path_per_video_and_exits_zero(
    session_kit, worker: FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = invoke(session_kit, session_kit.yt(), monkeypatch, PUBLIC)
    assert result.exit_code == 0, result.output
    line = next(line for line in result.output.splitlines() if line.startswith("✓"))
    assert PUBLIC in line and "context.md" in line and "2 segments" in line and "2 chapters" in line
    assert "1 ok, 0 failed" in result.output
    assert str(session_kit.home) not in result.output


def test_cli_second_run_is_skipped_and_force_redoes_it(
    session_kit, worker: FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    yt = session_kit.yt()
    invoke(session_kit, yt, monkeypatch, PUBLIC)
    again = invoke(session_kit, yt, monkeypatch, PUBLIC)
    assert again.exit_code == 0 and "already done" in again.output and len(worker.calls) == 1
    forced = invoke(session_kit, yt, monkeypatch, PUBLIC, "--force")
    assert forced.exit_code == 0 and "already done" not in forced.output and len(worker.calls) == 2


def test_cli_engine_flag_selects_the_hypothesis(
    session_kit, worker: FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = invoke(session_kit, session_kit.yt(), monkeypatch, PUBLIC, "--engine", "q-base")
    assert result.exit_code == 0, result.output
    assert worker.calls[0].engine.spec.id == "q-base"
    bad = invoke(session_kit, session_kit.yt(), monkeypatch, PUBLIC, "--engine", "nope")
    assert bad.exit_code == 1 and "unknown ASR hypothesis" in bad.output


def test_cli_exit_codes_follow_fetch(session_kit, worker: FakeWorker, monkeypatch: pytest.MonkeyPatch) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    auth = invoke(session_kit, yt, monkeypatch, "memVideo001")
    assert auth.exit_code == 2 and "auth_required" in auth.output and "Fix:" in auth.output

    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.GEO)]
    failed = invoke(session_kit, yt, monkeypatch, PUBLIC, "pubVideo002")
    assert failed.exit_code == 1 and "1 ok, 1 failed" in failed.output

    worker.error = EngineError("boom")
    broken = invoke(session_kit, session_kit.yt(), monkeypatch, "pubVideo003")
    assert broken.exit_code == 1 and "boom" in broken.output
    worker.error = None

    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.BOT_CHECK)]
    stopped = invoke(session_kit, yt, monkeypatch, "pubVideo004")
    assert stopped.exit_code == 3
    refused = invoke(session_kit, session_kit.yt(), monkeypatch, "pubVideo005")
    assert refused.exit_code == 3 and "--ignore-cooldown" in refused.output
    allowed = invoke(session_kit, session_kit.yt(), monkeypatch, "pubVideo005", "--ignore-cooldown")
    assert allowed.exit_code == 0, allowed.output


def test_cli_rejects_non_youtube_targets(session_kit, worker: FakeWorker, monkeypatch: pytest.MonkeyPatch) -> None:
    result = invoke(session_kit, session_kit.yt(), monkeypatch, "https://example.com/video")
    assert result.exit_code == 1 and "not a YouTube" in result.output
