import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import JsonValue

from ytverbatim import config, pipeline, proc
from ytverbatim.acquire import audio, session
from ytverbatim.acquire.metadata import Chapter, VideoMetadata
from ytverbatim.asr import registry
from ytverbatim.asr import runner as asr_runner
from ytverbatim.asr.models import HYPOTHESES_FILE, ChunkHypothesis, RunMetrics, WorkerRequest
from ytverbatim.asr.runner import WorkerResult
from ytverbatim.auth.fixes import fix_for
from ytverbatim.auth.source import CookieFileSource
from ytverbatim.auth.state import AuthState
from ytverbatim.cli import main
from ytverbatim.contract import ContextDocument, Word, load_document
from ytverbatim.errors import (
    AuthError,
    BusyError,
    ConfigError,
    ContractError,
    DependencyError,
    EngineError,
    EngineMemoryError,
    MediaValidationError,
    ProcMemoryExceeded,
    ProcTimeout,
    UnavailableError,
    YtDlpError,
)
from ytverbatim.pipeline import ProgressEvent, VideoResult, process, process_many, video_lock
from ytverbatim.store import input_hash
from ytverbatim.ytdlp.types import Signal

PUBLIC = "pubVideo001"
MEMBERS = "memVideo001"
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
    # Raise `error` only from this call on (1-based), so earlier videos finish.
    fail_from_call: int = 1
    # A line appended to the hypotheses of the named file, to stand in for a corrupt row.
    garbage: dict[str, str] = field(default_factory=dict[str, str])

    def __call__(self, request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
        self.calls.append(request)
        out = Path(request.out_dir)
        self.seen_wav.append(Path(request.files[0].wav).is_file())
        self.stale_output.append((out / HYPOTHESES_FILE).exists())
        if self.error is not None and len(self.calls) >= self.fail_from_call:
            raise self.error
        out.mkdir(parents=True, exist_ok=True)
        rows = whole_file_rows(request) if request.engine.spec.max_chunk_s is None else chunked_rows(request)
        lines = [row.model_dump_json() for row in rows]
        if request.files[0].id in self.garbage:
            lines.append(self.garbage[request.files[0].id])
        (out / HYPOTHESES_FILE).write_text("".join(line + "\n" for line in lines), encoding="utf-8")
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


@pytest.fixture(autouse=True)
def engine_installed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Report every engine library as installed; the list collects the modules that were asked about."""
    asked: list[str] = []

    def find_spec(name: str) -> object:
        asked.append(name)
        return object()

    monkeypatch.setattr(registry, "find_spec", find_spec)
    return asked


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
    assert Path(request.out_dir).parent == vdir / ".stages" / "asr" / "w-prod", "one folder per request, by engine"
    assert [e.stage for e in events] == ["fetch", "asr", "normalize", "punctuate", "numeric", "write"]
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


def memory_kill(trigger: str = "available") -> EngineMemoryError:
    cause = ProcMemoryExceeded(
        "python3",
        trigger,  # type: ignore[arg-type]
        footprint_bytes=5 << 30,
        cap_bytes=12 << 30,
        available_percent=6,
        floor_percent=10,
    )
    return EngineMemoryError(f"w-prod worker stopped to protect the machine: {cause}", "close memory-heavy apps", cause)


@pytest.mark.parametrize("trigger", ["available", "footprint"])
def test_a_memory_kill_stops_the_batch_and_keeps_the_report(session_kit, worker: FakeWorker, trigger: str) -> None:
    yt = session_kit.yt()
    worker.error = memory_kill(trigger)
    worker.fail_from_call = 2
    with pytest.raises(EngineMemoryError):
        run(session_kit, yt, ["pubVideo001", "pubVideo002", "pubVideo003"])
    assert len(worker.calls) == 2
    assert batch_ids(session_kit) == [("pubVideo001", "ok")]


def test_cli_prints_the_memory_fix_and_exits_nonzero(
    session_kit, worker: FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker.error = memory_kill()
    result = invoke(session_kit, session_kit.yt(), monkeypatch, PUBLIC, "pubVideo002")
    assert result.exit_code == 1
    assert "■ w-prod worker stopped to protect the machine" in result.output
    assert "  Fix: close memory-heavy apps" in result.output
    assert len(worker.calls) == 1


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


def test_process_raises_a_login_failure_as_an_auth_error_with_its_fix(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    with pytest.raises(AuthError) as raised:
        process(PUBLIC, out=session_kit.out, runner_factory=lambda _sleep: yt)
    assert raised.value.state is AuthState.AUTH_REQUIRED and "no auth source is set" in raised.value.detail
    assert raised.value.fix == fix_for(AuthState.AUTH_REQUIRED)
    assert worker.calls == []


def test_process_raises_a_yt_dlp_failure_as_it_was_with_its_fix(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.GEO)]
    with pytest.raises(YtDlpError) as raised:
        process(PUBLIC, out=session_kit.out, runner_factory=lambda _sleep: yt)
    assert raised.value.primary is Signal.GEO and "blocks this video in your country" in raised.value.fix


def test_process_raises_an_unavailable_video_as_it_was(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    yt.live_status[PUBLIC] = "is_live"
    with pytest.raises(UnavailableError) as raised:
        process(PUBLIC, out=session_kit.out, runner_factory=lambda _sleep: yt)
    assert raised.value.kind == "live" and raised.value.fix == ""


def test_process_raises_a_media_failure_as_it_was(session_kit, worker: FakeWorker) -> None:
    session_kit.proc.probe_returns("3.0\n")
    with pytest.raises(MediaValidationError):
        process(PUBLIC, out=session_kit.out, runner_factory=lambda _sleep: session_kit.yt())


def test_process_raises_a_timeout_as_it_was(session_kit, worker: FakeWorker) -> None:
    def hang(_argv: list[str]) -> proc.ProcResult:
        raise ProcTimeout("ffmpeg", 600.0)

    session_kit.proc.responses["ffmpeg"] = hang
    with pytest.raises(ProcTimeout) as raised:
        process(PUBLIC, out=session_kit.out, runner_factory=lambda _sleep: session_kit.yt())
    assert raised.value.timeout_s == 600.0


def test_a_recorded_failure_keeps_its_cause_out_of_the_batch_report(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    raw = "ERROR: [youtube] synthetic cause line\nTraceback-like detail that must stay out of the report"
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.GEO, preview=raw)]
    _, results, _ = run(session_kit, yt)
    cause = results[0].outcome.cause
    assert isinstance(cause, YtDlpError) and "Traceback-like detail" in cause.preview
    text = (session_kit.out / "batch_report.json").read_text(encoding="utf-8")
    assert "Traceback-like detail" not in text
    (video,) = json.loads(text)["videos"]
    assert video["detail"] == "ERROR: [youtube] synthetic cause line"
    assert "cause" not in video


def test_cookies_first_sends_the_first_extraction_with_cookies(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    process(
        MEMBERS,
        out=session_kit.out,
        auth="cookie-file:/nonexistent/cookies.txt",
        cookies_first=True,
        runner_factory=lambda _sleep: yt,
    )
    assert yt.labels("metadata", cookied=False) == []
    assert len(yt.labels("metadata", cookied=True)) == 1


def test_by_default_the_first_extraction_is_anonymous(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    process(
        MEMBERS,
        out=session_kit.out,
        auth="cookie-file:/nonexistent/cookies.txt",
        runner_factory=lambda _sleep: yt,
    )
    assert len(yt.labels("metadata", cookied=False)) == 1 and yt.labels("metadata", cookied=True) == []


def test_cookies_first_without_a_source_is_an_auth_error_with_a_fix(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    with pytest.raises(AuthError) as raised:
        process(MEMBERS, out=session_kit.out, cookies_first=True, runner_factory=lambda _sleep: yt)
    assert raised.value.state is AuthState.AUTH_REQUIRED and raised.value.fix
    assert yt.calls == []


NOTE = {
    "is_pinned": True,
    "author_is_uploader": True,
    "text": "Correction: the figure is 12%.",
    "author": "@synthetic",
    "like_count": 7,
    "_time_text": "1 day ago",
}


def note_options(*, on: bool) -> config.Options:
    return config.load_options(overrides={"ytdlp": {"creator_note": on}})


def test_the_document_carries_the_creator_note_only_when_the_option_is_on(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    yt.comments[PUBLIC] = [NOTE]
    _, results, _ = run(session_kit, yt)
    assert results[0].document is not None and results[0].document.video.creator_note is None
    assert results[0].document.schema_version == "1.1"
    assert "--get-comments" not in yt.labels("metadata")[0].args

    _, results, _ = run(session_kit, yt, options=note_options(on=True))
    document = results[0].document
    assert document is not None and document.video.creator_note is not None
    assert document.video.creator_note.model_dump() == {
        "text": "Correction: the figure is 12%.",
        "author": "@synthetic",
        "like_count": 7,
        "published": "1 day ago",
    }
    assert len(yt.labels("metadata")) == 2, "turning it on re-extracts once"
    assert len(worker.calls) == 1, "and reuses every stored stage"
    assert load_document(session_kit.out / PUBLIC / "context.json") == document

    _, results, _ = run(session_kit, yt, options=note_options(on=True))
    assert results[0].outcome.reused == ("context",) and len(yt.labels("metadata")) == 2


def test_a_video_without_a_pinned_creator_comment_has_no_note(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    yt.comments[PUBLIC] = [{**NOTE, "author_is_uploader": False}]
    _, results, _ = run(session_kit, yt, options=note_options(on=True))
    assert results[0].document is not None and results[0].document.video.creator_note is None
    assert "--get-comments" in yt.labels("metadata")[0].args


def test_turning_the_option_off_drops_a_saved_note_from_the_document(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    yt.comments[PUBLIC] = [NOTE]
    run(session_kit, yt, options=note_options(on=True))
    _, results, _ = run(session_kit, yt, options=note_options(on=False))
    assert results[0].outcome.reused != ("context",), "the document is made again"
    assert results[0].document is not None and results[0].document.video.creator_note is None
    assert len(yt.labels("metadata")) == 1


def settings_for(kit, options: config.Options) -> pipeline._Settings:
    return pipeline._settings(
        options, None, kit.out, force=False, captions=True, emit=lambda _e: None, clock=lambda: NOW
    )


def test_the_default_run_key_is_the_one_made_before_the_creator_note_existed(session_kit, worker: FakeWorker) -> None:
    default = settings_for(session_kit, config.load_options())
    versions = pipeline.stage_versions()
    before = input_hash(
        "context",
        versions["context"],
        {
            "versions": dict[str, JsonValue](versions),
            "source": default.source,
            "captions": True,
            "engine": pipeline._engine_key(default.definition),
            "second_opinion": None,
            "text": pipeline._plain(asdict(default.text)),
            "llm": None,
        },
    )
    assert default.run_key() == before
    assert settings_for(session_kit, note_options(on=False)).run_key() == before
    assert settings_for(session_kit, note_options(on=True)).run_key() != before


def video_options(*, on: bool) -> config.Options:
    return config.load_options(overrides={"ytdlp": {"video": on}})


def test_a_default_run_requests_no_video(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    run(session_kit, yt)
    assert [c.label for c in yt.calls] == ["metadata", "captions", "audio"]
    assert not (session_kit.out / PUBLIC / "video.mp4").exists()


def test_the_option_downloads_the_video_during_a_run(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    _, results, _ = run(session_kit, yt, options=video_options(on=True))
    assert [c.label for c in yt.calls] == ["metadata", "captions", "audio", "video"]
    assert (session_kit.out / PUBLIC / "video.mp4").is_file() and results[0].document is not None

    _, again, _ = run(session_kit, yt, options=video_options(on=True))
    assert len(yt.labels("video")) == 1 and again[0].outcome.reused == ("context",)


def test_a_current_document_with_a_wanted_video_makes_exactly_one_video_request(
    session_kit, worker: FakeWorker
) -> None:
    yt = session_kit.yt()
    _, first, _ = run(session_kit, yt)
    before = len(yt.calls)

    _, second, events = run(session_kit, yt, options=video_options(on=True))

    assert [c.label for c in yt.calls[before:]] == ["video"]
    assert len(worker.calls) == 1
    assert second[0].document == first[0].document
    assert second[0].outcome.reused == ("context", "info")
    assert [e.stage for e in events] == ["skip", "fetch"]
    assert [t.stage for t in second[0].timings] == ["fetch"]
    assert (session_kit.out / PUBLIC / "video.mp4").is_file()

    run(session_kit, yt, options=video_options(on=True))
    assert len(yt.labels("video")) == 1


def age_info(vdir: Path) -> None:
    old = (vdir / "info.json").stat().st_mtime - 2 * 3600
    os.utime(vdir / "info.json", (old, old))


def test_an_old_extraction_is_refreshed_once_before_the_video(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    run(session_kit, yt)
    age_info(session_kit.out / PUBLIC)
    before = len(yt.calls)

    run(session_kit, yt, options=video_options(on=True))

    assert [c.label for c in yt.calls[before:]] == ["metadata", "video"]
    assert all(c.cookies is None for c in yt.calls[before:])


def test_the_reuse_path_keeps_the_cookies_the_document_was_made_with(session_kit, worker: FakeWorker) -> None:
    source = CookieFileSource(Path("/nonexistent/cookies.txt"))
    yt = session_kit.yt()
    yt.availability[MEMBERS] = "subscriber_only"
    yt.failures[("metadata", False)] = [session_kit.fail(Signal.MEMBERS_ONLY)]
    yt.secondary[("metadata", True)] = (Signal.AUTH_COOKIES_FOUND,)
    _, first, _ = run(session_kit, yt, [MEMBERS], source=source)
    assert first[0].document is not None and first[0].document.access.cookies_used
    age_info(session_kit.out / MEMBERS)
    before = len(yt.calls)

    run(session_kit, yt, [MEMBERS], options=video_options(on=True), source=source)

    after = yt.calls[before:]
    assert [c.label for c in after] == ["metadata", "video"]
    assert all(c.cookies is not None for c in after), "no anonymous attempt that YouTube would refuse"


def test_the_reuse_path_stays_anonymous_under_policy_never(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    run(session_kit, yt)
    age_info(session_kit.out / PUBLIC)
    before = len(yt.calls)
    options = config.load_options(overrides={"auth": {"policy": "never"}, "ytdlp": {"video": True}})
    run(session_kit, yt, options=options, source=CookieFileSource(Path("/nonexistent/cookies.txt")))
    assert [c.label for c in yt.calls[before:]] == ["metadata", "video"]
    assert all(c.cookies is None for c in yt.calls[before:])


def test_a_stale_info_403_keeps_the_creator_note(
    session_kit, worker: FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audio, "INFO_MAX_AGE_S", -1.0)
    yt = session_kit.yt()
    yt.comments[PUBLIC] = [NOTE]
    yt.failures[("video", False)] = [session_kit.fail(Signal.HTTP_403)]
    options = config.load_options(overrides={"ytdlp": {"video": True, "creator_note": True}})

    _, results, _ = run(session_kit, yt, options=options)

    assert [c.label for c in yt.calls] == ["metadata", "captions", "audio", "video", "metadata", "video"]
    assert all("--get-comments" in c.args for c in yt.labels("metadata"))
    info = json.loads((session_kit.out / PUBLIC / "info.json").read_text())
    assert info["comments"], "the refreshed extraction still carries the comments"
    document = results[0].document
    assert document is not None and document.video.creator_note is not None


def test_a_present_video_still_waives_the_cooldown(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    run(session_kit, yt, options=video_options(on=True))
    session.start_cooldown(Signal.RATE_LIMITED, NOW)
    calls = len(yt.calls)

    report, results, _ = run(session_kit, yt, options=video_options(on=True))

    assert report.exit_code == 0 and results[0].outcome.reused == ("context",)
    assert len(yt.calls) == calls


def test_a_failed_video_probe_does_not_abort_the_batch_precheck(session_kit, worker: FakeWorker, fake_proc) -> None:
    yt = session_kit.yt()
    run(session_kit, yt, options=video_options(on=True))
    session.start_cooldown(Signal.RATE_LIMITED, NOW)

    def timed_out(argv: list[str]) -> proc.ProcResult:
        raise ProcTimeout("ffprobe", 20.0)

    fake_proc.responses["ffprobe"] = timed_out
    with pytest.raises(session.BatchStopError):
        run(session_kit, yt, options=video_options(on=True))


def test_a_video_failure_on_the_reuse_path_does_not_stop_the_next_video(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    targets = [PUBLIC, "pubVideo002"]
    run(session_kit, yt, targets)
    yt.failures[("video", False)] = [session_kit.fail(Signal.NO_FORMATS)]

    report, results, _ = run(session_kit, yt, targets, options=video_options(on=True))

    assert [r.outcome.status for r in results] == [session.VideoStatus.ERROR, session.VideoStatus.OK]
    assert report.exit_code == 1
    assert not (session_kit.out / PUBLIC / "video.mp4").exists()
    assert (session_kit.out / "pubVideo002" / "video.mp4").is_file()


def test_a_failed_video_download_fails_a_current_document_loudly(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    run(session_kit, yt)
    yt.failures[("video", False)] = [session_kit.fail(Signal.NO_FORMATS)]
    with pytest.raises(YtDlpError):
        process(PUBLIC, video_options(on=True), out=session_kit.out, runner_factory=lambda _sleep: yt)


def test_a_cooldown_still_refuses_a_current_document_whose_video_is_missing(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    run(session_kit, yt)
    session.start_cooldown(Signal.RATE_LIMITED, NOW)
    calls = len(yt.calls)
    with pytest.raises(session.BatchStopError):
        run(session_kit, yt, options=video_options(on=True))
    assert len(yt.calls) == calls


def test_the_video_option_is_not_part_of_the_run_key(session_kit, worker: FakeWorker) -> None:
    assert (
        settings_for(session_kit, video_options(on=True)).run_key()
        == settings_for(session_kit, config.load_options()).run_key()
    )


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
    assert Path(request.out_dir).parent.name == "q-base"


def test_an_unknown_engine_fails_before_any_request(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    with pytest.raises(ConfigError, match="unknown ASR hypothesis"):
        run(session_kit, yt, options=engine_options("nope"))
    assert yt.calls == []


def test_captions_as_the_source_fall_back_to_asr_without_a_manual_track(session_kit, worker: FakeWorker) -> None:
    options = config.load_options(overrides={"asr": {"source": "captions"}})
    yt = session_kit.yt()
    _, results, _ = run(session_kit, yt, options=options)
    doc = results[0].document
    assert doc is not None and doc.transcript.runs[0].kind == "asr" and len(worker.calls) == 1
    assert any("no manual caption track in en" in w for w in doc.quality.warnings)
    assert results[0].notices == ("no manual caption track in en; transcribed with ASR instead",)
    assert len(yt.labels("audio")) == 1, "the audio is fetched only once ASR turns out to be needed"


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


def batch_ids(kit) -> list[tuple[str, str]]:
    batch = json.loads((kit.out / "batch_report.json").read_text(encoding="utf-8"))
    return [(video["id"], video["status"]) for video in batch["videos"]]


def test_a_chapter_ending_before_it_starts_fails_that_video_only(
    session_kit, worker: FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = pipeline.load_metadata

    def backwards_chapters(vdir: Path) -> VideoMetadata:
        metadata = real(vdir)
        return replace(metadata, chapters=(Chapter("Backwards", 50.0, 10.0),)) if vdir.name == PUBLIC else metadata

    monkeypatch.setattr(pipeline, "load_metadata", backwards_chapters)
    report, results, _ = run(session_kit, session_kit.yt(), [PUBLIC, "pubVideo002"])
    first, second = (r.outcome for r in results)
    assert first.error == "contract" and "does not fit the contract" in first.detail
    assert isinstance(results[0].error, ContractError)
    assert second.status is session.VideoStatus.OK and report.exit_code == 1
    assert batch_ids(session_kit) == [(PUBLIC, "error"), ("pubVideo002", "ok")]
    assert not (session_kit.out / PUBLIC / "context.json").exists()
    with pytest.raises(ContractError):
        process(PUBLIC, out=session_kit.out, runner_factory=lambda _sleep: session_kit.yt())


def test_a_malformed_hypothesis_row_fails_that_video_only(session_kit, worker: FakeWorker) -> None:
    worker.garbage[PUBLIC] = '{"hypothesis": "w-prod", "file": "pubVideo001", "chunk": "zero"'
    report, results, _ = run(session_kit, session_kit.yt(), [PUBLIC, "pubVideo002"])
    first, second = (r.outcome for r in results)
    assert first.error == "contract" and "hypotheses.jsonl line 2" in first.detail
    assert second.status is session.VideoStatus.OK and report.exit_code == 1
    assert batch_ids(session_kit) == [(PUBLIC, "error"), ("pubVideo002", "ok")]


def test_an_os_error_is_a_per_video_error_and_process_raises_it_unchanged(
    session_kit, worker: FakeWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == "audio.wav" and self.parent.name == PUBLIC:
            raise PermissionError(13, "synthetic: permission denied", str(self))
        real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    report, results, _ = run(session_kit, session_kit.yt(), [PUBLIC, "pubVideo002"])
    first, second = (r.outcome for r in results)
    assert first.error == "os_error" and first.detail.startswith("PermissionError:")
    assert str(session_kit.home) not in first.detail
    assert second.status is session.VideoStatus.OK and report.exit_code == 1
    assert batch_ids(session_kit) == [(PUBLIC, "error"), ("pubVideo002", "ok")]
    with pytest.raises(PermissionError):
        process(PUBLIC, out=session_kit.out, runner_factory=lambda _sleep: session_kit.yt(), force=True)


def test_a_stop_seen_while_expanding_a_playlist_stops_before_any_video(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    yt.playlist = ["plVideo0001", "plVideo0002"]
    yt.secondary[("expand", False)] = (Signal.BOT_CHECK,)
    with pytest.raises(session.BatchStopError) as stopped:
        run(session_kit, yt, ["https://www.youtube.com/playlist?list=PLsynthetic"])
    assert stopped.value.signal is Signal.BOT_CHECK
    assert yt.labels("metadata") == [] and worker.calls == []
    assert session.read_cooldown() is not None


def test_process_returns_a_finished_document_despite_a_stop_on_a_successful_call(
    session_kit, worker: FakeWorker, caplog: pytest.LogCaptureFixture
) -> None:
    yt = session_kit.yt()
    yt.secondary[("metadata", False)] = (Signal.BOT_CHECK,)
    with caplog.at_level(logging.WARNING, logger="ytverbatim.pipeline"):
        doc = process(PUBLIC, out=session_kit.out, runner_factory=lambda _sleep: yt)
    assert isinstance(doc, ContextDocument) and len(doc.segments) == 2
    assert "back off" in caplog.text
    cooldown = session.read_cooldown()
    assert cooldown is not None and cooldown.signal is Signal.BOT_CHECK


def test_a_missing_engine_extra_fails_before_any_request(
    session_kit, worker: FakeWorker, monkeypatch: pytest.MonkeyPatch, engine_installed: list[str]
) -> None:
    monkeypatch.setattr(registry, "find_spec", lambda name: None)
    yt = session_kit.yt()
    with pytest.raises(DependencyError, match=r"ytverbatim\[whisper\]"):
        run(session_kit, yt)
    with pytest.raises(DependencyError, match=r"ytverbatim\[asr\]"):
        run(session_kit, yt, options=engine_options("q-base"))
    with pytest.raises(DependencyError, match=r"ytverbatim\[whisper,asr\]"):
        run(session_kit, yt, options=engine_options("w-vad"))
    assert yt.calls == [] and worker.calls == []


def test_the_dependency_check_asks_for_the_adapter_modules(session_kit, worker: FakeWorker, engine_installed) -> None:
    run(session_kit, session_kit.yt())
    assert engine_installed == ["mlx_whisper"]


def test_a_context_from_another_engine_is_transcribed_again(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    run(session_kit, yt)
    _, results, events = run(session_kit, yt, options=engine_options("q-base"))
    assert [call.engine.spec.id for call in worker.calls] == ["w-prod", "q-base"]
    assert "skip" not in [e.stage for e in events]
    doc = load_document(session_kit.out / PUBLIC / "context.json")
    assert doc.transcript.runs[0].engine == "q-base" and results[0].document == doc

    _, again, _ = run(session_kit, yt, options=engine_options("q-base"))
    assert len(worker.calls) == 2 and again[0].outcome.reused == ("context",)


def test_a_cooldown_does_not_refuse_a_batch_that_needs_no_request(session_kit, worker: FakeWorker) -> None:
    yt = session_kit.yt()
    run(session_kit, yt)
    calls = len(yt.calls)
    session.start_cooldown(Signal.RATE_LIMITED, NOW)

    report, results, _ = run(session_kit, yt)
    assert report.exit_code == 0 and results[0].outcome.reused == ("context",)
    assert len(yt.calls) == calls
    with pytest.raises(session.BatchStopError):
        run(session_kit, yt, [PUBLIC, "pubVideo002"])
    with pytest.raises(session.BatchStopError):
        run(session_kit, yt, options=engine_options("q-base"))
    assert len(yt.calls) == calls


def invoke(kit, yt, monkeypatch: pytest.MonkeyPatch, *args: str):
    monkeypatch.setattr(session, "SubprocessRunner", lambda *, sleep_args=(): yt)
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
