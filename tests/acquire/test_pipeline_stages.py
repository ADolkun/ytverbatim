"""The full `run` pipeline with a fake yt-dlp, a fake ASR worker and a fake LLM: stages, store, LLM and captions."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from ytverbatim import config, pipeline
from ytverbatim.acquire import session
from ytverbatim.asr import registry
from ytverbatim.asr import runner as asr_runner
from ytverbatim.asr.models import HYPOTHESES_FILE, ChunkHypothesis, RunMetrics, WorkerRequest
from ytverbatim.asr.runner import WorkerResult
from ytverbatim.cli import main
from ytverbatim.contract import ContextDocument, Word, load_document
from ytverbatim.errors import ConfigError, LLMUnavailableError
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage
from ytverbatim.pipeline import ProgressEvent, VideoResult, process_many

ZH = "zhVideo0001"
EN = "enVideo0001"
NOW = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)
# Whisper-style Mandarin: no punctuation, a space at each pause. The third phrase holds the number in dispute.
ZH_PHRASES = ("大家好欢迎回来", "今天聊聊这只股票", "上周涨了十五个百分点", "非常强劲")
ZH_AUTO = {
    "events": [
        {
            "tStartMs": 0,
            "dDurationMs": 12000,
            "segs": [
                {"utf8": "大家好欢迎回来"},
                {"utf8": "今天聊聊这只股票", "tOffsetMs": 3000},
                {"utf8": "上周涨了五十个百分点", "tOffsetMs": 6000},
                {"utf8": "非常强劲", "tOffsetMs": 9000},
            ],
        }
    ]
}
EN_MANUAL = {
    "events": [
        {"tStartMs": 5000, "dDurationMs": 3000, "segs": [{"utf8": "Markets rose today."}]},
        {"tStartMs": 60000, "dDurationMs": 3000, "segs": [{"utf8": "Then they fell sharply."}]},
    ]
}


def zh_rows(request: WorkerRequest) -> list[ChunkHypothesis]:
    words = tuple(
        Word(text=phrase, start=3.0 * index, end=3.0 * index + 2.5, conf=0.9) for index, phrase in enumerate(ZH_PHRASES)
    )
    return [
        ChunkHypothesis(
            hypothesis=request.engine.spec.id,
            file=request.files[0].id,
            chunk=0,
            start=0.0,
            end=12.0,
            text=" ".join(ZH_PHRASES),
            language="zh",
            words=words,
        )
    ]


@dataclass
class FakeWorker:
    calls: list[WorkerRequest] = field(default_factory=list[WorkerRequest])

    def __call__(self, request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
        self.calls.append(request)
        out = Path(request.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = zh_rows(request)
        (out / HYPOTHESES_FILE).write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
        metrics = RunMetrics(
            hypothesis=request.engine.spec.id,
            files=1,
            chunks_decoded=1,
            chunks_skipped=0,
            load_s=0.0,
            decode_s=0.0,
            audio_s=12.0,
            peak_memory_bytes=None,
            max_rss_bytes=0,
        )
        return WorkerResult(out / HYPOTHESES_FILE, metrics, 0, 0.0)


@dataclass
class FakeLLM:
    """Punctuates by turning pause spaces into commas and ending with 。; picks the caption reading in disputes."""

    model: str = "fake-model-1"
    fail: bool = False
    requests: list[LLMRequest] = field(default_factory=list[LLMRequest])

    @property
    def id(self) -> str:
        return f"fake:{self.model}"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.fail:
            raise LLMUnavailableError(f"{self.id}: synthetic outage")
        if "SEGMENTS:" in request.user and '"items"' in request.user:
            items = json.loads(request.user.split("SEGMENTS:\n", 1)[1])
            reply: object = {"items": [{"i": it["i"], "text": it["text"].replace(" ", "，") + "。"} for it in items]}
        elif "SEGMENTS:" in request.user:
            reply = {"fixed": []}
        else:
            items = json.loads(request.user.split("ITEMS:\n", 1)[1])
            reply = {
                "verdicts": [
                    {"id": it["id"], "value": it["readings"][-1], "confidence": "high", "reason": "context"}
                    for it in items
                ]
            }
        return LLMResponse(text=json.dumps(reply, ensure_ascii=False), usage=Usage(100, 20), elapsed_s=0.0)

    def stages(self) -> list[str]:
        return [
            "numeric" if "ITEMS:" in r.user else "punctuate" if '"items"' in r.user else "correct"
            for r in self.requests
        ]


@pytest.fixture(autouse=True)
def engine_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "find_spec", lambda _name: object())


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> FakeWorker:
    fake = FakeWorker()
    monkeypatch.setattr(asr_runner, "run_worker", fake)
    return fake


def zh_yt(kit):
    yt = kit.yt()
    yt.language[ZH] = "zh"
    yt.tracks[ZH] = {"zh-orig": json.dumps(ZH_AUTO, ensure_ascii=False)}
    return yt


def run(kit, yt, targets: Sequence[str] = (ZH,), *, options: config.Options | None = None, **kwargs: object):
    results: list[VideoResult] = []
    events: list[ProgressEvent] = []
    report = process_many(
        targets,
        options or config.load_options(),
        out=kit.out,
        runner_factory=lambda _sleep: yt,
        sleep=kit.slept.append,
        clock=lambda: NOW,
        progress=events.append,
        on_video=results.append,
        **kwargs,  # type: ignore[arg-type]
    )
    return report, results, events


def only_document(results: list[VideoResult]) -> ContextDocument:
    (result,) = results
    assert result.document is not None, result.outcome
    return result.document


def stored(result: VideoResult) -> dict[str, bool]:
    return {t.stage: t.reused for t in result.timings if t.stage not in ("fetch", "write")}


def ffmpeg_calls(kit) -> int:
    return sum(1 for call in kit.proc.calls if Path(call[0]).name == "ffmpeg")


def test_end_to_end_with_a_fake_llm_punctuates_verifies_numbers_and_meters_tokens(session_kit, worker) -> None:
    llm = FakeLLM()
    report, results, events = run(session_kit, zh_yt(session_kit), llm=llm)
    assert report.exit_code == 0
    doc = only_document(results)
    vdir = session_kit.out / ZH
    assert load_document(vdir / "context.json") == doc

    assert doc.transcript.punctuation == "llm" and doc.transcript.language == "zh"
    text = "".join(s.text for s in doc.segments)
    assert text == "大家好欢迎回来，今天聊聊这只股票，上周涨了五十个百分点，非常强劲。"
    fixed = [(s, f) for s in doc.segments for f in s.flags if f.kind == "numeric_fixed"]
    ((segment, flag),) = fixed
    assert segment.text[flag.char_start : flag.char_end] == "五十个百分点"
    assert [r.source for r in flag.readings] == ["asr:w-prod", "caption_auto"]
    assert ("numeric", "十五", "五十") in [(e.stage, e.before, e.after) for e in segment.edits]
    # The fixed word no longer matches its engine word, so its time is interpolated (JSON only).
    assert doc.quality.flags_by_kind == {"numeric_fixed": 1, "timing_interpolated": 1}
    assert {(u.stage, u.client, u.calls, u.input_tokens) for u in doc.llm_usage} == {
        ("punctuation", llm.id, 1, 100),
        ("numeric", llm.id, 1, 100),
    }

    markdown = (vdir / "context.md").read_text(encoding="utf-8")
    assert "五十个百分点[^1]" in markdown and "十五个百分点 → 五十个百分点" in markdown
    assert "punctuation: llm" in markdown
    assert "timing_interpolated" not in markdown
    assert [e.stage for e in events] == ["fetch", "asr", "normalize", "punctuate", "numeric", "write"]
    assert results[0].notices == () and not (vdir / "audio.wav").exists()
    assert {".lock", "audio.webm", "context.json", "context.md", ".stages"} <= {p.name for p in vdir.iterdir()}


def test_without_an_llm_the_stages_degrade_and_say_so(session_kit, worker) -> None:
    _, results, _ = run(session_kit, zh_yt(session_kit))
    doc = only_document(results)
    assert doc.transcript.punctuation == "none" and doc.llm_usage == ()
    assert len(doc.segments) >= 1 and "，" not in "".join(s.text for s in doc.segments)
    (disputed,) = [f for s in doc.segments for f in s.flags]
    assert disputed.kind == "numeric_disputed" and "no LLM" in disputed.note
    notices = results[0].notices
    assert len(notices) == 2
    assert notices[0].startswith("punctuation not restored") and notices[1].startswith("1 disputed number(s)")
    assert set(notices) <= set(doc.quality.warnings)


def test_an_option_change_recomputes_only_the_stages_downstream_of_it(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    _, first, _ = run(session_kit, yt)
    assert stored(first[0]) == {"asr": False, "normalize": False, "punctuate": False, "numeric": False}
    converted = ffmpeg_calls(session_kit)

    llm = FakeLLM()
    _, second, events = run(session_kit, yt, llm=llm)
    assert stored(second[0]) == {"asr": True, "normalize": True, "punctuate": False, "numeric": False}
    assert len(worker.calls) == 1 and ffmpeg_calls(session_kit) == converted, "no audio once the ASR is stored"
    assert len(yt.labels("audio")) == 1 and "asr" not in [e.stage for e in events]
    assert only_document(second).transcript.punctuation == "llm"

    asked = len(llm.requests)
    no_chapters = config.load_options(overrides={"text": {"chapters": "none"}})
    _, third, _ = run(session_kit, yt, options=no_chapters, llm=llm)
    assert stored(third[0]) == {"asr": True, "normalize": True, "punctuate": True, "numeric": True}
    assert len(llm.requests) == asked and only_document(third).chapters == ()

    keep_script = config.load_options(overrides={"text": {"script": "keep"}})
    _, fourth, _ = run(session_kit, yt, options=keep_script, llm=llm)
    # The text was Simplified already, so normalize's output is unchanged and nothing after it reruns.
    assert stored(fourth[0]) == {"asr": True, "normalize": False, "punctuate": True, "numeric": True}
    assert len(llm.requests) == asked


def test_unchanged_options_reuse_the_document_without_any_stage(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    llm = FakeLLM()
    run(session_kit, yt, llm=llm)
    calls, asked = len(yt.calls), len(llm.requests)
    _, again, events = run(session_kit, yt, llm=FakeLLM())
    assert [e.stage for e in events] == ["skip"] and again[0].outcome.reused == ("context",)
    assert len(yt.calls) == calls and len(llm.requests) == asked

    (session_kit.out / ZH / "context.json").write_text("{}", encoding="utf-8")
    _, edited, events = run(session_kit, yt, llm=FakeLLM())
    assert "skip" not in [e.stage for e in events], "a context.json that is not the one written is rebuilt"
    assert all(stored(edited[0]).values()) and len(worker.calls) == 1


def test_an_llm_failure_fails_the_video_and_keeps_every_finished_stage(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    report, results, _ = run(session_kit, yt, llm=FakeLLM(fail=True))
    assert (
        report.exit_code == 1 and results[0].outcome.error == "llm" and "synthetic outage" in results[0].outcome.detail
    )
    stages_dir = session_kit.out / ZH / ".stages"
    assert {p.name for p in stages_dir.iterdir() if p.is_file()} == {"asr.json", "normalize.json"}
    assert not (session_kit.out / ZH / "context.json").exists()

    _, retry, _ = run(session_kit, yt, llm=FakeLLM())
    assert stored(retry[0]) == {"asr": True, "normalize": True, "punctuate": False, "numeric": False}
    assert len(worker.calls) == 1 and len(yt.labels("audio")) == 1


def test_force_discards_every_stored_stage(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt, llm=FakeLLM())
    llm = FakeLLM()
    _, forced, _ = run(session_kit, yt, llm=llm, force=True)
    assert not any(stored(forced[0]).values())
    assert len(worker.calls) == 2 and llm.stages() == ["punctuate", "numeric"]


def test_captions_mode_trusts_the_manual_track_and_skips_the_audio(session_kit, worker) -> None:
    yt = session_kit.yt()
    yt.tracks[EN] = {"en": json.dumps(EN_MANUAL)}
    options = config.load_options(overrides={"asr": {"source": "captions"}})
    _, results, events = run(session_kit, yt, [EN], options=options)
    doc = only_document(results)
    assert worker.calls == [] and yt.labels("audio") == [] and ffmpeg_calls(session_kit) == 0
    assert [e.stage for e in events] == ["fetch", "captions", "normalize", "punctuate", "numeric", "write"]
    assert doc.transcript.runs[0].kind == "caption_manual" and doc.transcript.punctuation == "caption"
    assert [s.text for s in doc.segments] == ["Markets rose today.", "Then they fell sharply."]
    assert [s.chapter for s in doc.segments] == [0, 1]
    assert "Transcript: manual captions" in (session_kit.out / EN / "context.md").read_text(encoding="utf-8")


def test_correction_needs_an_llm_and_runs_with_one(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    correct = config.load_options(overrides={"text": {"correct": True}})
    with pytest.raises(ConfigError, match="correction needs an LLM"):
        run(session_kit, yt, options=correct)
    assert yt.calls == []

    llm = FakeLLM()
    _, results, _ = run(session_kit, yt, options=correct, llm=llm)
    assert only_document(results).transcript.correction_llm == llm.id
    assert llm.stages() == ["punctuate", "correct", "numeric"]


def test_a_whole_file_engine_is_refused_as_the_numeric_second_opinion(session_kit, worker) -> None:
    options = config.load_options(overrides={"text": {"numeric_second_opinion": "w-prod"}})
    with pytest.raises(ConfigError, match="decodes whole files"):
        run(session_kit, zh_yt(session_kit), options=options)


def invoke(kit, yt, monkeypatch: pytest.MonkeyPatch, *args: str):
    monkeypatch.setattr(session, "SubprocessRunner", lambda *, sleep_args=(): yt)
    return CliRunner().invoke(main, ["run", "--out", str(kit.out), *args])


def test_cli_without_an_llm_prints_a_notice_per_degraded_stage(session_kit, worker, monkeypatch) -> None:
    result = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH)
    assert result.exit_code == 0, result.output
    assert "! punctuation not restored" in result.output and "punctuation: none" in result.output
    assert "stages: fetch" in result.output and "asr 0s" in result.output


def test_cli_llm_spec_reaches_the_pipeline(session_kit, worker, monkeypatch) -> None:
    built: list[config.LLMOptions] = []
    llm = FakeLLM(model="claude-sonnet-5-5")

    def client_from_options(options: config.LLMOptions) -> FakeLLM:
        built.append(options)
        return llm

    monkeypatch.setattr(pipeline, "client_from_options", client_from_options)
    result = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--llm", "claude-cli:claude-sonnet-5-5")
    assert result.exit_code == 0, result.output
    assert (built[0].provider, built[0].model) == ("claude-cli", "claude-sonnet-5-5")
    assert "punctuation: llm" in result.output and "llm: punctuation 1 call(s), 100 in / 20 out" in result.output

    bad = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--llm", "sonnet")
    assert bad.exit_code == 1 and "PROVIDER:MODEL" in bad.output
    refused = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--llm", "none", "--correct")
    assert refused.exit_code == 1 and "correction needs an LLM" in refused.output


def test_render_rewrites_the_markdown_from_context_json(session_kit, worker, monkeypatch) -> None:
    invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH)
    vdir = session_kit.out / ZH
    expected = (vdir / "context.md").read_text(encoding="utf-8")
    (vdir / "context.md").unlink()
    result = CliRunner().invoke(main, ["render", str(vdir / "context.json")])
    assert result.exit_code == 0, result.output
    assert (vdir / "context.md").read_text(encoding="utf-8") == expected
    printed = CliRunner().invoke(main, ["render", "--stdout", str(vdir / "context.json")])
    assert printed.output == expected
    (vdir / "broken.json").write_text("{}", encoding="utf-8")
    assert CliRunner().invoke(main, ["render", str(vdir / "broken.json")]).exit_code == 1
