"""The full `run` pipeline with a fake yt-dlp, a fake ASR worker and a fake LLM: stages, store, LLM and captions."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from ytverbatim import config, pipeline, store
from ytverbatim.acquire import session
from ytverbatim.asr import registry
from ytverbatim.asr import runner as asr_runner
from ytverbatim.asr.models import HYPOTHESES_FILE, ChunkHypothesis, RunMetrics, WorkerRequest
from ytverbatim.asr.runner import WorkerResult
from ytverbatim.asr.worker import _claim_output as claim_output
from ytverbatim.cli import main
from ytverbatim.contract import ContextDocument, Word, load_document
from ytverbatim.errors import ConfigError, DependencyError, EngineError, LLMUnavailableError
from ytverbatim.llm import claude_cli
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage
from ytverbatim.pipeline import ProgressEvent, VideoResult, process_many
from ytverbatim.text import punctuate as punctuate_module

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


def zh_rows(request: WorkerRequest, *, pause: float = 0.5) -> list[ChunkHypothesis]:
    words = tuple(
        Word(text=phrase, start=3.0 * index, end=3.0 * index + 3.0 - pause, conf=0.9)
        for index, phrase in enumerate(ZH_PHRASES)
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


# Whisper's usual English failure: the opening stretch lowercase and unpunctuated, the rest punctuated.
EN_OPENING = (
    "so today we are going to talk about how the market reacts when the fed changes its mind about rates "
    "and why that matters for anyone who holds stocks or bonds right now because the next few months could be "
    "very different from the last few"
)
EN_REST = "Markets rose today. Then they fell sharply. Nobody was surprised by that outcome at all."


def en_rows(request: WorkerRequest) -> list[ChunkHypothesis]:
    texts = (EN_OPENING, EN_REST)
    return [
        ChunkHypothesis(
            hypothesis=request.engine.spec.id,
            file=request.files[0].id,
            chunk=index,
            start=10.0 * index,
            end=10.0 * index + 8.0,
            text=text,
            language="en",
        )
        for index, text in enumerate(texts)
    ]


@dataclass
class FakeWorker:
    """Claims its output folder exactly as the real worker does, then writes the rows a real decode would."""

    calls: list[WorkerRequest] = field(default_factory=list[WorkerRequest])
    # Raise this after claiming the folder, as a worker killed mid-decode would.
    error: EngineError | None = None
    # Seconds between the Mandarin phrases; from 0.6 s each phrase becomes a stretch of its own.
    pause: float = 0.5

    def __call__(self, request: WorkerRequest, *, python: str | None = None) -> WorkerResult:
        self.calls.append(request)
        out = Path(request.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        claim_output(out, request)
        if self.error is not None:
            raise self.error
        rows = en_rows(request) if request.files[0].id == EN else zh_rows(request, pause=self.pause)
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
            reply: object = {"items": [{"i": it["i"], "text": _punctuated(it["text"])} for it in items]}
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


def _punctuated(text: str) -> str:
    if text.isascii():
        return text[:1].upper() + text[1:] + "."
    return text.replace(" ", "，") + "。"


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
    monkeypatch.setattr(pipeline, "check_ready", lambda _options: None)
    result = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--llm", "claude-cli:claude-sonnet-5-5")
    assert result.exit_code == 0, result.output
    assert (built[0].provider, built[0].model) == ("claude-cli", "claude-sonnet-5-5")
    assert (
        "punctuation: llm" in result.output
        and "document LLM tokens: punctuation 1 call(s), 100 in / 20 out" in result.output
    )

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


def test_only_the_unpunctuated_stretch_of_a_punctuated_video_goes_to_the_llm(session_kit, worker) -> None:
    llm = FakeLLM()
    _, results, _ = run(session_kit, session_kit.yt(), [EN], llm=llm)
    doc = only_document(results)
    (asked,) = [r for r in llm.requests if '"items"' in r.user]
    sent = [item["text"] for item in json.loads(asked.user.split("SEGMENTS:\n", 1)[1])]
    assert len(sent) >= 2 and " ".join(sent) in EN_OPENING, "only the unpunctuated opening is sent"
    assert doc.segments[0].text.startswith("So today") and doc.transcript.punctuation == "llm"
    assert [s.text for s in doc.segments[-2:]] == ["Then they fell sharply.", EN_REST[44:]]


def set_info(kit, video: str, **fields: object) -> None:
    info = kit.out / video / "info.json"
    data = json.loads(info.read_text(encoding="utf-8"))
    info.write_text(json.dumps({**data, **fields}, ensure_ascii=False), encoding="utf-8")


def worker_dirs(kit, video: str = ZH) -> list[str]:
    root = kit.out / video / ".stages" / "asr"
    return sorted(str(path.relative_to(root)) for path in root.glob("*/*"))


def test_a_changed_asr_request_decodes_into_a_fresh_folder_and_prunes_the_old_one(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt)
    (first,) = worker_dirs(session_kit)
    # The folder V0 used for every request of an engine, holding another request's output: the worker refuses it.
    legacy = session_kit.out / ZH / ".stages" / "asr" / "w-prod"
    (legacy / "fingerprint.json").write_text('{"engine": "something else"}', encoding="utf-8")

    set_info(session_kit, ZH, language="zh-Hant")
    store.StageStore(session_kit.out / ZH).discard("context")
    report, results, _ = run(session_kit, yt)
    assert report.exit_code == 0, results[0].outcome
    assert stored(results[0])["asr"] is False and len(worker.calls) == 2
    (second,) = worker_dirs(session_kit)
    assert second != first and worker.calls[1].files[0].language == "zh-Hant"
    assert not (legacy / "fingerprint.json").exists(), "stale folders go once the new stage is stored"


def test_switching_engines_keeps_the_other_engines_output(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt)
    store.StageStore(session_kit.out / ZH).discard("context")
    run(session_kit, yt, options=config.load_options(overrides={"asr": {"engine": "w-vad"}}))
    assert [path.split("/")[0] for path in worker_dirs(session_kit)] == ["w-prod", "w-vad"]


def test_an_interrupted_asr_run_resumes_in_its_own_folder(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    worker.error = EngineError("killed mid-decode")
    report, _, _ = run(session_kit, yt)
    assert report.exit_code == 1
    worker.error = None
    report, _, _ = run(session_kit, yt)
    assert report.exit_code == 0
    assert worker.calls[0].out_dir == worker.calls[1].out_dir and len(worker_dirs(session_kit)) == 1


def test_the_memory_cap_reaches_the_worker_but_does_not_miss_the_stored_asr(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    capped = config.load_options(overrides={"asr": {"memory_cap_gib": 1.0}})
    run(session_kit, yt, options=capped)
    assert worker.calls[0].memory_cap_bytes == 1 << 30
    store.StageStore(session_kit.out / ZH).discard("context")
    _, results, _ = run(session_kit, yt)
    assert stored(results[0])["asr"] is True and len(worker.calls) == 1


def test_a_module_version_bump_redoes_its_stage_and_the_document(session_kit, worker, monkeypatch) -> None:
    yt = zh_yt(session_kit)
    llm = FakeLLM()
    run(session_kit, yt, llm=llm)
    monkeypatch.setattr(punctuate_module, "PROMPT_VERSION", punctuate_module.PROMPT_VERSION + "-bumped")
    _, results, events = run(session_kit, yt, llm=llm)
    assert "skip" not in [e.stage for e in events], "the context record names the module versions"
    # Same reply, so the text after punctuation is unchanged and numeric is reused.
    assert stored(results[0]) == {"asr": True, "normalize": True, "punctuate": False, "numeric": True}


def test_correction_reruns_when_its_glossary_changes(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    correct = config.load_options(overrides={"text": {"correct": True}})
    run(session_kit, yt, options=correct, llm=FakeLLM())
    set_info(session_kit, ZH, tags=["估值"])
    store.StageStore(session_kit.out / ZH).discard("context")
    _, results, _ = run(session_kit, yt, options=correct, llm=FakeLLM())
    assert stored(results[0])["correct"] is False and stored(results[0])["punctuate"] is True


def test_numbers_without_independent_readings_are_called_unchecked(session_kit, worker, monkeypatch) -> None:
    yt = zh_yt(session_kit)
    yt.tracks[ZH] = {}
    _, results, _ = run(session_kit, yt, llm=FakeLLM())
    doc = only_document(results)
    (notice,) = results[0].notices
    assert notice.startswith("numbers not cross-checked: no captions in zh and no second opinion")
    assert notice in doc.quality.warnings
    cli = invoke(session_kit, zh_yt(session_kit), monkeypatch, "zhVideo0002")
    assert "! numbers not cross-checked" in cli.output


def test_a_run_without_captions_does_not_count_for_a_run_with_them(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt, captions=False)
    assert yt.labels("captions") == []
    _, results, events = run(session_kit, yt)
    assert "skip" not in [e.stage for e in events] and len(yt.labels("captions")) == 1
    assert any(f.kind == "numeric_disputed" for s in only_document(results).segments for f in s.flags)


def test_llm_usage_is_what_the_document_cost_even_when_every_batch_is_reused(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    _, first, _ = run(session_kit, yt, llm=FakeLLM())
    stages_dir = session_kit.out / ZH / ".stages"
    for name in ("punctuate.json", "numeric.json", "context.json"):
        (stages_dir / name).unlink()
    llm = FakeLLM()
    _, again, _ = run(session_kit, yt, llm=llm)
    assert llm.requests == [] and stored(again[0])["punctuate"] is False, "recomputed from batch checkpoints"
    assert only_document(again).llm_usage == only_document(first).llm_usage != ()


def test_a_resumed_partial_stage_counts_every_batch(session_kit, worker, monkeypatch) -> None:
    monkeypatch.setattr(punctuate_module, "BATCH_CHARS", 8)
    worker.pause = 1.0
    yt = zh_yt(session_kit)
    failing = FakeLLM()
    original = failing.complete

    def second_call_fails(request):
        if len(failing.requests) == 1:
            failing.requests.append(request)
            raise LLMUnavailableError("synthetic outage")
        return original(request)

    failing.complete = second_call_fails  # type: ignore[method-assign]
    report, _, _ = run(session_kit, yt, llm=failing)
    assert report.exit_code == 1
    llm = FakeLLM()
    _, results, _ = run(session_kit, yt, llm=llm)
    batches = len([r for r in llm.requests if '"items"' in r.user]) + 1
    (punctuation,) = [u for u in only_document(results).llm_usage if u.stage == "punctuation"]
    assert batches >= 3 and punctuation.calls == batches


def test_a_missing_claude_cli_fails_before_any_request(session_kit, worker, monkeypatch) -> None:
    monkeypatch.setattr(claude_cli.shutil, "which", lambda _name: None)
    yt = zh_yt(session_kit)
    options = config.load_options(overrides={"llm": {"provider": "claude-cli", "model": "claude-sonnet-5-5"}})
    with pytest.raises(DependencyError, match="claude CLI not found"):
        run(session_kit, yt, options=options)
    assert yt.calls == [] and worker.calls == []


def test_cli_llm_spec_switches_provider_over_the_configured_one(session_kit, worker, monkeypatch) -> None:
    config_file = session_kit.home / ".config" / "ytverbatim" / "config.toml"
    config_file.parent.mkdir(parents=True)
    config_file.write_text('[llm]\nprovider = "openai-compat"\nmodel = "m"\nbase_url = "http://127.0.0.1:8080/v1"\n')
    built: list[config.LLMOptions] = []

    def client_from_options(options: config.LLMOptions) -> FakeLLM:
        built.append(options)
        return FakeLLM()

    monkeypatch.setattr(pipeline, "client_from_options", client_from_options)
    monkeypatch.setattr(pipeline, "check_ready", lambda _options: None)
    result = invoke(session_kit, zh_yt(session_kit), monkeypatch, ZH, "--llm", "claude-cli:claude-sonnet-5-5")
    assert result.exit_code == 0, result.output
    assert (built[0].provider, built[0].model, built[0].base_url) == ("claude-cli", "claude-sonnet-5-5", None)


def test_a_skipped_video_gets_its_markdown_back_and_an_unloadable_document_is_rebuilt(session_kit, worker) -> None:
    yt = zh_yt(session_kit)
    run(session_kit, yt)
    vdir = session_kit.out / ZH
    expected = (vdir / "context.md").read_text(encoding="utf-8")
    (vdir / "context.md").unlink()
    _, _, events = run(session_kit, yt)
    assert [e.stage for e in events] == ["skip"] and (vdir / "context.md").read_text(encoding="utf-8") == expected

    broken = '{"schema_version": "9.0"}'
    (vdir / "context.json").write_text(broken, encoding="utf-8")
    record = json.loads((vdir / ".stages" / "context.json").read_text(encoding="utf-8"))
    record["output"]["sha256"] = __import__("hashlib").sha256(broken.encode()).hexdigest()
    (vdir / ".stages" / "context.json").write_text(json.dumps(record), encoding="utf-8")
    report, results, events = run(session_kit, yt)
    assert report.exit_code == 0 and "skip" not in [e.stage for e in events]
    assert load_document(vdir / "context.json") == only_document(results)


def test_captions_mode_says_why_it_cannot_choose_among_manual_tracks(session_kit, worker) -> None:
    yt = session_kit.yt()
    yt.language[EN] = None  # type: ignore[assignment]
    yt.tracks[EN] = {"en": json.dumps(EN_MANUAL), "zh": json.dumps(EN_MANUAL)}
    options = config.load_options(overrides={"asr": {"source": "captions"}})
    _, results, _ = run(session_kit, yt, [EN], options=options)
    assert results[0].notices[0] == (
        "2 manual caption tracks and no declared video language to choose one; transcribed with ASR instead"
    )
