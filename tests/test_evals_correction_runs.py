import contextlib
import fcntl
import json
import os
from collections.abc import Callable, Generator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from evals_fixture import EN, ZH, make_eval, whole_file_rows

from evals import correction, disk, run_correction
from evals.correction import (
    CANDIDATES,
    BaseFile,
    CandidateFile,
    RecordingClient,
    ServerSeams,
    build_base,
    cleanup_candidate,
    first_try_validity,
    load_base,
    load_candidate_file,
    local_server,
    pending_files,
    run_candidate,
)
from evals.manifest import Manifest
from evals.store import Layout, write_rows
from ytverbatim import paths, pipeline, stages
from ytverbatim.acquire.metadata import trim_info
from ytverbatim.asr.runner import worker_lock
from ytverbatim.contract.models import Flag
from ytverbatim.errors import (
    ConfigError,
    EngineError,
    EngineMemoryError,
    LLMOutputError,
    LLMUnavailableError,
    ProcMemoryExceeded,
)
from ytverbatim.llm import ClaudeCLIClient
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Readiness, Usage
from ytverbatim.text import correct as correct_module
from ytverbatim.text import punctuate as punctuate_module
from ytverbatim.text.correct import CorrectionContext, CorrectionResult

GB = 1_000_000_000
HYPOTHESIS = "w-prod"
ZH_TEXTS = ("我们看这个公司的适应率", "然后卖出了一百股", "看NVDA的财报", "大家好")
EN_TEXTS = ("we bought the stock", "then we sold it", "the cat sat down", "good night")
LOCAL = CANDIDATES["l-qwen35-9b"]
SONNET = CANDIDATES["c-sonnet"]


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    monkeypatch.setenv("YTVERBATIM_HOME", str(home / "share"))


def segments(request: LLMRequest) -> list[dict[str, Any]]:
    """The numbered items of a prompt; a retry prompt appends the rejected reply after them."""
    items, _ = json.JSONDecoder().raw_decode(request.user.split("SEGMENTS:\n", 1)[1])
    return items


@dataclass
class FakeLLM:
    """Punctuates by closing every segment, and fixes 适应率 as the proofreader."""

    name: str = "fake:llm"
    invalid_first: int = 0
    always_invalid: bool = False
    requests: list[LLMRequest] = field(default_factory=list[LLMRequest])

    @property
    def id(self) -> str:
        return self.name

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=self._reply(request), usage=Usage(100, 20), elapsed_s=0.25)

    def _reply(self, request: LLMRequest) -> str:
        invalid = self.always_invalid or len(self.requests) <= self.invalid_first
        if invalid and "proofreader" in request.system:
            return "not json"
        items = segments(request)
        if "proofreader" in request.system:
            fixed = [
                {"i": i["i"], "text": i["text"].replace("适应率", "市盈率")} for i in items if "适应率" in i["text"]
            ]
            return json.dumps({"fixed": fixed}, ensure_ascii=False)
        stop = "。" if any("一" <= ch <= "鿿" for item in items for ch in item["text"]) else "."
        return json.dumps({"items": [{"i": i["i"], "text": i["text"] + stop} for i in items]}, ensure_ascii=False)


def with_hypothesis(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Layout, Manifest]:
    layout, manifest = make_eval(root, monkeypatch)
    write_rows(
        layout,
        HYPOTHESIS,
        [*whole_file_rows(HYPOTHESIS, ZH, ZH_TEXTS), *whole_file_rows(HYPOTHESIS, EN, EN_TEXTS)],
    )
    return layout, manifest


def based(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Layout, Manifest]:
    layout, manifest = with_hypothesis(tmp_path / "eval", monkeypatch)
    build_base(layout, manifest, HYPOTHESIS, [], FakeLLM("fake:punct"))
    return layout, manifest


# --- base ---


def test_base_writes_punctuated_stretches_and_a_rerun_makes_no_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = with_hypothesis(tmp_path / "eval", monkeypatch)
    llm = FakeLLM("fake:punct")
    first = build_base(layout, manifest, HYPOTHESIS, [], llm)
    assert [b.file for b in first.built] == [ZH, EN]
    zh = load_base(layout, ZH)
    assert zh.language == "zh"
    assert zh.hypothesis == HYPOTHESIS
    assert zh.punctuation_llm == "fake:punct"
    assert zh.stretches
    assert all(s.text.endswith("。") for s in zh.stretches)
    assert "".join(s.text for s in zh.stretches).replace("。", "") == "".join(ZH_TEXTS)
    assert zh.usage[0].calls >= 1
    assert zh.versions == correction.base_versions()
    assert all(s.edits == () for s in zh.stretches)
    calls = len(llm.requests)
    assert calls > 0

    again = build_base(layout, manifest, HYPOTHESIS, [], llm)
    assert again.built == ()
    assert again.reused == (ZH, EN)
    assert len(llm.requests) == calls


def test_base_refuses_another_punctuation_llm_or_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = based(tmp_path, monkeypatch)
    with pytest.raises(ConfigError, match=r"punctuation LLM.*move .*base aside"):
        build_base(layout, manifest, HYPOTHESIS, [], FakeLLM("fake:other"))
    monkeypatch.setattr(punctuate_module, "PROMPT_VERSION", "punctuation-v99")
    with pytest.raises(ConfigError, match=r"stage versions.*move .*base aside"):
        build_base(layout, manifest, HYPOTHESIS, [], FakeLLM("fake:punct"))


def test_base_refuses_a_changed_input_hypothesis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = based(tmp_path, monkeypatch)
    write_rows(layout, HYPOTHESIS, whole_file_rows(HYPOTHESIS, ZH, ("别的内容", "完全不同", "的文本", "了")))
    with pytest.raises(ConfigError, match="input text"):
        build_base(layout, manifest, HYPOTHESIS, [ZH], FakeLLM("fake:punct"))


def test_base_needs_a_complete_hypothesis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    write_rows(layout, HYPOTHESIS, whole_file_rows(HYPOTHESIS, ZH, ZH_TEXTS))
    with pytest.raises(ConfigError, match=r"run_asr asr --hyp w-prod"):
        build_base(layout, manifest, HYPOTHESIS, [EN], FakeLLM())


# --- correct ---


def test_correct_writes_the_whole_result_and_a_resume_makes_no_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = based(tmp_path, monkeypatch)
    llm = FakeLLM("fake:inner")
    outcome = run_candidate(layout, manifest, SONNET, [], llm)
    assert [f.file for f in outcome.corrected] == [ZH, EN]
    zh = outcome.corrected[0]
    base = load_base(layout, ZH)
    assert zh.candidate == "c-sonnet"
    assert zh.client == "claude-cli:claude-sonnet-5-5"
    assert zh.base_digest == base.text_digest
    assert zh.versions == correction.candidate_versions()
    assert len(zh.texts) == len(base.stretches)
    assert any("市盈率" in t for t in zh.texts)
    assert not any("适应率" in t for t in zh.texts)
    edit = next(e for edits in zh.edits for e in edits)
    assert (edit.before.count("适应率"), edit.after.count("市盈率"), edit.llm) == (1, 1, "claude-cli:claude-sonnet-5-5")
    assert zh.flags == ((),) * len(base.stretches)
    assert zh.rejections == ()
    assert zh.batches == 1
    assert zh.usage[0].client == "claude-cli:claude-sonnet-5-5"
    assert zh.calls == 1
    assert load_candidate_file(layout, SONNET, ZH) == zh

    lines = [json.loads(line) for line in (correction_run(layout) / "calls.jsonl").read_text().splitlines()]
    assert [(line["file"], line["input_tokens"], line["output_tokens"]) for line in lines] == [
        (ZH, 100, 20),
        (EN, 100, 20),
    ]
    assert lines[0]["elapsed_s"] == 0.25

    before = len(llm.requests)
    assert pending_files(layout, manifest, SONNET, []) == []
    again = run_candidate(layout, manifest, SONNET, [], llm)
    assert again.corrected == ()
    assert again.reused == (ZH, EN)
    assert len(llm.requests) == before


def correction_run(layout: Layout, candidate: correction.Candidate = SONNET) -> Path:
    return correction.run_dir(layout, candidate)


def test_a_rejected_proposal_is_stored_with_its_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = based(tmp_path, monkeypatch)

    class Numbers(FakeLLM):
        def _reply(self, request: LLMRequest) -> str:
            items = segments(request)
            fixed = [{"i": i["i"], "text": i["text"].replace("一百", "二百")} for i in items if "一百" in i["text"]]
            return json.dumps({"fixed": fixed}, ensure_ascii=False)

    done = run_candidate(layout, manifest, SONNET, [ZH], Numbers()).corrected[0]
    assert [r.reason for r in done.rejections] == ["cjk_numerals"]
    assert "一百" in "".join(done.texts)


def test_first_try_validity_counts_only_the_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = based(tmp_path, monkeypatch)
    outcome = run_candidate(layout, manifest, SONNET, [], FakeLLM(invalid_first=1))
    assert [(f.batches, f.calls) for f in outcome.corrected] == [(1, 2), (1, 1)]
    assert first_try_validity(outcome.corrected) == pytest.approx(0.5)
    assert first_try_validity(()) is None


def test_a_double_json_failure_is_recorded_and_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = based(tmp_path, monkeypatch)
    with pytest.raises(LLMOutputError):
        run_candidate(layout, manifest, SONNET, [], FakeLLM(always_invalid=True))
    failure = correction.load_failure(layout, SONNET)
    assert failure is not None
    assert failure.file == ZH
    assert "invalid JSON twice" in failure.error
    assert pending_files(layout, manifest, SONNET, []) == [ZH, EN]


class Thinking(FakeLLM):
    def _reply(self, request: LLMRequest) -> str:
        return '<think>\n{"fixed": []}\n</think>\n' + super()._reply(request)


def test_a_think_block_from_a_local_model_disqualifies_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = based(tmp_path, monkeypatch)
    with pytest.raises(LLMOutputError, match="<think>"):
        run_candidate(layout, manifest, LOCAL, [], Thinking())
    failure = correction.load_failure(layout, LOCAL)
    assert failure is not None
    assert "<think>" in failure.error
    (record,) = correction.load_calls(layout, LOCAL)
    assert record.error is not None
    assert record.output_tokens == 20


def test_a_claude_candidate_is_not_screened_for_think_tags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = based(tmp_path, monkeypatch)
    run_candidate(layout, manifest, SONNET, [ZH], Thinking())
    assert correction.load_failure(layout, SONNET) is None


def test_a_call_that_raised_is_logged_with_its_error(tmp_path: Path) -> None:
    class Down(FakeLLM):
        def complete(self, request: LLMRequest) -> LLMResponse:
            raise LLMUnavailableError("quota resets at 5pm")

    calls = tmp_path / "calls.jsonl"
    client = RecordingClient(Down("inner"), client_id="outer", calls=calls, file="f")
    with pytest.raises(LLMUnavailableError):
        client.complete(LLMRequest(system="s", user="SEGMENTS:\n[]"))
    (line,) = calls.read_text().splitlines()
    record = correction.CallRecord.model_validate_json(line)
    assert record.error == "LLMUnavailableError: quota resets at 5pm"
    assert record.input_tokens is None


def test_correct_refuses_another_base_version_or_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = based(tmp_path, monkeypatch)
    run_candidate(layout, manifest, SONNET, [ZH], FakeLLM())

    with monkeypatch.context() as patched:
        patched.setattr(correct_module, "PROMPT_VERSION", "correction-v99")
        with pytest.raises(ConfigError, match=r"stage versions.*runs/c-sonnet aside"):
            pending_files(layout, manifest, SONNET, [ZH])

    path = correction.base_path(layout, ZH)
    base = BaseFile.model_validate_json(path.read_text())
    edited = base.model_copy(update={"stretches": (base.stretches[0].model_copy(update={"text": "改过的。"}),)})
    path.write_text(edited.model_dump_json())
    with pytest.raises(ConfigError, match=r"base text.*runs/c-sonnet aside"):
        pending_files(layout, manifest, SONNET, [ZH])

    stored = correction.candidate_path(layout, SONNET, ZH)
    stored.write_text(json.dumps({"file": ZH}))
    with pytest.raises(ConfigError, match=r"CandidateFile format.*aside"):
        pending_files(layout, manifest, SONNET, [ZH])


def test_correct_needs_a_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = make_eval(tmp_path / "eval", monkeypatch)
    with pytest.raises(ConfigError, match=r"run_correction base --hyp"):
        run_candidate(layout, manifest, SONNET, [ZH], FakeLLM())


def test_a_claude_candidate_id_is_the_clients_own() -> None:
    for candidate in (CANDIDATES["c-sonnet"], CANDIDATES["c-haiku"]):
        assert candidate.client_id == ClaudeCLIClient(model=candidate.model).id


def test_a_local_candidate_is_keyed_by_its_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = based(tmp_path, monkeypatch)
    assert LOCAL.client_id == "mlx:mlx-community/Qwen3.5-9B-8bit@16daa4818c54"
    llm = FakeLLM("openai-compat:default_model@127.0.0.1:50001")
    done = run_candidate(layout, manifest, LOCAL, [ZH], llm).corrected[0]
    assert done.client == LOCAL.client_id
    assert next(e for edits in done.edits for e in edits).llm == LOCAL.client_id
    assert done.usage[0].client == LOCAL.client_id


def test_recording_client_reports_the_given_id_and_appends_each_reply(tmp_path: Path) -> None:
    calls = tmp_path / "run" / "calls.jsonl"
    client = RecordingClient(FakeLLM("inner"), client_id="outer", calls=calls, file="f")
    assert client.id == "outer"
    client.complete(LLMRequest(system="s", user="SEGMENTS:\n[]"))
    client.complete(LLMRequest(system="s", user="SEGMENTS:\n[]"))
    assert len(calls.read_text().splitlines()) == 2


def test_parity_the_eval_passes_the_pipelines_context_and_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = with_hypothesis(tmp_path / "eval", monkeypatch)
    meta = layout.root / "meta" / f"{ZH}.info.json"
    raw = json.loads(meta.read_text())
    raw["tags"] = [f"term{i}" for i in range(25)] + [3]
    meta.write_text(json.dumps(raw))
    rows = [
        row.model_copy(update={"words": tuple(w.model_copy(update={"conf": 0.3}) for w in row.words)})
        for row in whole_file_rows(HYPOTHESIS, ZH, ZH_TEXTS)
    ]
    write_rows(layout, HYPOTHESIS, rows)
    build_base(layout, manifest, HYPOTHESIS, [ZH], FakeLLM("fake:punct"))
    base = load_base(layout, ZH)

    seen: list[dict[str, object]] = []

    def fake(segments: list[str], **kwargs: object) -> CorrectionResult:
        seen.append({"segments": segments, **kwargs})
        return CorrectionResult(
            texts=tuple(segments),
            edits=((),) * len(segments),
            flags=((),) * len(segments),
            usage=(),
            rejections=(),
            batches=0,
        )

    monkeypatch.setattr(correction, "correct_segments", fake)
    run_candidate(layout, manifest, SONNET, [ZH], FakeLLM())

    (call,) = seen
    expected_context = pipeline.correction_context(trim_info(raw), pipeline.info_tags(raw))
    assert call["context"] == expected_context
    assert isinstance(call["context"], CorrectionContext)
    assert len(call["context"].glossary) == 20
    assert call["confidence"] == stages.word_confidence(base.stretches)
    assert call["confidence"] is not None
    assert call["segments"] == [s.text for s in base.stretches]
    assert call["language"] == base.language
    assert call["checkpoint_dir"] == correction_run(layout) / "ckpt" / ZH
    assert call["client"].id == SONNET.client_id  # type: ignore[attr-defined]


def test_entity_flags_are_stored_with_the_candidate_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = based(tmp_path, monkeypatch)

    class Tickers(FakeLLM):
        def _reply(self, request: LLMRequest) -> str:
            items = segments(request)
            fixed = [{"i": i["i"], "text": i["text"].replace("NVDA", "NVDIA")} for i in items if "NVDA" in i["text"]]
            return json.dumps({"fixed": fixed}, ensure_ascii=False)

    done = run_candidate(layout, manifest, SONNET, [ZH], Tickers()).corrected[0]
    flags = [f for fs in done.flags for f in fs]
    assert flags
    assert all(isinstance(f, Flag) and f.kind == "entity_corrected" for f in flags)
    assert CandidateFile.model_validate_json(correction.candidate_path(layout, SONNET, ZH).read_text()) == done


# --- local server ---


class FakeServer:
    pid = 1

    def __init__(self, *, alive: bool = True) -> None:
        self._alive = alive
        self.peak_footprint_bytes: int | None = 5 * GB
        self.returncode: int | None = None if alive else 3

    def alive(self) -> bool:
        return self._alive


@dataclass
class Rig:
    cache: Path
    directory: Path
    started: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    downloads: list[tuple[str, str, Path]] = field(default_factory=list[tuple[str, str, Path]])
    locked_during: list[bool] = field(default_factory=list[bool])
    answers: list[bool] = field(default_factory=lambda: [False, True])
    server: FakeServer = field(default_factory=FakeServer)
    breach: ProcMemoryExceeded | None = None
    free: int = 100 * GB

    def seams(self) -> ServerSeams:
        return ServerSeams(
            free=lambda _: self.free,
            download=self._download,
            runtime=lambda: None,
            background=self._background,
            ready=self._ready,
            port=lambda: 50123,
            poll_s=0.0,
            ready_timeout_s=5.0,
        )

    def _download(self, repo: str, revision: str, cache: Path) -> Path:
        self.downloads.append((repo, revision, cache))
        return cache / "downloaded"

    def _ready(self, base_url: str) -> Readiness:
        assert base_url == "http://127.0.0.1:50123/v1"
        return Readiness(self.answers.pop(0) if self.answers else True, "fake")

    @contextlib.contextmanager
    def _background(self, argv: list[str], **kwargs: object) -> Generator[FakeServer]:
        self.started.append({"argv": argv, **kwargs})
        self.locked_during.append(_lock_is_held())
        try:
            yield self.server
        except BaseException as exc:
            if self.breach is not None:
                raise self.breach from exc
            raise
        if self.breach is not None:
            raise self.breach


def _lock_is_held() -> bool:
    fd = os.open(paths.locks_dir() / "mlx-worker.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(cache=tmp_path / "cache", directory=tmp_path / "run")


def breach(trigger: str = "footprint") -> ProcMemoryExceeded:
    return ProcMemoryExceeded(
        "python",
        trigger,
        footprint_bytes=13 << 30,
        cap_bytes=12 << 30,
        available_percent=40,
        floor_percent=10,  # type: ignore[arg-type]
    )


def test_the_local_server_runs_pinned_bounded_and_under_the_worker_lock(rig: Rig) -> None:
    with local_server(LOCAL, rig.cache, rig.directory, memory_cap=12 << 30, seams=rig.seams()) as client:
        assert client.body(LLMRequest(system="s", user="u"))["model"] == "default_model"
        assert client.id == "openai-compat:default_model@127.0.0.1:50123"
        assert _lock_is_held()
    assert not _lock_is_held()
    assert rig.downloads == [(LOCAL.model, LOCAL.revision, rig.cache)]
    (started,) = rig.started
    argv = started["argv"]
    assert isinstance(argv, list)
    assert argv[1:4] == ["-m", "evals.mlx_server", "--memory-cap-bytes"]
    assert argv[4] == str(12 << 30)
    assert argv[5:9] == ["--", "--model", str(rig.cache / "downloaded"), "--host"]
    assert argv[argv.index("--port") + 1] == "50123"
    assert argv[argv.index("--chat-template-args") + 1] == '{"enable_thinking":false}'
    assert rig.locked_during == [True]
    assert started["cwd"] == correction.REPO_ROOT
    assert started["log"] == rig.directory / "server.log"
    assert started["memory_watch"].cap_bytes == 12 << 30  # type: ignore[attr-defined]
    env = started["env"]
    assert isinstance(env, dict)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_HUB_CACHE"] == str(rig.cache)
    assert set(env) <= {"PATH", "HOME", "TMPDIR", "LANG", "HF_HUB_OFFLINE", "HF_HUB_CACHE"} | {
        k for k in env if k.startswith("LC_")
    }
    record = json.loads((rig.directory / "server.json").read_text())
    assert record["peak_footprint_bytes"] == 5 * GB
    assert record["cap_bytes"] == 12 << 30
    assert record["candidate"] == "l-qwen35-9b"


def test_a_present_snapshot_is_used_without_a_download(rig: Rig) -> None:
    assert LOCAL.revision is not None
    snapshot = disk.repo_dir(rig.cache, LOCAL.model) / "snapshots" / LOCAL.revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    rig.free = 0
    with local_server(LOCAL, rig.cache, rig.directory, seams=rig.seams()):
        pass
    assert rig.downloads == []
    assert rig.started[0]["argv"][rig.started[0]["argv"].index("--model") + 1] == str(snapshot)  # type: ignore[index]


def test_the_disk_gate_refuses_before_any_download_or_server(rig: Rig) -> None:
    rig.free = 5 * GB
    with (
        pytest.raises(ConfigError, match=r"needs 10\.45 GB.*free space or clean up"),
        local_server(LOCAL, rig.cache, rig.directory, seams=rig.seams()),
    ):
        pytest.fail("the server must not start")
    assert rig.downloads == []
    assert rig.started == []
    assert not _lock_is_held()


def test_the_default_hugging_face_cache_is_refused(rig: Rig) -> None:
    default = Path.home() / ".cache" / "huggingface" / "hub"
    with (
        pytest.raises(ConfigError, match="default Hugging Face cache"),
        local_server(LOCAL, default, rig.directory, seams=rig.seams()),
    ):
        pytest.fail("the server must not start")
    assert rig.downloads == []


def test_a_claude_candidate_has_no_server(rig: Rig) -> None:
    with (
        pytest.raises(ConfigError, match="no local weights"),
        local_server(SONNET, rig.cache, rig.directory, seams=rig.seams()),
    ):
        pytest.fail("the server must not start")


@pytest.mark.parametrize(
    ("trigger", "fix"), [("footprint", "memory_cap_gib"), ("available", "close memory-heavy apps")]
)
def test_a_watch_breach_becomes_an_engine_memory_error(rig: Rig, trigger: str, fix: str) -> None:
    rig.breach = breach(trigger)
    with pytest.raises(EngineMemoryError) as info, local_server(LOCAL, rig.cache, rig.directory, seams=rig.seams()):
        raise LLMUnavailableError("cannot reach the server")
    assert info.value.trigger == trigger
    assert fix in str(info.value)
    assert "finished batches are kept" in str(info.value)
    assert isinstance(info.value.__cause__, ProcMemoryExceeded)
    assert (rig.directory / "server.json").exists()
    assert not _lock_is_held()


def test_a_server_that_exits_before_it_is_ready_names_its_log(rig: Rig) -> None:
    rig.server = FakeServer(alive=False)
    rig.directory.mkdir()
    (rig.directory / "server.log").write_text("loading\nValueError: unsupported model\n")
    with (
        pytest.raises(EngineError, match=r"(?s)exited with code 3.*unsupported model"),
        local_server(LOCAL, rig.cache, rig.directory, seams=rig.seams()),
    ):
        pytest.fail("the server never became ready")


def test_a_server_that_never_answers_times_out(rig: Rig) -> None:
    rig.answers = [False] * 1000
    seams = replace(rig.seams(), ready_timeout_s=0.0)
    with (
        pytest.raises(EngineError, match=r"did not answer within 0s.*l-qwen3-8b"),
        local_server(LOCAL, rig.cache, rig.directory, seams=seams),
    ):
        pytest.fail("the server never became ready")


def test_the_worker_lock_is_not_shared_with_a_second_holder(rig: Rig) -> None:
    with local_server(LOCAL, rig.cache, rig.directory, seams=rig.seams()):
        fd = os.open(paths.locks_dir() / "mlx-worker.lock", os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    with worker_lock():
        pass


# --- cleanup ---


def recorder() -> tuple[list[str], Callable[[Path, str], int]]:
    deleted: list[str] = []

    def delete(cache: Path, repo: str) -> int:
        deleted.append(repo)
        return 10 * GB

    return deleted, delete


def test_cleanup_refuses_an_incomplete_run_and_names_the_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = based(tmp_path, monkeypatch)
    run_candidate(layout, manifest, LOCAL, [ZH], FakeLLM())
    deleted, delete = recorder()
    with pytest.raises(ConfigError, match=rf"no output for 1 file.*{EN}.*correct --candidate l-qwen35-9b"):
        cleanup_candidate(layout, manifest, LOCAL, tmp_path / "cache", delete=delete)
    assert deleted == []


def test_cleanup_deletes_the_weights_of_a_finished_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, manifest = based(tmp_path, monkeypatch)
    run_candidate(layout, manifest, LOCAL, [], FakeLLM())
    deleted, delete = recorder()
    assert cleanup_candidate(layout, manifest, LOCAL, tmp_path / "cache", delete=delete) == 10 * GB
    assert deleted == ["mlx-community/Qwen3.5-9B-8bit"]


def test_cleanup_never_touches_the_default_cache_or_a_claude_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, manifest = based(tmp_path, monkeypatch)
    run_candidate(layout, manifest, LOCAL, [], FakeLLM())
    deleted, delete = recorder()
    default = Path.home() / ".cache" / "huggingface" / "hub"
    with pytest.raises(ConfigError, match="default Hugging Face cache"):
        cleanup_candidate(layout, manifest, LOCAL, default, delete=delete)
    with pytest.raises(ConfigError, match="no local weights"):
        cleanup_candidate(layout, manifest, SONNET, tmp_path / "cache", delete=delete)
    assert deleted == []


# --- commands ---


def test_the_commands_run_base_correct_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout, _ = with_hypothesis(tmp_path / "eval", monkeypatch)
    punct = FakeLLM("fake:punct")
    corrector = FakeLLM("fake:corrector")
    opened: list[str] = []
    monkeypatch.setattr(run_correction, "punctuation_client", lambda spec: punct)

    @contextlib.contextmanager
    def open_client(store: Layout, candidate: correction.Candidate, cache: Path) -> Generator[FakeLLM]:
        opened.append(candidate.id)
        yield corrector

    monkeypatch.setattr(run_correction, "open_client", open_client)
    runner = CliRunner()

    result = runner.invoke(run_correction.main, ["base", "--hyp", HYPOTHESIS])
    assert result.exit_code == 0, result.output
    assert "2 files built" in result.output
    assert load_base(layout, ZH).punctuation_llm == "fake:punct"
    again = runner.invoke(run_correction.main, ["base", "--hyp", HYPOTHESIS])
    assert "0 files built (0 punctuation calls), 2 already done" in again.output

    result = runner.invoke(run_correction.main, ["correct", "--candidate", "c-haiku"])
    assert result.exit_code == 0, result.output
    assert "2 files corrected, 2 calls over 2 batches, first-try valid JSON 100.0%, 1 edits applied" in result.output
    calls = len(corrector.requests)
    again = runner.invoke(run_correction.main, ["correct", "--candidate", "c-haiku"])
    assert "complete for all selected files" in again.output
    assert opened == ["c-haiku"]
    assert len(corrector.requests) == calls

    refused = runner.invoke(run_correction.main, ["cleanup", "--candidate", "l-qwen3-8b"])
    assert refused.exit_code != 0
    assert isinstance(refused.exception, ConfigError)
    assert "correct --candidate l-qwen3-8b" in str(refused.exception)


def test_cleanup_only_offers_local_candidates() -> None:
    result = CliRunner().invoke(run_correction.main, ["cleanup", "--candidate", "c-sonnet"])
    assert result.exit_code == 2
    assert "l-qwen35-9b" in result.output
