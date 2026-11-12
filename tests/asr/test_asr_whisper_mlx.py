import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from fake_engine import write_silence

from ytverbatim.asr import registry, worker
from ytverbatim.asr.engines import whisper_mlx
from ytverbatim.asr.engines.whisper_mlx import WhisperEngine
from ytverbatim.asr.models import HYPOTHESES_FILE, ChunkHypothesis, FileJob, WorkerRequest
from ytverbatim.errors import ConfigError, DependencyError

REPO_DIR = "models--mlx-community--whisper-large-v3-mlx"
RESULT: dict[str, Any] = {
    "text": " 示例，你好 ",
    "language": "zh",
    "segments": [
        {
            "words": [
                {"word": "示例，", "start": 0.5, "end": 1.0, "probability": 0.75},
                {"word": "你好", "start": 1.0, "end": 1.5, "probability": 0.25},
                {"word": "漏", "start": None, "end": None, "probability": 0.9},
            ]
        },
        {"words": [{"word": "x", "start": 2.0, "end": 2.5}]},
    ],
}


class FakeMx:
    def __init__(self) -> None:
        self.events: list[tuple[str, int | None]] = []
        self.clear_error: list[Exception | None] = []
        self.float16 = object()

    def set_cache_limit(self, limit: int) -> int:
        self.events.append(("limit", limit))
        return 123

    def clear_cache(self) -> None:
        self.events.append(("clear", None))
        if self.clear_error and (error := self.clear_error.pop(0)) is not None:
            raise error


class FakeWhisper:
    def __init__(self, mx: FakeMx) -> None:
        self.mx = mx
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.result: dict[str, Any] = RESULT
        self.error: Exception | None = None
        self.loads: list[tuple[str, object]] = []

    def transcribe(self, audio: Any, **kwargs: Any) -> dict[str, Any]:
        self.mx.events.append(("transcribe", None))
        self.calls.append((audio, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def mx() -> FakeMx:
    return FakeMx()


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "default-hub"
    path = cache / REPO_DIR / "snapshots" / registry.get("w-prod").spec.revision
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "weights.npz").write_bytes(b"")
    constants = ModuleType("huggingface_hub.constants")
    constants.default_cache_path = str(cache)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", ModuleType("huggingface_hub"))
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "candidates-hub"))
    return path


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch, mx: FakeMx, snapshot: Path) -> FakeWhisper:
    whisper = FakeWhisper(mx)
    package, transcribe_module, tokenizer = (
        ModuleType("mlx_whisper"),
        ModuleType("mlx_whisper.transcribe"),
        ModuleType("mlx_whisper.tokenizer"),
    )
    package.transcribe = whisper.transcribe  # type: ignore[attr-defined]

    class ModelHolder:
        @classmethod
        def get_model(cls, model_path: str, dtype: object) -> object:
            whisper.loads.append((model_path, dtype))
            return object()

    transcribe_module.ModelHolder = ModelHolder  # type: ignore[attr-defined]
    tokenizer.LANGUAGES = {"en": "english", "zh": "chinese", "yue": "cantonese"}  # type: ignore[attr-defined]
    tokenizer.TO_LANGUAGE_CODE = {"chinese": "zh", "mandarin": "zh"}  # type: ignore[attr-defined]
    core = ModuleType("mlx.core")
    core.set_cache_limit = mx.set_cache_limit  # type: ignore[attr-defined]
    core.clear_cache = mx.clear_cache  # type: ignore[attr-defined]
    core.float16 = mx.float16  # type: ignore[attr-defined]
    core.get_peak_memory = lambda: 42  # type: ignore[attr-defined]
    mlx = ModuleType("mlx")
    mlx.core = core  # type: ignore[attr-defined]
    for name, module in (
        ("mlx", mlx),
        ("mlx.core", core),
        ("mlx_whisper", package),
        ("mlx_whisper.transcribe", transcribe_module),
        ("mlx_whisper.tokenizer", tokenizer),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return whisper


def loaded(hypothesis: str, fake: FakeWhisper) -> WhisperEngine:
    engine = registry.build_engine(registry.get(hypothesis))
    assert isinstance(engine, WhisperEngine)
    engine.load()
    return engine


AUDIO = np.zeros(16_000, dtype=np.float32)


@pytest.mark.parametrize(("hypothesis", "mode"), [("w-prod", "whole_file"), ("w-vad", "chunk")])
def test_both_modes_make_the_production_call(hypothesis: str, mode: str, fake: FakeWhisper, snapshot: Path) -> None:
    engine = loaded(hypothesis, fake)
    assert engine.mode == mode
    engine.transcribe(AUDIO, language="zh", context=None)
    audio, kwargs = fake.calls[0]
    assert audio is AUDIO
    assert kwargs == {
        "path_or_hf_repo": str(snapshot),
        "word_timestamps": True,
        "language": "zh",
        "initial_prompt": whisper_mlx.ZH_INITIAL_PROMPT,
        "condition_on_previous_text": False,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    }


def test_no_language_means_no_prompt_and_auto_detection(fake: FakeWhisper) -> None:
    loaded("w-prod", fake).transcribe(AUDIO, language=None, context=None)
    assert fake.calls[0][1]["language"] is None
    assert fake.calls[0][1]["initial_prompt"] is None


def test_english_gets_no_prompt(fake: FakeWhisper) -> None:
    loaded("w-prod", fake).transcribe(AUDIO, language="en-US", context=None)
    assert fake.calls[0][1]["language"] == "en"
    assert fake.calls[0][1]["initial_prompt"] is None


def test_the_temperature_schedule_is_the_increasing_fallback() -> None:
    schedule = whisper_mlx.TEMPERATURE_SCHEDULE
    assert len(schedule) > 1
    assert list(schedule) == sorted(schedule)
    assert schedule[0] == 0.0


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("zh-Hans", "zh"), ("ZH_cn", "zh"), ("en", "en"), ("auto", None), (" ", None), ("-x", None), ("mandarin", None)],
)
def test_language_tags_reduce_to_the_primary_code(tag: str, expected: str | None, fake: FakeWhisper) -> None:
    assert whisper_mlx._normalize_language(tag) == expected


def test_an_unknown_language_warns_and_auto_detects(fake: FakeWhisper, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert whisper_mlx._normalize_language("xx-YY") is None
    assert "xx-YY" in caplog.text


def test_words_and_confidence_come_from_the_word_probabilities(fake: FakeWhisper) -> None:
    raw = loaded("w-vad", fake).transcribe(AUDIO, language="zh", context=None)
    assert raw.text == "示例，你好"
    assert raw.language == "zh"
    assert [(w.text, w.start, w.end, w.conf) for w in raw.words] == [
        ("示例，", 0.5, 1.0, 0.75),
        ("你好", 1.0, 1.5, 0.25),
        ("x", 2.0, 2.5, None),
    ]
    assert raw.confidence == 0.5
    assert raw.generated_tokens is None
    assert raw.max_tokens is None


def test_a_decode_without_words_has_no_confidence(fake: FakeWhisper) -> None:
    fake.result = {"text": "", "language": "en", "segments": []}
    raw = loaded("w-prod", fake).transcribe(AUDIO, language="en", context=None)
    assert (raw.text, raw.words, raw.confidence) == ("", (), None)


def test_load_resolves_the_default_cache_snapshot_not_the_worker_cache(
    fake: FakeWhisper, snapshot: Path, mx: FakeMx
) -> None:
    loaded("w-prod", fake)
    assert fake.loads == [(str(snapshot), mx.float16)]
    assert "candidates-hub" not in str(snapshot)


def test_a_missing_snapshot_fails_loud_with_the_download_command(fake: FakeWhisper, snapshot: Path) -> None:
    (snapshot / "config.json").unlink()
    engine = registry.build_engine(registry.get("w-prod"))
    with pytest.raises(
        DependencyError, match=r"hf download mlx-community/whisper-large-v3-mlx --revision [0-9a-f]{40}"
    ):
        engine.load()
    assert fake.loads == []


def test_a_snapshot_without_weights_is_missing(fake: FakeWhisper, snapshot: Path) -> None:
    (snapshot / "weights.npz").unlink()
    with pytest.raises(DependencyError, match="missing"):
        registry.build_engine(registry.get("w-prod")).load()


def test_a_missing_whisper_extra_names_the_install(fake: FakeWhisper, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    with pytest.raises(DependencyError, match=r"install ytverbatim\[whisper\]"):
        registry.build_engine(registry.get("w-prod")).load()


def test_transcribing_before_loading_is_a_programming_error(fake: FakeWhisper) -> None:
    engine = registry.build_engine(registry.get("w-prod"))
    with pytest.raises(RuntimeError, match="load"):
        engine.transcribe(AUDIO, language=None, context=None)


def test_decode_runs_inside_a_cache_limit_of_zero_and_restores_it(fake: FakeWhisper, mx: FakeMx) -> None:
    loaded("w-prod", fake).transcribe(AUDIO, language=None, context=None)
    assert mx.events == [("limit", 0), ("clear", None), ("transcribe", None), ("clear", None), ("limit", 123)]


def test_a_decode_error_restores_the_cache_limit_and_propagates(fake: FakeWhisper, mx: FakeMx) -> None:
    error = ValueError("decode failed")
    fake.error = error
    engine = loaded("w-prod", fake)
    with pytest.raises(ValueError, match="decode failed") as raised:
        engine.transcribe(AUDIO, language=None, context=None)
    assert raised.value is error
    assert mx.events[-1] == ("limit", 123)


def test_a_cleanup_failure_does_not_mask_the_decode_error(
    fake: FakeWhisper, mx: FakeMx, caplog: pytest.LogCaptureFixture
) -> None:
    error = ValueError("decode failed")
    fake.error = error
    engine = loaded("w-prod", fake)
    mx.clear_error = [None, RuntimeError("cleanup failed")]
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="decode failed") as raised:
        engine.transcribe(AUDIO, language=None, context=None)
    assert raised.value is error
    assert mx.events[-1] == ("limit", 123)
    assert "cleanup failed" in caplog.text


def test_a_cleanup_failure_after_success_is_raised(fake: FakeWhisper, mx: FakeMx) -> None:
    engine = loaded("w-prod", fake)
    mx.clear_error = [None, RuntimeError("cleanup failed")]
    with pytest.raises(RuntimeError, match="cleanup failed"):
        engine.transcribe(AUDIO, language=None, context=None)
    assert mx.events[-1] == ("limit", 123)


def test_mode_must_be_known_and_agree_with_the_spec() -> None:
    whole, chunked = registry.get("w-prod"), registry.get("w-vad")
    with pytest.raises(ConfigError, match="mode"):
        whisper_mlx.create(whole.spec, {"mode": "other"})
    with pytest.raises(ConfigError, match="contradicts"):
        whisper_mlx.create(whole.spec, {"mode": "chunk"})
    with pytest.raises(ConfigError, match="contradicts"):
        whisper_mlx.create(chunked.spec, {"mode": "whole_file"})


def test_the_registry_pins_the_default_cache_revision() -> None:
    for hypothesis in ("w-prod", "w-vad"):
        spec = registry.get(hypothesis).spec
        assert (spec.repo, spec.revision) == (
            "mlx-community/whisper-large-v3-mlx",
            "49e6aa286ad60c14352c404340ded53710378a11",
        )


def test_the_worker_decodes_a_whole_file_in_one_call_with_absolute_times(tmp_path: Path, fake: FakeWhisper) -> None:
    wav = write_silence(tmp_path / "-dash.wav", 12.0)
    out = tmp_path / "out"
    request = WorkerRequest(
        engine=registry.get("w-prod"), out_dir=str(out), files=(FileJob(id="f1", wav=str(wav), language="zh"),)
    )
    metrics = worker.run_request(request)
    assert metrics.chunks_decoded == 1
    assert len(fake.calls) == 1
    assert len(fake.calls[0][0]) == 12 * 16_000
    (line,) = (out / HYPOTHESES_FILE).read_text(encoding="utf-8").splitlines()
    row = ChunkHypothesis.model_validate_json(line)
    assert (row.start, row.end, row.text, row.language, row.confidence) == (0.0, 12.0, "示例，你好", "zh", 0.5)
    assert [w.text for w in row.words] == ["示例，", "你好", "x"]
