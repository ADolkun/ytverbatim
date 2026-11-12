import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ytverbatim.asr import registry
from ytverbatim.asr.engines import mimo
from ytverbatim.errors import ConfigError, EngineError

DEFINITION = registry.get("m-4bit")
SPEC = DEFINITION.spec
TOKENIZER_REPO = str(DEFINITION.variant["tokenizer_repo"])
TOKENIZER_REVISION = str(DEFINITION.variant["tokenizer_revision"])


class FakeAsrModel:
    """Stands in for mlx-audio's `Model` class and records the tokenizer directory `post_load_hook` would use."""

    @staticmethod
    def _resolve_audio_tokenizer_dir(model_path: Path) -> Path:
        return model_path / "manifest-tokenizer"


class FakeModel:
    def __init__(self, tokens: int = 40, text: str = "你好 NVDA") -> None:
        self.tokens = tokens
        self.text = text
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def generate(self, audio: Any, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((audio, kwargs))
        return SimpleNamespace(text=self.text, language=kwargs["language"], generation_tokens=self.tokens)


class Recorder:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.snapshots: list[tuple[str, dict[str, Any]]] = []
        self.loads: list[tuple[str, dict[str, Any]]] = []
        self.tokenizer_dir_seen: Path | None = None
        self.resampled: list[tuple[int, int]] = []


def install(monkeypatch: pytest.MonkeyPatch, model: FakeModel) -> Recorder:
    rec = Recorder(model)
    stt, utils, asr = ModuleType("mlx_audio.stt"), ModuleType("mlx_audio.utils"), ModuleType(mimo._MODEL_MODULE)
    scipy, signal = ModuleType("scipy"), ModuleType("scipy.signal")
    asr.Model = FakeAsrModel  # type: ignore[attr-defined]

    def get_model_path(repo: str, **kwargs: Any) -> Path:
        rec.snapshots.append((repo, kwargs))
        return Path("/snapshots/tokenizer")

    def load(repo: str, **kwargs: Any) -> FakeModel:
        rec.loads.append((repo, kwargs))
        rec.tokenizer_dir_seen = FakeAsrModel._resolve_audio_tokenizer_dir(Path("/snapshots/model"))
        return model

    def resample_poly(x: Any, up: int, down: int) -> Any:
        rec.resampled.append((up, down))
        return np.repeat(x, 2)[: len(x) * up // down]

    utils.get_model_path = get_model_path  # type: ignore[attr-defined]
    stt.load = load  # type: ignore[attr-defined]
    signal.resample_poly = resample_poly  # type: ignore[attr-defined]
    scipy.signal = signal  # type: ignore[attr-defined]
    for name, module in (
        ("mlx_audio.stt", stt),
        ("mlx_audio.utils", utils),
        (mimo._MODEL_MODULE, asr),
        ("scipy", scipy),
        ("scipy.signal", signal),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return rec


def loaded_engine(monkeypatch: pytest.MonkeyPatch, model: FakeModel) -> tuple[Any, Recorder]:
    rec = install(monkeypatch, model)
    engine = mimo.create(SPEC, DEFINITION.variant)
    engine.load()
    return engine, rec


def test_loads_the_pinned_model_and_tokenizer_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    _, rec = loaded_engine(monkeypatch, FakeModel())

    assert rec.snapshots == [(TOKENIZER_REPO, {"revision": TOKENIZER_REVISION})]
    assert rec.loads == [(SPEC.repo, {"revision": SPEC.revision})]


def test_the_pinned_tokenizer_replaces_the_manifest_lookup_only_while_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    _, rec = loaded_engine(monkeypatch, FakeModel())

    assert rec.tokenizer_dir_seen == Path("/snapshots/tokenizer")
    assert FakeAsrModel._resolve_audio_tokenizer_dir(Path("/m")) == Path("/m/manifest-tokenizer")


def test_the_lookup_is_restored_when_loading_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = install(monkeypatch, FakeModel())
    stt = sys.modules["mlx_audio.stt"]

    def load(repo: str, **kwargs: Any) -> None:
        raise OSError("disk full")

    stt.load = load  # type: ignore[attr-defined]

    with pytest.raises(EngineError, match=rf"{SPEC.repo}@{SPEC.revision}.*{TOKENIZER_REPO}.*disk full"):
        mimo.create(SPEC, DEFINITION.variant).load()

    assert FakeAsrModel._resolve_audio_tokenizer_dir(Path("/m")) == Path("/m/manifest-tokenizer")
    assert rec.loads == []


def test_a_load_that_never_consults_the_pin_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = install(monkeypatch, FakeModel())
    stt = sys.modules["mlx_audio.stt"]

    def load_ignoring_the_hook(repo: str, **kwargs: Any) -> FakeModel:
        rec.loads.append((repo, kwargs))
        return rec.model

    stt.load = load_ignoring_the_hook  # type: ignore[attr-defined]
    engine = mimo.create(SPEC, DEFINITION.variant)
    with pytest.raises(EngineError, match=rf"without consulting the pinned tokenizer {TOKENIZER_REPO}"):
        engine.load()
    with pytest.raises(EngineError, match="load"):
        engine.transcribe(np.zeros(16_000, dtype=np.float32), language=None, context=None)


def test_every_generation_parameter_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    engine, rec = loaded_engine(monkeypatch, model)

    engine.transcribe(np.zeros(16_000 * 10, dtype=np.float32), language="zh", context=None)

    [(audio, kwargs)] = model.calls
    assert kwargs == {
        "max_tokens": 120,
        "temperature": 0.0,
        "top_p": 0.95,
        "top_k": 0,
        "language": "zh",
        "verbose": False,
    }
    assert rec.resampled == [(3, 2)]
    assert len(audio) == 24_000 * 10


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.5, 64), (5.0, 64), (10.0, 120), (30.0, 360), (30.1, 362)],
)
def test_max_tokens_follows_the_chunk_duration(seconds: float, expected: int) -> None:
    assert mimo.token_budget(seconds) == expected


def test_a_chunk_that_reaches_the_cap_reports_both_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _ = loaded_engine(monkeypatch, FakeModel(tokens=120))

    raw = engine.transcribe(np.zeros(16_000 * 10, dtype=np.float32), language=None, context=None)

    assert (raw.generated_tokens, raw.max_tokens) == (120, 120)
    assert raw.text == "你好 NVDA"
    assert raw.confidence is None
    assert raw.words == ()


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("zh", "zh"),
        ("zh-CN", "zh"),
        ("zh_TW", "zh"),
        ("Chinese", "zh"),
        ("mandarin", "zh"),
        ("en", "en"),
        ("en-US", "en"),
        ("English", "en"),
        ("zh+en", None),
        ("ja", None),
        (None, None),
    ],
)
def test_language_map(given: str | None, expected: str | None) -> None:
    assert mimo.model_language(given) == expected


def test_the_resampled_chunk_has_one_and_a_half_times_the_samples() -> None:
    pytest.importorskip("scipy.signal")
    audio = np.sin(np.linspace(0, 200, 16_000 * 7)).astype(np.float32)

    resampled = mimo.resample_to_model_rate(audio)

    assert len(resampled) == 24_000 * 7
    assert resampled.dtype == np.float32


def test_the_registry_spec_matches_the_resampling_constants() -> None:
    assert SPEC.sample_rate == mimo.MODEL_SAMPLE_RATE
    assert mimo.RESAMPLE_UP / mimo.RESAMPLE_DOWN == mimo.MODEL_SAMPLE_RATE / 16_000


def test_transcribe_before_load_is_an_engine_error() -> None:
    with pytest.raises(EngineError, match="load"):
        mimo.create(SPEC, DEFINITION.variant).transcribe(
            np.zeros(16_000, dtype=np.float32), language=None, context=None
        )


@pytest.mark.parametrize(
    "variant", [{}, {"tokenizer_repo": "a/b"}, {"tokenizer_repo": "a/b", "tokenizer_revision": ""}]
)
def test_the_variant_must_pin_the_tokenizer(variant: dict[str, Any]) -> None:
    with pytest.raises(ConfigError, match="tokenizer_revision"):
        mimo.create(SPEC, variant)
