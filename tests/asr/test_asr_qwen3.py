import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ytverbatim.asr import registry
from ytverbatim.asr.engines import qwen3
from ytverbatim.asr.protocol import AsrContext
from ytverbatim.errors import ConfigError, DependencyError

RATE = 16_000


class FakeModel:
    def __init__(self, text: str = "hello", language: str | list[str] = "English", tokens: int = 7) -> None:
        self.reply = SimpleNamespace(text=text, language=language, generation_tokens=tokens)
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def generate(self, audio: Any, **kwargs: Any) -> Any:
        self.calls.append((audio, kwargs))
        return self.reply


def install_fake(monkeypatch: pytest.MonkeyPatch, model: FakeModel) -> list[tuple[str, dict[str, Any]]]:
    loads: list[tuple[str, dict[str, Any]]] = []
    package, stt, utils = ModuleType("mlx_audio"), ModuleType("mlx_audio.stt"), ModuleType("mlx_audio.stt.utils")

    def load(repo: str, **kwargs: Any) -> FakeModel:
        loads.append((repo, kwargs))
        return model

    utils.load = load  # type: ignore[attr-defined]
    for name, module in (("mlx_audio", package), ("mlx_audio.stt", stt), ("mlx_audio.stt.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    return loads


def engine(hypothesis: str) -> qwen3.Qwen3Engine:
    definition = registry.get(hypothesis)
    built = qwen3.create(definition.spec, definition.variant)
    built.load()
    return built


def clip(seconds: float) -> Any:
    return np.zeros(round(seconds * RATE), dtype=np.float32)


CONTEXT = AsrContext(system_prompt="Bitcoin: Q4 2026", hotwords=("BTC", "ETF"))


def test_q_base_pins_every_parameter_and_sends_no_biasing(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    install_fake(monkeypatch, model)

    engine("q-base").transcribe(clip(30), language=None, context=CONTEXT)

    samples, kwargs = model.calls[0]
    assert len(samples) == 30 * RATE
    assert kwargs == {
        "max_tokens": 360,
        "batch_size": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "min_tokens_to_keep": 1,
        "repetition_penalty": None,
        "repetition_context_size": 100,
        "language": None,
        "prefill_step_size": 2048,
        "chunk_duration": 40.0,
        "min_chunk_duration": 1.0,
        "verbose": False,
        "stream": False,
        "system_prompt": None,
        "hotwords": None,
    }


def test_q_ctx_differs_from_q_base_only_in_the_biasing(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    install_fake(monkeypatch, model)

    engine("q-base").transcribe(clip(30), language=None, context=CONTEXT)
    engine("q-ctx").transcribe(clip(30), language=None, context=CONTEXT)

    base, with_context = model.calls[0][1], model.calls[1][1]
    assert with_context["system_prompt"] == "Bitcoin: Q4 2026"
    assert with_context["hotwords"] == ["BTC", "ETF"]
    assert {k: v for k, v in with_context.items() if k not in {"system_prompt", "hotwords"}} == {
        k: v for k, v in base.items() if k not in {"system_prompt", "hotwords"}
    }


@pytest.mark.parametrize("context", [None, AsrContext()])
def test_q_ctx_without_context_text_sends_none(monkeypatch: pytest.MonkeyPatch, context: AsrContext | None) -> None:
    model = FakeModel()
    install_fake(monkeypatch, model)

    engine("q-ctx").transcribe(clip(5), language=None, context=context)

    assert model.calls[0][1]["system_prompt"] is None
    assert model.calls[0][1]["hotwords"] is None


@pytest.mark.parametrize(
    ("seconds", "max_tokens"),
    [(1.0, 64), (5.0, 64), (5.4, 65), (10.0, 120), (30.0, 360)],
)
def test_token_budget_scales_with_duration_above_a_floor(
    monkeypatch: pytest.MonkeyPatch, seconds: float, max_tokens: int
) -> None:
    model = FakeModel()
    install_fake(monkeypatch, model)

    raw = engine("q-base").transcribe(clip(seconds), language=None, context=None)

    kwargs = model.calls[0][1]
    assert kwargs["max_tokens"] == max_tokens == raw.max_tokens
    assert kwargs["chunk_duration"] == pytest.approx(seconds + 10)


def test_result_carries_text_language_and_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch, FakeModel(text="  你好 world ", language=["Chinese"], tokens=360))

    raw = engine("q-base").transcribe(clip(30), language=None, context=None)

    assert (raw.text, raw.language) == ("你好 world", "Chinese")
    assert (raw.generated_tokens, raw.max_tokens) == (360, 360)
    assert raw.words == ()


def test_a_forced_language_is_passed_by_name_not_code(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel()
    install_fake(monkeypatch, model)

    engine("q-base").transcribe(clip(5), language="zh", context=None)

    assert model.calls[0][1]["language"] == "Chinese"
    with pytest.raises(ConfigError, match="'fr'"):
        engine("q-base").transcribe(clip(5), language="fr", context=None)


def test_the_model_loads_from_the_pinned_repo_and_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    loads = install_fake(monkeypatch, FakeModel())

    engine("q-base")

    spec = registry.get("q-base").spec
    assert loads == [(spec.repo, {"revision": spec.revision})]


def test_transcribing_before_loading_fails_loudly() -> None:
    definition = registry.get("q-base")
    with pytest.raises(RuntimeError, match="load"):
        qwen3.create(definition.spec, definition.variant).transcribe(clip(1), language=None, context=None)


def test_a_missing_asr_extra_is_a_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", None)
    definition = registry.get("q-base")
    with pytest.raises(DependencyError, match="ytverbatim\\[asr\\]"):
        qwen3.create(definition.spec, definition.variant).load()


def test_the_variant_must_say_whether_to_use_context() -> None:
    with pytest.raises(ConfigError, match="context"):
        qwen3.create(registry.get("q-base").spec, {})
