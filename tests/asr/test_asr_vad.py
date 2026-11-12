import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from ytverbatim.asr import vad
from ytverbatim.asr.audio import Audio
from ytverbatim.asr.chunks import Region
from ytverbatim.asr.vad import SILERO_REPO, SILERO_REVISION, VadParams, speech_regions
from ytverbatim.errors import DependencyError


class FakeModel:
    def __init__(self, stamps: list[dict[str, float]]) -> None:
        self.stamps = stamps
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def get_speech_timestamps(self, audio: Any, **kwargs: Any) -> list[dict[str, float]]:
        self.calls.append((audio, kwargs))
        return self.stamps


def install_fake(monkeypatch: pytest.MonkeyPatch, model: FakeModel) -> list[tuple[str, dict[str, Any]]]:
    loads: list[tuple[str, dict[str, Any]]] = []
    package, module = ModuleType("mlx_audio"), ModuleType("mlx_audio.vad")

    def load(repo: str, **kwargs: Any) -> FakeModel:
        loads.append((repo, kwargs))
        return model

    module.load = load  # type: ignore[attr-defined]
    package.vad = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_audio", package)
    monkeypatch.setitem(sys.modules, "mlx_audio.vad", module)
    return loads


def test_regions_come_from_the_pinned_model_with_the_shared_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel([{"start": 0.5, "end": 3.25}, {"start": 4.0, "end": 9.0}])
    loads = install_fake(monkeypatch, model)
    audio = Audio(np.full(16_000 * 10, 1000, dtype=np.int16))

    assert speech_regions(audio) == [Region(0.5, 3.25), Region(4.0, 9.0)]

    assert loads == [(SILERO_REPO, {"revision": SILERO_REVISION})]
    samples, kwargs = model.calls[0]
    assert samples.dtype == np.float32
    assert len(samples) == 160_000
    assert kwargs == {
        "sample_rate": 16_000,
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 300,
        "speech_pad_ms": 100,
        "return_seconds": True,
    }


def test_parameters_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeModel([])
    install_fake(monkeypatch, model)
    params = VadParams(threshold=0.6, min_speech_ms=100, min_silence_ms=500, speech_pad_ms=50)
    assert speech_regions(Audio(np.zeros(16_000, dtype=np.int16)), params) == []
    assert model.calls[0][1]["threshold"] == 0.6
    assert model.calls[0][1]["speech_pad_ms"] == 50


def test_revision_is_a_full_commit_sha() -> None:
    assert len(vad.SILERO_REVISION) == 40
    int(vad.SILERO_REVISION, 16)


def test_missing_extra_names_the_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlx_audio", None)
    monkeypatch.setitem(sys.modules, "mlx_audio.vad", None)
    with pytest.raises(DependencyError, match=r"install ytverbatim\[asr\]"):
        speech_regions(Audio(np.zeros(16_000, dtype=np.int16)))
