import importlib
from dataclasses import dataclass
from typing import Any, Protocol, cast

from ytverbatim.asr.audio import SAMPLE_RATE, Audio
from ytverbatim.asr.chunks import Region
from ytverbatim.errors import DependencyError

SILERO_REPO = "mlx-community/silero-vad"
# Pinned so the VAD weights, and therefore the chunk plan, cannot change under a recorded run.
SILERO_REVISION = "7bc17f22d3c0451bd3a6cd71e759b009271ff49a"


@dataclass(frozen=True, slots=True)
class VadParams:
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 300
    speech_pad_ms: int = 100


DEFAULT_VAD_PARAMS = VadParams()


class _SileroModel(Protocol):
    def get_speech_timestamps(self, audio: Any, **kwargs: Any) -> list[dict[str, float]]: ...


def speech_regions(audio: Audio, params: VadParams = DEFAULT_VAD_PARAMS) -> list[Region]:
    """Speech regions in seconds, already padded by `params.speech_pad_ms`."""
    try:
        vad = importlib.import_module("mlx_audio.vad")
    except ImportError as exc:
        raise DependencyError("install ytverbatim[asr]") from exc
    model = cast(_SileroModel, vad.load(SILERO_REPO, revision=SILERO_REVISION))
    stamps = model.get_speech_timestamps(
        audio.slice(),
        sample_rate=SAMPLE_RATE,
        threshold=params.threshold,
        min_speech_duration_ms=params.min_speech_ms,
        min_silence_duration_ms=params.min_silence_ms,
        speech_pad_ms=params.speech_pad_ms,
        return_seconds=True,
    )
    return [Region(float(stamp["start"]), float(stamp["end"])) for stamp in stamps]
