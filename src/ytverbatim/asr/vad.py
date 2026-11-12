import importlib
from typing import Any, Protocol, cast

from ytverbatim.asr.audio import SAMPLE_RATE, Audio
from ytverbatim.asr.chunks import DEFAULT_VAD_PARAMS, SILERO_REPO, SILERO_REVISION, Region, VadParams
from ytverbatim.errors import DependencyError

__all__ = ["DEFAULT_VAD_PARAMS", "SILERO_REPO", "SILERO_REVISION", "VadParams", "speech_regions"]


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
