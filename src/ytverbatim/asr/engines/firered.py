"""FireRedASR2-AED adapter (hypothesis `f-aed`).

The model emits lowercase text because mlx-audio's `_detokenize` lowercases every token. The adapter returns that
text untouched: the eval scores the raw output, and a Latin ticker coming out as "nvda" counts as an engine error.
There is no punctuation guarantee, no word timestamps, no language selection and no context biasing, so `language`
and `context` are ignored.
"""

import importlib
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import JsonValue

from ytverbatim.asr.audio import SAMPLE_RATE
from ytverbatim.asr.protocol import AsrContext, Engine, EngineSpec, RawChunk
from ytverbatim.errors import ConfigError, DependencyError, EngineError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Above every chunk we hand over, so mlx-audio never re-splits an input with its Qwen splitter.
CHUNK_DURATION_S = 60.0


class _Output(Protocol):
    text: str
    segments: list[dict[str, Any]]
    generation_tokens: int


class _Model(Protocol):
    def generate(self, audio: object, **kwargs: Any) -> _Output: ...


class FireRedEngine:
    def __init__(self, spec: EngineSpec, beam_size: int) -> None:
        if spec.max_chunk_s is None:
            raise ConfigError(f"{spec.id} decodes bounded chunks, but its spec has no max_chunk_s")
        self._spec = spec
        self._max_input_s: float = spec.max_chunk_s
        self._beam_size = beam_size
        self._model: _Model | None = None

    def load(self) -> None:
        spec = self._spec
        try:
            stt = importlib.import_module("mlx_audio.stt")
        except ImportError as exc:
            raise DependencyError("install ytverbatim[asr]") from exc
        try:
            self._model = cast(_Model, stt.load(spec.repo, revision=spec.revision))
        except Exception as exc:
            cache = os.environ.get("HF_HUB_CACHE", "the default Hugging Face cache")
            raise EngineError(
                f"cannot load {spec.id} from {spec.repo}@{spec.revision} (weights are fetched into {cache} on first "
                f"use, so this also fails offline or on a full disk): {exc}"
            ) from exc

    def transcribe(self, audio: "NDArray[np.float32]", *, language: str | None, context: AsrContext | None) -> RawChunk:
        model = self._model
        if model is None:
            raise EngineError(f"{self._spec.id}: load() must run before transcribe()")
        seconds = len(audio) / SAMPLE_RATE
        if seconds > self._max_input_s:
            raise EngineError(f"{self._spec.id} takes at most {self._max_input_s:g}s per chunk, got {seconds:.1f}s")
        mx = importlib.import_module("mlx.core")
        output = model.generate(mx.array(audio), beam_size=self._beam_size, chunk_duration=CHUNK_DURATION_S)
        # One chunk is one segment; an empty decode reports 0.0, which is "no tokens", not "low confidence".
        confidence = float(output.segments[0]["confidence"]) if output.text else None
        # The decoder's step cap is the encoder length and is not exposed, so a cap hit cannot be detected.
        return RawChunk(
            text=output.text, generated_tokens=output.generation_tokens, max_tokens=None, confidence=confidence
        )


def create(spec: EngineSpec, variant: Mapping[str, JsonValue]) -> Engine:
    beam_size = variant.get("beam_size")
    if not isinstance(beam_size, int) or isinstance(beam_size, bool) or beam_size < 1:
        raise ConfigError(f"{spec.id} needs a positive integer variant beam_size, got {beam_size!r}")
    return FireRedEngine(spec, beam_size)
