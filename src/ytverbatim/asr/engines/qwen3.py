"""Qwen3-ASR through mlx-audio, for the `q-base` and `q-ctx` hypotheses.

`mlx_audio` is untyped and heavy, so it is imported on `load()` and never leaks past this module.
"""

import importlib
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast

from pydantic import JsonValue

from ytverbatim.asr.protocol import AsrContext, EngineSpec, RawChunk
from ytverbatim.errors import ConfigError, DependencyError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

SAMPLE_RATE = 16_000
_MIN_TOKENS = 64
_TOKENS_PER_SECOND = 12
# mlx-audio splits anything longer than `chunk_duration`; this headroom keeps our own chunks whole.
_CHUNK_HEADROOM_S = 10.0

# Qwen3 names languages in English; an ISO code would be passed through unchanged and decode wrongly.
_LANGUAGE_NAMES = {"zh": "Chinese", "en": "English"}

# Every default of `Model.generate` that could change a decode is pinned, so an mlx-audio upgrade cannot shift results.
_FIXED_PARAMS: dict[str, object] = {
    "batch_size": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "min_tokens_to_keep": 1,
    "repetition_penalty": None,
    "repetition_context_size": 100,
    "prefill_step_size": 2048,
    "min_chunk_duration": 1.0,
    "verbose": False,
    "stream": False,
}


class _Output(Protocol):
    text: str
    language: str | list[str] | None
    generation_tokens: int


class _Model(Protocol):
    def generate(self, audio: Any, **kwargs: Any) -> _Output: ...


class _Loader(Protocol):
    def load(self, model_path: str, **kwargs: Any) -> _Model: ...


def max_tokens_for(seconds: float) -> int:
    """About twice the token count of fast speech; a decode that reaches it is flagged as truncated."""
    return max(_MIN_TOKENS, math.ceil(_TOKENS_PER_SECOND * seconds))


class Qwen3Engine:
    def __init__(self, spec: EngineSpec, *, use_context: bool) -> None:
        self._spec = spec
        self._use_context = use_context
        self._model: _Model | None = None

    def load(self) -> None:
        try:
            utils = importlib.import_module("mlx_audio.stt.utils")
        except ImportError as exc:
            raise DependencyError("install ytverbatim[asr]") from exc
        # The download honors HF_HUB_CACHE, which the worker exports before anything imports huggingface_hub.
        self._model = cast(_Loader, utils).load(self._spec.repo, revision=self._spec.revision)

    def transcribe(self, audio: "NDArray[np.float32]", *, language: str | None, context: AsrContext | None) -> RawChunk:
        if self._model is None:
            raise RuntimeError("Qwen3Engine.transcribe called before load()")
        seconds = len(audio) / SAMPLE_RATE
        max_tokens = max_tokens_for(seconds)
        biasing = context if self._use_context and context is not None else AsrContext()
        output = self._model.generate(
            audio,
            **_FIXED_PARAMS,
            max_tokens=max_tokens,
            chunk_duration=seconds + _CHUNK_HEADROOM_S,
            language=_language_name(language),
            system_prompt=biasing.system_prompt or None,
            hotwords=list(biasing.hotwords) or None,
        )
        return RawChunk(
            text=output.text.strip(),
            language=_emitted_language(output.language),
            generated_tokens=output.generation_tokens,
            max_tokens=max_tokens,
        )


def create(spec: EngineSpec, variant: Mapping[str, JsonValue]) -> Qwen3Engine:
    use_context = variant.get("context")
    if not isinstance(use_context, bool):
        raise ConfigError(f"{spec.id} needs a boolean `context` variant, got {use_context!r}")
    return Qwen3Engine(spec, use_context=use_context)


def _language_name(language: str | None) -> str | None:
    if language is None:
        return None
    try:
        return _LANGUAGE_NAMES[language.lower()]
    except KeyError:
        raise ConfigError(f"Qwen3-ASR language must be one of {', '.join(_LANGUAGE_NAMES)}, got {language!r}") from None


def _emitted_language(language: str | list[str] | None) -> str | None:
    # mlx-audio reports "English" when the model's language tag cannot be parsed, so this is a record, not a detection.
    if isinstance(language, list):
        return language[0] if language else None
    return language
