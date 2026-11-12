"""MiMo-V2.5-ASR adapter (hypothesis `m-4bit`, the 4-bit screening contender).

mlx-audio silently misbehaves for MiMo in three ways, and the adapter sets each explicitly:
- the model reads 24 kHz audio and does not resample arrays, so the worker's 16 kHz chunk is resampled here;
- `max_tokens` defaults to 256 and truncates a long chunk without any signal, so it is derived from the duration;
- the audio tokenizer repo is resolved from the model's manifest without a revision, so it is pinned here.

The output is one segment with no word timestamps and no confidence.
"""

import importlib
import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import JsonValue

from ytverbatim.asr.audio import SAMPLE_RATE
from ytverbatim.asr.protocol import AsrContext, Engine, EngineSpec, RawChunk
from ytverbatim.errors import ConfigError, DependencyError, EngineError

try:
    import numpy as np
    from numpy.typing import NDArray
except ImportError as exc:
    raise DependencyError("install ytverbatim[asr]") from exc

MODEL_SAMPLE_RATE = 24_000
RESAMPLE_UP, RESAMPLE_DOWN = 3, 2
# Same budget as the other LLM-style engines: about twice the fastest measured speech rate in tokens per second.
TOKENS_PER_SECOND = 12
MIN_TOKENS = 64
TEMPERATURE = 0.0
TOP_P = 0.95
TOP_K = 0

_MODEL_MODULE = "mlx_audio.stt.models.mimo_v2_asr.asr"
# The model only distinguishes zh and en; code-switched ("zh+en") and every other language is left to auto-detection.
_LANGUAGES = {"zh": "zh", "chinese": "zh", "mandarin": "zh", "en": "en", "english": "en"}


class _Output(Protocol):
    text: str
    language: str | None
    generation_tokens: int


class _Model(Protocol):
    def generate(self, audio: object, **kwargs: Any) -> _Output: ...


class _Resampler(Protocol):
    def __call__(self, x: "NDArray[np.float32]", up: int, down: int) -> "NDArray[np.float32]": ...


def resample_to_model_rate(audio: "NDArray[np.float32]") -> "NDArray[np.float32]":
    try:
        resample_poly = cast(_Resampler, importlib.import_module("scipy.signal").resample_poly)
    except ImportError as exc:
        raise DependencyError("install ytverbatim[asr]") from exc
    return np.asarray(resample_poly(audio, RESAMPLE_UP, RESAMPLE_DOWN), dtype=np.float32)


def token_budget(seconds: float) -> int:
    return max(MIN_TOKENS, math.ceil(TOKENS_PER_SECOND * seconds))


def model_language(language: str | None) -> str | None:
    if language is None:
        return None
    return _LANGUAGES.get(language.lower().replace("_", "-").split("-")[0])


@contextmanager
def _pinned_tokenizer(model_class: Any, tokenizer_dir: Path) -> Iterator[None]:
    """Make `Model.post_load_hook` use the pinned tokenizer snapshot instead of resolving the manifest's repo name.

    The manifest names the repo without a revision, and redirecting that lookup is the only seam that does not mean
    copying the loader.
    """

    def pinned(_model_path: Path) -> Path:
        return tokenizer_dir

    original = vars(model_class)["_resolve_audio_tokenizer_dir"]
    model_class._resolve_audio_tokenizer_dir = staticmethod(pinned)
    try:
        yield
    finally:
        model_class._resolve_audio_tokenizer_dir = original


class MimoEngine:
    def __init__(self, spec: EngineSpec, tokenizer_repo: str, tokenizer_revision: str) -> None:
        self._spec = spec
        self._tokenizer_repo = tokenizer_repo
        self._tokenizer_revision = tokenizer_revision
        self._model: _Model | None = None

    def load(self) -> None:
        spec = self._spec
        try:
            stt = importlib.import_module("mlx_audio.stt")
            utils = importlib.import_module("mlx_audio.utils")
            asr = importlib.import_module(_MODEL_MODULE)
        except ImportError as exc:
            raise DependencyError("install ytverbatim[asr]") from exc
        try:
            tokenizer_dir = Path(utils.get_model_path(self._tokenizer_repo, revision=self._tokenizer_revision))
            with _pinned_tokenizer(asr.Model, tokenizer_dir):
                self._model = cast(_Model, stt.load(spec.repo, revision=spec.revision))
        except Exception as exc:
            cache = os.environ.get("HF_HUB_CACHE", "the default Hugging Face cache")
            raise EngineError(
                f"cannot load {spec.id} from {spec.repo}@{spec.revision} with tokenizer {self._tokenizer_repo}@"
                f"{self._tokenizer_revision} (weights are fetched into {cache} on first use, so this also fails "
                f"offline or on a full disk): {exc}"
            ) from exc

    def transcribe(self, audio: "NDArray[np.float32]", *, language: str | None, context: AsrContext | None) -> RawChunk:
        model = self._model
        if model is None:
            raise EngineError(f"{self._spec.id}: load() must run before transcribe()")
        max_tokens = token_budget(len(audio) / SAMPLE_RATE)
        output = model.generate(
            resample_to_model_rate(audio),
            max_tokens=max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            language=model_language(language),
            verbose=False,
        )
        # generation_tokens excludes the trailing stop token, so reaching the cap means the decode was cut off.
        return RawChunk(
            text=output.text,
            language=output.language,
            generated_tokens=output.generation_tokens,
            max_tokens=max_tokens,
        )


def create(spec: EngineSpec, variant: Mapping[str, JsonValue]) -> Engine:
    repo, revision = variant.get("tokenizer_repo"), variant.get("tokenizer_revision")
    if not isinstance(repo, str) or not isinstance(revision, str) or not repo or not revision:
        raise ConfigError(f"{spec.id} needs the variant keys tokenizer_repo and tokenizer_revision, both strings")
    return MimoEngine(spec, repo, revision)
