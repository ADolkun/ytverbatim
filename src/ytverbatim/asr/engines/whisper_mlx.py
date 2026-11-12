"""The Whisper control (`w-prod`, `w-vad`): mlx-whisper decoding exactly as the production pipeline does it.

Both modes make the same call. `whole_file` hands over the entire recording and lets mlx-whisper window it;
`chunk` hands over one planned chunk (at most 30 s). Production quirks are kept on purpose, for example the Mandarin
punctuation prompt that is only sent when a language is declared; fixing them belongs to a later text unit.
"""

import importlib
import logging
import re
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast

from pydantic import JsonValue

from ytverbatim.asr.protocol import AsrContext, EngineSpec, RawChunk
from ytverbatim.contract.models import Word
from ytverbatim.errors import ConfigError, DependencyError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

Mode = Literal["whole_file", "chunk"]

# A single fixed temperature would disable Whisper's fallback re-decode and let a repetition loop run unchecked.
TEMPERATURE_SCHEDULE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
ZH_INITIAL_PROMPT = "以下是普通话内容，请使用简体中文并加上标点符号。"
_WEIGHT_FILES = ("weights.safetensors", "weights.npz")


class _MlxCore(Protocol):
    def set_cache_limit(self, limit: int) -> int: ...
    def clear_cache(self) -> None: ...


class _WhisperWord(TypedDict, total=False):
    word: str
    start: float | None
    end: float | None
    probability: float


class _WhisperSegment(TypedDict, total=False):
    words: list[_WhisperWord]


class _WhisperModule(Protocol):
    def transcribe(self, audio: Any, **kwargs: Any) -> dict[str, Any]: ...


class _ModelHolder(Protocol):
    def get_model(self, model_path: str, dtype: object) -> object: ...


def create(spec: EngineSpec, variant: Mapping[str, JsonValue]) -> "WhisperEngine":
    mode = variant.get("mode")
    if mode not in ("whole_file", "chunk"):
        raise ConfigError(f"{spec.id}: Whisper mode must be 'whole_file' or 'chunk', got {mode!r}")
    if (mode == "whole_file") != (spec.max_chunk_s is None):
        raise ConfigError(f"{spec.id}: mode {mode!r} contradicts max_chunk_s={spec.max_chunk_s}")
    return WhisperEngine(spec, mode)


def snapshot_path(spec: EngineSpec) -> Path:
    """Return the pinned snapshot in the default Hugging Face cache.

    The worker points `HF_HUB_CACHE` at its own candidates directory; that variable is deliberately ignored so the
    existing 2.9 GB snapshot is reused instead of downloaded again.
    """
    constants = importlib.import_module("huggingface_hub.constants")
    cache = Path(cast(str, constants.default_cache_path))
    path = cache / f"models--{spec.repo.replace('/', '--')}" / "snapshots" / spec.revision
    if not (path / "config.json").is_file() or not any((path / name).is_file() for name in _WEIGHT_FILES):
        raise DependencyError(
            f"the {spec.id} model snapshot is missing at {path}; download it once into the default Hugging Face "
            f"cache with: hf download {spec.repo} --revision {spec.revision}"
        )
    return path


class WhisperEngine:
    def __init__(self, spec: EngineSpec, mode: Mode) -> None:
        self.spec = spec
        self.mode = mode
        self._model_path: str | None = None
        self._whisper: _WhisperModule | None = None
        self._mx: _MlxCore | None = None

    def load(self) -> None:
        path = snapshot_path(self.spec)
        try:
            whisper = importlib.import_module("mlx_whisper")
            mx = importlib.import_module("mlx.core")
            holder = cast(_ModelHolder, importlib.import_module("mlx_whisper.transcribe").ModelHolder)
        except ImportError as exc:
            raise DependencyError("install ytverbatim[whisper]") from exc
        # transcribe() reuses the holder's model for the same path and dtype, so this moves the load out of the first
        # decode and times it as load.
        holder.get_model(str(path), mx.float16)
        self._model_path = str(path)
        self._whisper = cast(_WhisperModule, whisper)
        self._mx = cast(_MlxCore, mx)

    def transcribe(self, audio: "NDArray[np.float32]", *, language: str | None, context: AsrContext | None) -> RawChunk:
        if self._whisper is None or self._mx is None or self._model_path is None:
            raise RuntimeError("load() must be called before transcribe()")
        hint = _normalize_language(language)
        with _bounded_mlx_cache(self._mx):
            result = self._whisper.transcribe(
                audio,
                path_or_hf_repo=self._model_path,
                word_timestamps=True,
                language=hint,
                initial_prompt=ZH_INITIAL_PROMPT if hint is not None and hint.startswith("zh") else None,
                condition_on_previous_text=False,
                temperature=TEMPERATURE_SCHEDULE,
            )
        words = _words(cast(Sequence[_WhisperSegment], result.get("segments") or ()))
        probabilities = [w.conf for w in words if w.conf is not None]
        return RawChunk(
            text=str(result.get("text", "")).strip(),
            language=cast("str | None", result.get("language")),
            confidence=sum(probabilities) / len(probabilities) if probabilities else None,
            words=words,
        )


def _normalize_language(code: str | None) -> str | None:
    """Reduce a tag such as `zh-Hans` to the primary code Whisper accepts.

    An unknown tag auto-detects with a warning, as in production.
    """
    if not code or code.strip().lower() == "auto":
        return None
    primary = re.split(r"[-_]", code.strip(), maxsplit=1)[0].lower()
    if not primary:
        return None
    tokenizer = importlib.import_module("mlx_whisper.tokenizer")
    known = set(cast(Mapping[str, str], tokenizer.LANGUAGES)) | set(
        cast(Mapping[str, str], tokenizer.TO_LANGUAGE_CODE).values()
    )
    if primary not in known:
        log.warning("unknown language %r; letting Whisper auto-detect", code)
        return None
    return primary


def _words(segments: Sequence[_WhisperSegment]) -> tuple[Word, ...]:
    return tuple(
        Word(text=w["word"], start=w["start"], end=w["end"], conf=w.get("probability"))
        for segment in segments
        for w in segment.get("words") or ()
        if w.get("start") is not None and w.get("end") is not None
    )


@contextmanager
def _bounded_mlx_cache(mx: _MlxCore) -> Generator[None]:
    """Run inference with unused-buffer caching off so the allocator does not hold a whole decode's temporaries."""
    previous = mx.set_cache_limit(0)
    failed = False
    try:
        mx.clear_cache()
        yield
    except BaseException:
        failed = True
        raise
    finally:
        try:
            mx.clear_cache()
        except Exception:
            if not failed:
                raise
            log.warning("MLX cache cleanup failed while preserving the inference error", exc_info=True)
        finally:
            mx.set_cache_limit(previous)
