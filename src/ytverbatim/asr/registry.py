import importlib
from collections.abc import Mapping
from importlib.util import find_spec

from pydantic import JsonValue

from ytverbatim.asr.protocol import Engine, EngineDef, EngineFactory, EngineSpec
from ytverbatim.errors import ConfigError, DependencyError

_MIMO_TOKENIZER = {
    "tokenizer_repo": "mlx-community/MiMo-Audio-Tokenizer",
    "tokenizer_revision": "6d451ed9a73024b4d33b87afa69e0dfd40d8f306",
}
_QWEN3_REPO = "mlx-community/Qwen3-ASR-1.7B-8bit"
_QWEN3_REVISION = "a8379a2e2f9e313c9292cdf1af4055ab56d50d55"
_WHISPER_REPO = "mlx-community/whisper-large-v3-mlx"
_WHISPER_REVISION = "49e6aa286ad60c14352c404340ded53710378a11"

# `emits_punctuation` is the vendors' claim for Qwen3 and MiMo and unverified for FireRed; the engine runs settle it.
# Whisper passes the metadata language as production does. The challengers auto-detect: Qwen3 accepts only English
# language names (§B2), FireRed has no language input, and MiMo knows only zh and en, so a metadata tag such as
# "zh+en" would either abort the decode or force a wrong language.
# Adapters are named by dotted path so that looking up a hypothesis never imports an engine library. Each entry also
# names the top-level modules its worker imports, so a missing extra fails before anything is downloaded; chunked
# hypotheses need mlx_audio for the Silero VAD that plans their chunks.
_WHISPER = ("mlx_whisper",)
_MLX_AUDIO = ("mlx_audio",)
_EXTRAS: Mapping[str, str] = {"mlx_whisper": "whisper", "mlx_audio": "asr"}
_ENTRIES: tuple[tuple[EngineDef, tuple[str, ...]], ...] = (
    (
        EngineDef(
            EngineSpec(
                id="w-prod",
                family="whisper",
                repo=_WHISPER_REPO,
                revision=_WHISPER_REVISION,
                max_chunk_s=None,
                native_timestamps=True,
                emits_punctuation=True,
                supports_context=False,
                sample_rate=16_000,
            ),
            "ytverbatim.asr.engines.whisper_mlx.create",
            {"mode": "whole_file", "language": "pass"},
        ),
        _WHISPER,
    ),
    (
        EngineDef(
            EngineSpec(
                id="w-vad",
                family="whisper",
                repo=_WHISPER_REPO,
                revision=_WHISPER_REVISION,
                max_chunk_s=30.0,
                native_timestamps=True,
                emits_punctuation=True,
                supports_context=False,
                sample_rate=16_000,
            ),
            "ytverbatim.asr.engines.whisper_mlx.create",
            {"mode": "chunk", "language": "pass"},
        ),
        (*_WHISPER, *_MLX_AUDIO),
    ),
    (
        EngineDef(
            EngineSpec(
                id="q-base",
                family="qwen3-asr",
                repo=_QWEN3_REPO,
                revision=_QWEN3_REVISION,
                max_chunk_s=30.0,
                native_timestamps=False,
                emits_punctuation=True,
                supports_context=False,
                sample_rate=16_000,
            ),
            "ytverbatim.asr.engines.qwen3.create",
            {"context": False, "language": "auto"},
        ),
        _MLX_AUDIO,
    ),
    (
        EngineDef(
            EngineSpec(
                id="q-ctx",
                family="qwen3-asr",
                repo=_QWEN3_REPO,
                revision=_QWEN3_REVISION,
                max_chunk_s=30.0,
                native_timestamps=False,
                emits_punctuation=True,
                supports_context=True,
                sample_rate=16_000,
            ),
            "ytverbatim.asr.engines.qwen3.create",
            {"context": True, "language": "auto"},
        ),
        _MLX_AUDIO,
    ),
    (
        EngineDef(
            EngineSpec(
                id="f-aed",
                family="firered-aed",
                repo="mlx-community/FireRedASR2-AED-mlx",
                revision="f3212eacfa49b851130b97c63653c8e06ee09bdb",
                max_chunk_s=30.0,
                native_timestamps=False,
                emits_punctuation=False,
                supports_context=False,
                sample_rate=16_000,
            ),
            "ytverbatim.asr.engines.firered.create",
            {"beam_size": 3, "language": "auto"},
        ),
        _MLX_AUDIO,
    ),
    (
        EngineDef(
            EngineSpec(
                id="m-4bit",
                family="mimo-asr",
                repo="mlx-community/MiMo-V2.5-ASR-MLX",
                revision="69813f0d57fb9bb5328735c4e907a4558b47d341",
                max_chunk_s=30.0,
                native_timestamps=False,
                emits_punctuation=True,
                supports_context=False,
                sample_rate=24_000,
            ),
            "ytverbatim.asr.engines.mimo.create",
            {**_MIMO_TOKENIZER, "language": "auto"},
        ),
        _MLX_AUDIO,
    ),
)
HYPOTHESES: Mapping[str, EngineDef] = {definition.spec.id: definition for definition, _ in _ENTRIES}
_MODULES: Mapping[str, tuple[str, ...]] = {definition.spec.id: modules for definition, modules in _ENTRIES}


def get(hypothesis_id: str) -> EngineDef:
    try:
        return HYPOTHESES[hypothesis_id]
    except KeyError:
        raise ConfigError(f"unknown ASR hypothesis {hypothesis_id!r}; known: {', '.join(HYPOTHESES)}") from None


def build_engine(definition: EngineDef) -> Engine:
    """Import the adapter named by `definition.factory` and construct the engine; this is the lazy import."""
    module_name, _, attribute = definition.factory.rpartition(".")
    try:
        factory: EngineFactory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise DependencyError(f"cannot load the {definition.spec.id} adapter {definition.factory}: {exc}") from exc
    variant: Mapping[str, JsonValue] = definition.variant
    return factory(definition.spec, variant)


def check_dependencies(definition: EngineDef) -> None:
    """Raise `DependencyError` naming the extra to install when the engine's libraries are not importable.

    Checked by `find_spec` in this interpreter, which is the one the worker runs in; nothing is imported.
    """
    missing = [module for module in _MODULES[definition.spec.id] if find_spec(module) is None]
    if missing:
        extras = ",".join(dict.fromkeys(_EXTRAS[module] for module in missing))
        raise DependencyError(
            f"the {definition.spec.id} engine needs {', '.join(missing)}, which is not installed; "
            f"install ytverbatim[{extras}] (in a checkout: uv sync --extra {extras.replace(',', ' --extra ')})"
        )
