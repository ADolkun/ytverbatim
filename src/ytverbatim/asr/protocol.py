from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import JsonValue

from ytverbatim.contract.models import Word

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

ChunkFlag = Literal["forced_cut", "token_cap_hit", "decoder_anomaly", "empty_output", "redecoded"]
LanguagePolicy = Literal["auto", "pass"]


@dataclass(frozen=True, slots=True)
class EngineSpec:
    """What the pipeline needs to know about an engine without loading it.

    `max_chunk_s` is None for a whole-file hypothesis. `sample_rate` is the rate the model consumes; the worker
    always hands over 16 kHz audio and an adapter whose model wants another rate converts it itself.
    """

    id: str
    family: str
    repo: str
    revision: str
    max_chunk_s: float | None
    native_timestamps: bool
    emits_punctuation: bool
    supports_context: bool
    sample_rate: int


@dataclass(frozen=True, slots=True)
class EngineDef:
    """A hypothesis as the worker receives it: spec, adapter factory and variant parameters.

    The variant's `language` is the engine's language policy: `pass` hands it the file's metadata language as
    production does, `auto` always lets it detect the language itself.
    """

    spec: EngineSpec
    factory: str
    variant: dict[str, JsonValue]

    def language_policy(self) -> LanguagePolicy:
        policy = self.variant.get("language")
        if policy == "auto" or policy == "pass":
            return policy
        raise ValueError(f"{self.spec.id} needs the variant language set to 'auto' or 'pass', got {policy!r}")

    def engine_language(self, language: str | None) -> str | None:
        """Return the language hint this engine decodes with for a file whose metadata says `language`."""
        return language if self.language_policy() == "pass" else None


@dataclass(frozen=True, slots=True)
class AsrContext:
    system_prompt: str = ""
    hotwords: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RawChunk:
    """One decode. Word times are relative to the start of the audio handed to the engine."""

    text: str
    language: str | None = None
    generated_tokens: int | None = None
    max_tokens: int | None = None
    confidence: float | None = None
    words: tuple[Word, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkInput:
    """What the worker handed to the engine for one chunk, kept so a guard can decode again."""

    audio: "NDArray[np.float32]"
    start: float
    end: float
    language: str | None
    context: AsrContext | None


@dataclass(frozen=True, slots=True)
class GuardedChunk:
    raw: RawChunk
    flags: tuple[ChunkFlag, ...] = ()


class Engine(Protocol):
    def load(self) -> None: ...

    def transcribe(
        self, audio: "NDArray[np.float32]", *, language: str | None, context: AsrContext | None
    ) -> RawChunk: ...


class Guard(Protocol):
    """Quality check run on every decoded chunk; it may decode again through `engine` before returning."""

    def __call__(self, engine: Engine, chunk: ChunkInput, raw: RawChunk) -> GuardedChunk: ...


EngineFactory = Callable[[EngineSpec, Mapping[str, JsonValue]], Engine]
