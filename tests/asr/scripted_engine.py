"""Engine that answers from a script, so guard tests control exactly what each decode returns."""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import JsonValue

from ytverbatim.asr.protocol import AsrContext, EngineSpec, RawChunk

LOOP_TEXT = " ".join(["repeat risk"] * 10)


def looping(tokens: int = 64) -> RawChunk:
    return RawChunk(text=LOOP_TEXT, generated_tokens=tokens, max_tokens=64)


def sound(text: str = "a sound decode", tokens: int = 5) -> RawChunk:
    return RawChunk(text=text, generated_tokens=tokens, max_tokens=64)


class ScriptedEngine:
    """Replies in order and repeats the last reply once the script runs out; keeps what it was asked to decode."""

    def __init__(self, script: Sequence[RawChunk]) -> None:
        self.script = script
        self.heard: list[NDArray[np.float32]] = []

    def load(self) -> None:
        pass

    def transcribe(self, audio: NDArray[np.float32], *, language: str | None, context: AsrContext | None) -> RawChunk:
        self.heard.append(audio)
        return self.script[min(len(self.heard), len(self.script)) - 1]


def create(spec: EngineSpec, variant: Mapping[str, JsonValue]) -> ScriptedEngine:
    """A worker-loadable engine whose first decode loops; with `always`, every decode does."""
    loop = looping()
    return ScriptedEngine([loop] if variant.get("always") else [loop, sound()])
