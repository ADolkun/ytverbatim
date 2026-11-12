"""Deterministic stand-in engine, named by dotted path (`fake_engine.create`) so the worker imports it lazily."""

import wave
from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue

from ytverbatim.asr.protocol import AsrContext, EngineDef, EngineSpec, RawChunk
from ytverbatim.contract.models import Word

RATE = 16_000


def write_silence(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        writer.writeframes(b"\x00\x00" * round(seconds * RATE))
    return path


def definition(max_chunk_s: float | None = 30.0, **variant: object) -> EngineDef:
    spec = EngineSpec("fake", "fake", "example/fake", "0" * 40, max_chunk_s, False, False, True, RATE)
    return EngineDef(spec, "fake_engine.create", dict(variant))  # type: ignore[arg-type]


class FakeEngine:
    def __init__(self, variant: Mapping[str, JsonValue]) -> None:
        self.variant = variant
        self.calls = 0

    def _log(self, line: str) -> None:
        target = self.variant.get("calls_log")
        if isinstance(target, str):
            with Path(target).open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def load(self) -> None:
        self._log("load")

    def transcribe(self, audio: object, *, language: str | None, context: AsrContext | None) -> RawChunk:
        samples = len(audio)  # type: ignore[arg-type]
        self._log(f"transcribe {samples}")
        self.calls += 1
        if self.variant.get("fail_on_call") == self.calls:
            raise RuntimeError("fake decoder exploded")
        cap = bool(self.variant.get("hit_cap"))
        return RawChunk(
            text=f"{language or 'auto'}:{context.system_prompt if context else ''}:{samples}",
            language=language,
            generated_tokens=64 if cap else 5,
            max_tokens=64,
            confidence=0.9,
            words=(Word(text="a", start=0.0, end=0.5), Word(text="b", start=0.5, end=1.0, conf=0.8)),
        )


def create(spec: EngineSpec, variant: Mapping[str, JsonValue]) -> FakeEngine:
    return FakeEngine(variant)
