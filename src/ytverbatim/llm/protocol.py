from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    json_schema: Mapping[str, object] | None = None
    max_output_tokens: int = 4096
    temperature: float | None = None


@dataclass(frozen=True)
class Usage:
    """Billed tokens; `input_tokens` includes cached prompt tokens, `output_tokens` includes thinking."""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage | None
    elapsed_s: float


@dataclass(frozen=True)
class Readiness:
    ready: bool
    detail: str


class LLMClient(Protocol):
    @property
    def id(self) -> str:
        """Stable `<transport>:<model>` key; checkpoints of LLM stages are keyed by it."""
        ...

    def complete(self, request: LLMRequest) -> LLMResponse: ...
