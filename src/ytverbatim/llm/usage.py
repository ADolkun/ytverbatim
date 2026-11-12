from dataclasses import dataclass, field

from ytverbatim.contract.models import LLMUsage
from ytverbatim.errors import LLMError
from ytverbatim.llm.protocol import LLMClient, LLMRequest, LLMResponse, Usage


@dataclass
class _Tally:
    calls: int = 0
    input_tokens: int | None = 0
    output_tokens: int | None = 0


@dataclass
class UsageMeter:
    """Per-(stage, client) token totals for `context.json`'s `llm_usage`."""

    _tallies: dict[tuple[str, str], _Tally] = field(default_factory=dict)

    def record(self, stage: str, client_id: str, usage: Usage | None) -> None:
        tally = self._tallies.setdefault((stage, client_id), _Tally())
        tally.calls += 1
        # One unmetered call makes the stage total unknown; a partial sum would understate billed tokens.
        if usage is None:
            tally.input_tokens = None
            tally.output_tokens = None
            return
        if tally.input_tokens is not None:
            tally.input_tokens += usage.input_tokens
        if tally.output_tokens is not None:
            tally.output_tokens += usage.output_tokens

    def metered(self, stage: str, client: LLMClient) -> LLMClient:
        """`client`, with every call that got a reply recorded under `stage`, including a reply it then rejected."""
        return _MeteredClient(client, stage, self)

    def summary(self) -> tuple[LLMUsage, ...]:
        return tuple(
            LLMUsage(
                stage=stage,
                client=client_id,
                calls=tally.calls,
                input_tokens=tally.input_tokens,
                output_tokens=tally.output_tokens,
            )
            for (stage, client_id), tally in self._tallies.items()
        )


class _MeteredClient:
    def __init__(self, inner: LLMClient, stage: str, meter: UsageMeter) -> None:
        self._inner = inner
        self._stage = stage
        self._meter = meter

    @property
    def id(self) -> str:
        return self._inner.id

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            response = self._inner.complete(request)
        except LLMError as exc:
            # A reply cut off at the token limit was still billed in full.
            if exc.responded:
                self._meter.record(self._stage, self._inner.id, exc.usage)
            raise
        self._meter.record(self._stage, self._inner.id, response.usage)
        return response
