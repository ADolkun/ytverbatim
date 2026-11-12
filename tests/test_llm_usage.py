from dataclasses import dataclass

import pytest

from ytverbatim.contract.models import LLMUsage
from ytverbatim.errors import LLMOutputError, LLMUnavailableError
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Usage
from ytverbatim.llm.usage import UsageMeter


@dataclass
class FixedClient:
    usage: Usage | None
    name: str = "fake:fixed"

    @property
    def id(self) -> str:
        return self.name

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(text="{}", usage=self.usage, elapsed_s=0.1)


def test_accumulates_per_stage_and_client() -> None:
    meter = UsageMeter()
    meter.record("correct", "claude-cli:sonnet", Usage(10, 2))
    meter.record("correct", "claude-cli:sonnet", Usage(5, 3))
    meter.record("judge", "claude-cli:sonnet", Usage(1, 1))
    meter.record("correct", "openai-compat:qwen", Usage(7, 7))
    assert meter.summary() == (
        LLMUsage(stage="correct", client="claude-cli:sonnet", calls=2, input_tokens=15, output_tokens=5),
        LLMUsage(stage="judge", client="claude-cli:sonnet", calls=1, input_tokens=1, output_tokens=1),
        LLMUsage(stage="correct", client="openai-compat:qwen", calls=1, input_tokens=7, output_tokens=7),
    )


def test_one_unmetered_call_makes_the_totals_unknown() -> None:
    meter = UsageMeter()
    meter.record("s", "c", Usage(10, 2))
    meter.record("s", "c", None)
    meter.record("s", "c", Usage(1, 1))
    assert meter.summary() == (LLMUsage(stage="s", client="c", calls=3, input_tokens=None, output_tokens=None),)


def test_metered_client_records_every_completed_call() -> None:
    meter = UsageMeter()
    client = meter.metered("punctuate", FixedClient(Usage(4, 6)))
    assert client.id == "fake:fixed"
    client.complete(LLMRequest(system="", user="a"))
    client.complete(LLMRequest(system="", user="b"))
    assert meter.summary() == (
        LLMUsage(stage="punctuate", client="fake:fixed", calls=2, input_tokens=8, output_tokens=12),
    )


def test_empty_meter() -> None:
    assert UsageMeter().summary() == ()


@dataclass
class FailingClient:
    error: Exception
    name: str = "fake:failing"

    @property
    def id(self) -> str:
        return self.name

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise self.error


def test_a_reply_rejected_after_it_arrived_is_still_billed() -> None:
    meter = UsageMeter()
    cut_off = LLMOutputError("cut off", responded=True, usage=Usage(30, 4096))
    with pytest.raises(LLMOutputError):
        meter.metered("correct", FailingClient(cut_off)).complete(LLMRequest(system="", user="a"))
    assert meter.summary() == (
        LLMUsage(stage="correct", client="fake:failing", calls=1, input_tokens=30, output_tokens=4096),
    )


def test_a_rejected_reply_with_unreported_usage_makes_the_totals_unknown() -> None:
    meter = UsageMeter()
    with pytest.raises(LLMOutputError):
        meter.metered("s", FailingClient(LLMOutputError("empty", responded=True))).complete(
            LLMRequest(system="", user="a")
        )
    assert meter.summary() == (
        LLMUsage(stage="s", client="fake:failing", calls=1, input_tokens=None, output_tokens=None),
    )


def test_a_call_that_got_no_reply_is_not_recorded() -> None:
    meter = UsageMeter()
    with pytest.raises(LLMUnavailableError):
        meter.metered("s", FailingClient(LLMUnavailableError("down"))).complete(LLMRequest(system="", user="a"))
    assert meter.summary() == ()
