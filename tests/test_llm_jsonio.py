from dataclasses import dataclass, field
from typing import Literal

import pytest
from pydantic import BaseModel

from ytverbatim.errors import LLMOutputError
from ytverbatim.llm.jsonio import complete_json, extract_json_object
from ytverbatim.llm.protocol import LLMRequest, LLMResponse


class Verdict(BaseModel):
    ok: Literal[True]
    note: str = ""


@dataclass
class ScriptedClient:
    replies: list[str]
    requests: list[LLMRequest] = field(default_factory=list)

    @property
    def id(self) -> str:
        return "fake:scripted"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(text=self.replies.pop(0), usage=None, elapsed_s=0.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"ok": true}', {"ok": True}),
        ('Sure!\n```json\n{"ok": true, "note": "a {brace}"}\n```\nDone.', {"ok": True, "note": "a {brace}"}),
        ('{not json} then {"ok": true}', {"ok": True}),
        ('[1, 2] {"a": {"b": 1}}', {"a": {"b": 1}}),
    ],
)
def test_extract_first_object(text: str, expected: dict[str, object]) -> None:
    assert extract_json_object(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2]", "{broken"])
def test_extract_raises_without_an_object(text: str) -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json_object(text)


def test_first_reply_valid() -> None:
    fake = ScriptedClient(['```json\n{"ok": true}\n```'])
    result = complete_json(fake, LLMRequest(system="Judge.", user="Is it ok?"), Verdict)
    assert result == Verdict(ok=True)
    (request,) = fake.requests
    assert request.json_schema == Verdict.model_json_schema()
    assert request.system.startswith("Judge.\n\n")
    assert '"ok"' in request.system


def test_explicit_schema_is_kept() -> None:
    fake = ScriptedClient(['{"ok": true}'])
    complete_json(fake, LLMRequest(system="", user="u", json_schema={"type": "object"}), Verdict)
    assert fake.requests[0].json_schema == {"type": "object"}
    assert fake.requests[0].system.startswith("Reply with exactly one JSON object")


def test_retries_once_with_the_error_then_succeeds() -> None:
    fake = ScriptedClient(['{"ok": false}', '{"ok": true}'])
    assert complete_json(fake, LLMRequest(system="s", user="Is it ok?"), Verdict).ok
    first, second = fake.requests
    assert second.system == first.system
    assert second.user.startswith('Is it ok?\n\n<previous_reply>\n{"ok": false}\n</previous_reply>')
    assert "does not match the schema" in second.user


def test_retry_then_raise() -> None:
    fake = ScriptedClient(["nope", "still nope"])
    with pytest.raises(LLMOutputError, match="fake:scripted returned invalid JSON twice"):
        complete_json(fake, LLMRequest(system="", user="u"), Verdict)
    assert len(fake.requests) == 2
    assert "no JSON object" in fake.requests[1].user


def test_long_previous_reply_is_truncated_in_the_retry() -> None:
    fake = ScriptedClient(["x" * 5000, '{"ok": true}'])
    complete_json(fake, LLMRequest(system="", user="u"), Verdict)
    assert "x" * 2000 + " [truncated]" in fake.requests[1].user
    assert "x" * 2001 not in fake.requests[1].user
