import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.message import Message

import pytest

from ytverbatim.errors import ConfigError, LLMOutputError, LLMUnavailableError
from ytverbatim.llm import openai_compat
from ytverbatim.llm.openai_compat import OpenAICompatClient
from ytverbatim.llm.protocol import LLMRequest, Usage

BASE = "http://127.0.0.1:8080/v1"


def reply(content: str = '{"ok": true}', *, finish: str = "stop", usage: object = None) -> bytes:
    payload: dict[str, object] = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": usage if usage is not None else {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }
    return json.dumps(payload).encode()


@dataclass
class StubOpener:
    body: bytes = field(default_factory=reply)
    raises: Exception | None = None
    requests: list[tuple[urllib.request.Request, float]] = field(default_factory=list)

    def __call__(self, request: urllib.request.Request, timeout_s: float) -> bytes:
        self.requests.append((request, timeout_s))
        if self.raises is not None:
            raise self.raises
        return self.body

    def sent(self) -> dict[str, object]:
        data = self.requests[-1][0].data
        assert isinstance(data, bytes)
        return json.loads(data)


def client(opener: StubOpener, **kwargs: object) -> OpenAICompatClient:
    return OpenAICompatClient(base_url=BASE + "/", model="qwen-local", opener=opener, **kwargs)


def test_request_body_and_headers() -> None:
    opener = StubOpener()
    request = LLMRequest(
        system="Be terse.", user="Hi", json_schema={"type": "object"}, max_output_tokens=64, temperature=0.0
    )
    response = client(opener, timeout_s=30.0).complete(request)

    http_request, timeout_s = opener.requests[0]
    assert http_request.full_url == f"{BASE}/chat/completions"
    assert http_request.get_method() == "POST"
    assert http_request.get_header("Content-type") == "application/json"
    assert http_request.get_header("Authorization") is None
    assert timeout_s == 30.0
    assert opener.sent() == {
        "model": "qwen-local",
        "messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}],
        "max_tokens": 64,
        "stream": False,
        "temperature": 0.0,
        "response_format": {"type": "json_schema", "json_schema": {"name": "response", "schema": {"type": "object"}}},
    }
    assert response.text == '{"ok": true}'
    assert response.usage == Usage(input_tokens=12, output_tokens=5)


def test_optional_fields_are_omitted() -> None:
    opener = StubOpener()
    client(opener).complete(LLMRequest(system="", user="Hi"))
    body = opener.sent()
    assert body["messages"] == [{"role": "user", "content": "Hi"}]
    assert "temperature" not in body
    assert "response_format" not in body


def test_response_format_can_be_turned_off() -> None:
    opener = StubOpener()
    client(opener, response_format=False).complete(LLMRequest(system="", user="Hi", json_schema={"type": "object"}))
    assert "response_format" not in opener.sent()


def test_api_key_comes_from_the_named_env_var_per_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_LLM_KEY", "sk-fake-1")
    opener = StubOpener()
    llm = client(opener, api_key_env="FAKE_LLM_KEY")
    assert "sk-fake-1" not in repr(vars(llm))
    llm.complete(LLMRequest(system="", user="Hi"))
    assert opener.requests[0][0].get_header("Authorization") == "Bearer sk-fake-1"

    monkeypatch.delenv("FAKE_LLM_KEY")
    with pytest.raises(ConfigError, match="FAKE_LLM_KEY"):
        llm.complete(LLMRequest(system="", user="Hi"))


def test_missing_usage_is_unknown() -> None:
    opener = StubOpener(body=reply(usage={}))
    assert client(opener).complete(LLMRequest(system="", user="Hi")).usage is None


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timed out"),
        urllib.error.URLError(TimeoutError("timed out")),
        urllib.error.URLError("connection refused"),
    ],
)
def test_timeouts_and_network_errors_are_unavailable(error: Exception) -> None:
    with pytest.raises(LLMUnavailableError, match=r"127\.0\.0\.1"):
        client(StubOpener(raises=error)).complete(LLMRequest(system="", user="Hi"))


def test_http_error_includes_status_and_body() -> None:
    error = urllib.error.HTTPError(
        f"{BASE}/chat/completions", 400, "Bad Request", Message(), io.BytesIO(b"no such model")
    )
    with pytest.raises(LLMUnavailableError, match=r"HTTP 400.*no such model"):
        client(StubOpener(raises=error)).complete(LLMRequest(system="", user="Hi"))


@pytest.mark.parametrize("body", [b"<html>", b'{"choices": []}', b"{}"])
def test_unusable_response_is_unavailable(body: bytes) -> None:
    with pytest.raises(LLMUnavailableError):
        client(StubOpener(body=body)).complete(LLMRequest(system="", user="Hi"))


def test_truncated_or_empty_reply_is_an_output_error_that_keeps_its_usage() -> None:
    with pytest.raises(LLMOutputError, match="max_tokens") as truncated:
        client(StubOpener(body=reply(finish="length"))).complete(LLMRequest(system="", user="Hi"))
    assert (truncated.value.responded, truncated.value.usage) == (True, Usage(input_tokens=12, output_tokens=5))
    with pytest.raises(LLMOutputError, match="empty") as empty:
        client(StubOpener(body=reply(content=""))).complete(LLMRequest(system="", user="Hi"))
    assert empty.value.responded


def test_no_reply_means_nothing_was_billed() -> None:
    with pytest.raises(LLMUnavailableError) as caught:
        client(StubOpener(raises=TimeoutError("timed out"))).complete(LLMRequest(system="", user="Hi"))
    assert (caught.value.responded, caught.value.usage) == (False, None)


@pytest.mark.parametrize(
    "base_url", ["http://example.com/v1", "http://192.168.1.20:8080/v1", "ftp://127.0.0.1/v1", "127.0.0.1:8080"]
)
def test_a_key_is_never_sent_in_clear_text_off_this_machine(monkeypatch: pytest.MonkeyPatch, base_url: str) -> None:
    monkeypatch.setenv("FAKE_LLM_KEY", "sk-fake-3")
    with pytest.raises(ConfigError, match="https"):
        OpenAICompatClient(base_url=base_url, model="m", api_key_env="FAKE_LLM_KEY")
    opener = StubOpener()
    status = openai_compat.readiness(base_url, "FAKE_LLM_KEY", opener=opener)
    assert not status.ready
    assert opener.requests == []


@pytest.mark.parametrize(
    "base_url",
    ["https://api.example.com/v1", "http://127.0.0.1:8080/v1", "http://localhost:11434/v1", "http://[::1]/v1"],
)
def test_a_key_may_go_over_https_or_to_loopback(base_url: str) -> None:
    OpenAICompatClient(base_url=base_url, model="m", api_key_env="FAKE_LLM_KEY")


def test_without_a_key_plain_http_anywhere_is_fine() -> None:
    OpenAICompatClient(base_url="http://192.168.1.20:8080/v1", model="m")


def test_constructor_requires_base_url_and_model() -> None:
    with pytest.raises(ConfigError, match="base_url"):
        OpenAICompatClient(base_url="", model="m")
    with pytest.raises(ConfigError, match="model"):
        OpenAICompatClient(base_url=BASE, model="")
    assert OpenAICompatClient(base_url=BASE, model="m").id == "openai-compat:m@127.0.0.1:8080"
    hosted = OpenAICompatClient(base_url="https://user:secret@api.example.com/v1", model="m")
    assert hosted.id == "openai-compat:m@api.example.com", "never credentials or path"


def test_readiness_gets_models_without_generating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_LLM_KEY", "sk-fake-2")
    opener = StubOpener(body=b'{"data": []}')
    status = openai_compat.readiness(BASE, "FAKE_LLM_KEY", opener=opener)
    assert status.ready
    request, _ = opener.requests[0]
    assert request.full_url == f"{BASE}/models"
    assert request.get_method() == "GET"
    assert request.data is None
    assert request.get_header("Authorization") == "Bearer sk-fake-2"


def test_readiness_not_ready() -> None:
    assert not openai_compat.readiness(None).ready
    assert not openai_compat.readiness(BASE, "FAKE_LLM_KEY_UNSET", opener=StubOpener()).ready
    down = StubOpener(raises=urllib.error.URLError("connection refused"))
    status = openai_compat.readiness(BASE, opener=down)
    assert not status.ready
    assert "connection refused" in status.detail
