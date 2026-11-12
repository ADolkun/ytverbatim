"""Any OpenAI-compatible `/chat/completions` endpoint (mlx_lm.server, Ollama, hosted APIs) over stdlib HTTP."""

import ipaddress
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

from ytverbatim.errors import ConfigError, LLMOutputError, LLMUnavailableError
from ytverbatim.llm.jsonio import as_object
from ytverbatim.llm.protocol import LLMRequest, LLMResponse, Readiness, Usage

PROVIDER = "openai-compat"
_READINESS_TIMEOUT_S = 10.0
_BODY_PREVIEW = 300

# (request, timeout_s) -> response body. Tests inject a stub; HTTP and network errors propagate as urllib raises them.
Opener = Callable[[urllib.request.Request, float], bytes]


def urlopen(request: urllib.request.Request, timeout_s: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body: bytes = response.read()
        return body


class OpenAICompatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        timeout_s: float = 900.0,
        response_format: bool = True,
        opener: Opener = urlopen,
    ) -> None:
        if not base_url:
            raise ConfigError("openai-compat needs [llm] base_url, e.g. http://127.0.0.1:8080/v1")
        if not model:
            raise ConfigError("openai-compat needs [llm] model")
        _require_safe_transport(base_url, api_key_env)
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key_env = api_key_env
        self._timeout_s = timeout_s
        self._response_format = response_format
        self._opener = opener

    @property
    def id(self) -> str:
        # The same model name on another server is another model; the host (never credentials or path) tells them apart.
        parts = urllib.parse.urlsplit(self._base_url)
        host = parts.hostname or self._base_url
        return f"{PROVIDER}:{self._model}@{host}" + (f":{parts.port}" if parts.port else "")

    def body(self, request: LLMRequest) -> dict[str, object]:
        messages = [{"role": "system", "content": request.system}] if request.system else []
        messages.append({"role": "user", "content": request.user})
        body: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        # Off for servers that reject the field; complete_json also states the schema in the prompt.
        if request.json_schema is not None and self._response_format:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": request.json_schema},
            }
        return body

    def complete(self, request: LLMRequest) -> LLMResponse:
        http_request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(self.body(request), ensure_ascii=False).encode(),
            headers=_headers(self._api_key_env),
            method="POST",
        )
        started = time.monotonic()
        raw = _send(self._opener, http_request, self._timeout_s, self.id)
        elapsed = time.monotonic() - started
        choice, usage = _parse(raw, self.id)
        if choice.get("finish_reason") == "length":
            raise LLMOutputError(f"{self.id}: the reply hit max_tokens and was cut off", responded=True, usage=usage)
        content = as_object(choice.get("message")).get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMOutputError(f"{self.id}: empty reply", responded=True, usage=usage)
        return LLMResponse(text=content, usage=usage, elapsed_s=elapsed)


def readiness(
    base_url: str | None,
    api_key_env: str | None = None,
    *,
    opener: Opener = urlopen,
    timeout_s: float = _READINESS_TIMEOUT_S,
) -> Readiness:
    """`GET {base_url}/models`: the server answers and accepts the key. Sends no generation."""
    if not base_url:
        return Readiness(False, "no [llm] base_url configured")
    try:
        _require_safe_transport(base_url, api_key_env)
        request = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers=_headers(api_key_env), method="GET")
        _send(opener, request, timeout_s, PROVIDER)
    except (ConfigError, LLMUnavailableError) as exc:
        return Readiness(False, str(exc))
    return Readiness(True, f"{base_url} answered /models")


def _require_safe_transport(base_url: str, api_key_env: str | None) -> None:
    """Refuse to send a bearer key in clear text anywhere but this machine."""
    if api_key_env is None:
        return
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme == "https" or (parts.scheme == "http" and _is_loopback(parts.hostname)):
        return
    raise ConfigError(
        f"refusing to send the key from {api_key_env} to {base_url}: use https, or http only to a loopback address"
    )


def _is_loopback(host: str | None) -> bool:
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_key(api_key_env: str | None) -> None:
    """Raise `ConfigError` when the configured key variable is not set; sends nothing."""
    _headers(api_key_env)


def _headers(api_key_env: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key_env is not None:
        # Read per request and never kept, so the key lives only in the caller's environment.
        key = os.environ.get(api_key_env)
        if not key:
            raise ConfigError(f"environment variable {api_key_env} (named by [llm] api_key_env) is not set")
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _send(opener: Opener, request: urllib.request.Request, timeout_s: float, label: str) -> bytes:
    try:
        return opener(request, timeout_s)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:_BODY_PREVIEW].strip()
        raise LLMUnavailableError(f"{label}: HTTP {exc.code} from {request.full_url}: {detail}") from exc
    except TimeoutError as exc:
        raise LLMUnavailableError(f"{label}: no response from {request.full_url} within {timeout_s:g}s") from exc
    except urllib.error.URLError as exc:
        raise LLMUnavailableError(f"{label}: cannot reach {request.full_url}: {exc.reason}") from exc


def _parse(raw: bytes, label: str) -> tuple[dict[str, object], Usage | None]:
    try:
        payload = as_object(json.loads(raw))
    except ValueError as exc:
        raise LLMUnavailableError(f"{label}: the response is not JSON") from exc
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMUnavailableError(f"{label}: the response has no choices")
    counts = as_object(payload.get("usage"))
    prompt, completion = counts.get("prompt_tokens"), counts.get("completion_tokens")
    usage = (
        Usage(input_tokens=prompt, output_tokens=completion)
        if isinstance(prompt, int) and isinstance(completion, int)
        else None
    )
    return as_object(choices[0]), usage
