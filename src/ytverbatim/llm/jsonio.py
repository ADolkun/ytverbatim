import json
from dataclasses import replace

from pydantic import BaseModel, ValidationError

from ytverbatim.errors import LLMOutputError
from ytverbatim.llm.protocol import LLMClient, LLMRequest

_ECHO_LIMIT = 2000


def complete_json[M: BaseModel](client: LLMClient, request: LLMRequest, model_type: type[M]) -> M:
    """Complete and validate one JSON object; one corrective retry, then `LLMOutputError`."""
    schema = request.json_schema if request.json_schema is not None else model_type.model_json_schema()
    first = replace(request, system=_with_schema(request.system, schema), json_schema=schema)
    reply = client.complete(first).text
    try:
        return _parse(reply, model_type)
    except ValueError as exc:
        error = str(exc)
    retry = replace(first, user=_correction(request.user, reply, error))
    reply = client.complete(retry).text
    try:
        return _parse(reply, model_type)
    except ValueError as exc:
        raise LLMOutputError(f"{client.id} returned invalid JSON twice: {exc}") from exc


def extract_json_object(text: str) -> dict[str, object]:
    """First decodable JSON object in `text`; fences and surrounding prose are skipped."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, dict):
                return as_object(value)
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in the reply")


def _parse[M: BaseModel](text: str, model_type: type[M]) -> M:
    try:
        return model_type.model_validate(extract_json_object(text))
    except ValidationError as exc:
        raise ValueError(f"the JSON object does not match the schema: {exc}") from exc


def _with_schema(system: str, schema: object) -> str:
    instruction = (
        "Reply with exactly one JSON object that validates against this JSON Schema, and nothing else:\n"
        + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    )
    return f"{system}\n\n{instruction}" if system else instruction


def _correction(user: str, reply: str, error: str) -> str:
    echoed = reply if len(reply) <= _ECHO_LIMIT else reply[:_ECHO_LIMIT] + " [truncated]"
    return (
        f"{user}\n\n<previous_reply>\n{echoed}\n</previous_reply>\n"
        f"That reply was rejected: {error}\n"
        "Reply again with only the corrected JSON object."
    )


def as_object(value: object) -> dict[str, object]:
    """`value` as a string-keyed dict, or empty when it is not a JSON object."""
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}
