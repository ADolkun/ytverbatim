import json
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from ytverbatim.contract.models import SCHEMA_VERSION, ContextDocument
from ytverbatim.errors import ContractError

SCHEMA_PATH = Path(__file__).with_name("context.v1.schema.json")
_SUPPORTED_MAJOR = int(SCHEMA_VERSION.split(".")[0])
_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def export_schema() -> dict[str, Any]:
    return {"$schema": _JSON_SCHEMA_DIALECT, **ContextDocument.model_json_schema()}


def dump_document(doc: ContextDocument) -> str:
    """Field-declaration order, with free-form maps already key-sorted by the models: output is byte-stable."""
    return json.dumps(doc.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"


def load_document(source: Path | str) -> ContextDocument:
    """Read a document from a file (`Path`) or from JSON text (`str`).

    An unknown major version is refused. A newer minor version (`1.x`) loads with the fields this reader
    does not know dropped, which is why it is reported as `1.0` afterwards; re-saving it discards them.
    """
    text = source.read_text(encoding="utf-8") if isinstance(source, Path) else source
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractError("a context document must be a JSON object")
    data = cast("dict[str, Any]", raw)
    minor = _check_version(data.get("schema_version"))
    if minor > 0:
        schema = export_schema()
        data = {**_prune_unknown(data, schema, schema["$defs"]), "schema_version": SCHEMA_VERSION}
    try:
        return ContextDocument.model_validate(data)
    except ValidationError as exc:
        raise ContractError(f"invalid context document:\n{exc}") from exc


def _check_version(version: object) -> int:
    if not isinstance(version, str):
        raise ContractError(f"missing or non-string schema_version: {version!r}")
    parts = version.split(".")
    if len(parts) != 2 or not all(part.isascii() and part.isdigit() for part in parts):  # noqa: PLR2004
        raise ContractError(f"malformed schema_version {version!r}; expected '<major>.<minor>'")
    major, minor = int(parts[0]), int(parts[1])
    if major != _SUPPORTED_MAJOR:
        raise ContractError(
            f"unsupported context.json schema version {version!r}; this reader supports major {_SUPPORTED_MAJOR}"
        )
    return minor


def _prune_unknown(value: object, node: dict[str, Any], defs: dict[str, Any]) -> Any:
    """Drop object keys that the schema node does not declare, following $ref, anyOf and array items."""
    if "$ref" in node:
        return _prune_unknown(value, defs[node["$ref"].rsplit("/", 1)[-1]], defs)
    if isinstance(value, dict) and "properties" in node:
        properties: dict[str, Any] = node["properties"]
        return {key: _prune_unknown(item, properties[key], defs) for key, item in value.items() if key in properties}
    if isinstance(value, list) and "items" in node:
        return [_prune_unknown(item, node["items"], defs) for item in value]
    for branch in node.get("anyOf", ()):
        if _matches_shape(value, branch, defs):
            return _prune_unknown(value, branch, defs)
    return value


def _matches_shape(value: object, node: dict[str, Any], defs: dict[str, Any]) -> bool:
    if "$ref" in node:
        return _matches_shape(value, defs[node["$ref"].rsplit("/", 1)[-1]], defs)
    declared = node.get("type")
    if declared == "object":
        return isinstance(value, dict)
    if declared == "array":
        return isinstance(value, list)
    return False
