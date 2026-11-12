import json
import os
import secrets
from pathlib import Path


def write_bytes(path: Path, data: bytes, *, mode: int | None = None) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    # O_EXCL plus an explicit fchmod: the file is never visible with looser permissions than requested.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666 if mode is None else mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            if mode is not None:
                os.fchmod(handle.fileno(), mode)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    write_bytes(path, text.encode("utf-8"), mode=mode)


def write_json(path: Path, obj: object, *, mode: int | None = None) -> None:
    write_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n", mode=mode)
