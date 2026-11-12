import json
import stat
from pathlib import Path

import pytest

from ytverbatim import atomic


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_write_bytes_text_and_json(tmp_path: Path) -> None:
    atomic.write_bytes(tmp_path / "a.bin", b"\x00\x01")
    atomic.write_text(tmp_path / "b.txt", "你好\n")
    atomic.write_json(tmp_path / "c.json", {"k": "你好", "n": [1, 2]})
    assert (tmp_path / "a.bin").read_bytes() == b"\x00\x01"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "你好\n"
    assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8")) == {"k": "你好", "n": [1, 2]}
    assert "你好" in (tmp_path / "c.json").read_text(encoding="utf-8")


def test_mode_is_honoured_even_under_a_restrictive_umask(tmp_path: Path) -> None:
    atomic.write_text(tmp_path / "secret", "x", mode=0o600)
    atomic.write_text(tmp_path / "public", "x", mode=0o644)
    assert _mode(tmp_path / "secret") == 0o600
    assert _mode(tmp_path / "public") == 0o644


def test_mode_applies_when_replacing_a_looser_file(tmp_path: Path) -> None:
    target = tmp_path / "f"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o644)
    atomic.write_text(target, "new", mode=0o600)
    assert target.read_text(encoding="utf-8") == "new"
    assert _mode(target) == 0o600


def test_overwrite_leaves_no_temp_files(tmp_path: Path) -> None:
    atomic.write_text(tmp_path / "f", "one")
    atomic.write_text(tmp_path / "f", "two")
    assert [p.name for p in tmp_path.iterdir()] == ["f"]
    assert (tmp_path / "f").read_text(encoding="utf-8") == "two"


def test_failure_keeps_previous_content_and_leaves_no_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "f.json"
    atomic.write_json(target, {"v": 1})
    with pytest.raises(TypeError):
        atomic.write_json(target, {"v": object()})
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["f.json"]


def test_failure_in_replace_cleans_up_the_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(src: object, dst: object) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(OSError, match="disk on fire"):
        atomic.write_text(tmp_path / "f", "x", mode=0o600)
    assert list(tmp_path.iterdir()) == []
