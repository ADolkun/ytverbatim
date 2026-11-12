import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from ytverbatim.errors import BusyError
from ytverbatim.store import StageStore, digest, input_hash, lock


class Output(BaseModel):
    value: int


def test_digest_ignores_key_order_and_models_digest_as_their_json() -> None:
    assert digest({"a": 1, "b": [1, 2]}) == digest({"b": [1, 2], "a": 1})
    assert digest({"a": 1}) != digest({"a": 2})
    assert digest(Output(value=3)) == digest({"value": 3})


def test_a_matching_hash_skips_the_stage(tmp_path: Path) -> None:
    store = StageStore(tmp_path)
    key = input_hash("demo", 1, {"text": "abc"})
    calls: list[int] = []

    def compute() -> Output:
        calls.append(1)
        return Output(value=len(calls))

    first, reused_first = store.run("demo", 1, key, Output, compute)
    second, reused_second = store.run("demo", 1, key, Output, compute)
    assert (first, reused_first) == (Output(value=1), False)
    assert (second, reused_second) == (Output(value=1), True) and len(calls) == 1
    record = json.loads((tmp_path / ".stages" / "demo.json").read_text(encoding="utf-8"))
    assert record == {"stage_version": 1, "input_hash": key, "output": {"value": 1}}


def test_another_input_or_stage_version_recomputes(tmp_path: Path) -> None:
    store = StageStore(tmp_path)
    store.save("demo", 1, input_hash("demo", 1, {"text": "abc"}), Output(value=1))
    assert store.load("demo", 1, input_hash("demo", 1, {"text": "abd"}), Output) is None
    assert store.load("demo", 2, input_hash("demo", 1, {"text": "abc"}), Output) is None
    assert store.load("demo", 1, input_hash("demo", 1, {"text": "abc"}), Output) == Output(value=1)


def test_an_exception_mid_stage_leaves_no_file_and_keeps_the_previous_checkpoint(tmp_path: Path) -> None:
    store = StageStore(tmp_path)
    old = input_hash("demo", 1, {"n": 1})
    store.save("demo", 1, old, Output(value=1))

    def broken() -> Output:
        raise RuntimeError("killed mid-stage")

    with pytest.raises(RuntimeError):
        store.run("demo", 1, input_hash("demo", 1, {"n": 2}), Output, broken)
    with pytest.raises(RuntimeError):
        store.run("fresh", 1, input_hash("fresh", 1, {}), Output, broken)
    assert sorted(path.name for path in (tmp_path / ".stages").iterdir()) == ["demo.json"]
    assert store.load("demo", 1, old, Output) == Output(value=1)


def test_an_unreadable_or_outdated_checkpoint_is_recomputed(tmp_path: Path) -> None:
    store = StageStore(tmp_path)
    key = input_hash("demo", 1, {})
    store.root.mkdir()
    store.path("demo").write_text("{not json", encoding="utf-8")
    assert store.load("demo", 1, key, Output) is None
    store.path("demo").write_text(json.dumps({"stage_version": 1, "input_hash": key, "output": {"x": 1}}), "utf-8")
    assert store.load("demo", 1, key, Output) is None
    assert store.run("demo", 1, key, Output, lambda: Output(value=7)) == (Output(value=7), False)


def test_clear_drops_every_stage_and_work_dir(tmp_path: Path) -> None:
    store = StageStore(tmp_path)
    store.save("demo", 1, "k", Output(value=1))
    store.work_dir("demo").mkdir(parents=True)
    store.clear()
    assert not store.root.exists()


def test_lock_contention_raises_busy_at_once_and_releases(tmp_path: Path) -> None:
    vdir = tmp_path / "abcdefghijk"
    with lock(vdir), pytest.raises(BusyError, match="abcdefghijk"), lock(vdir):
        pass
    with lock(vdir):
        pass
