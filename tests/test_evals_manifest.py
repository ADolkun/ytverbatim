import json
from pathlib import Path
from typing import Any

import pytest

from evals.manifest import EVAL_DIR_ENV, eval_dir, load_manifest
from ytverbatim.errors import ConfigError


def video(video_id: str, language: str, access: str = "public", **extra: Any) -> dict[str, Any]:
    return {
        "id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "channel": "Example Channel",
        "title": "Example title",
        "duration_s": 600,
        "language": language,
        "access": access,
        "audio": f"audio/{video_id}.wav",
        "audio_sha256": "0" * 64,
        "audio_format": "wav pcm_s16le 16000 Hz mono",
        "audio_duration_s": 600.5,
        "upload_date": "20250101",
        "chapters": f"meta/{video_id}.chapters.json",
        "chapter_count": 0,
        "captions": {
            "listed": [{"kind": "auto", "lang": "en", "formats": ["json3", "vtt"]}],
            "listed_note": "synthetic",
            "obtained": [
                {
                    "kind": "auto",
                    "lang": "en",
                    "format": "json3",
                    "path": f"captions/{video_id}.json3",
                    "bytes": 10,
                    "word_timing": True,
                },
                {"kind": "manual", "lang": "zh", "format": "vtt", "path": f"captions/{video_id}.vtt", "bytes": 5},
            ],
        },
        "source": "downloaded",
        "notes": "",
        "meta": f"meta/{video_id}.info.json",
        **extra,
    }


def write_manifest(directory: Path, videos: list[dict[str, Any]]) -> None:
    payload = {"created": "2025-01-01", "audio_format": "wav", "videos": videos, "unknown_top_level": 1}
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_loads_every_field(tmp_path: Path) -> None:
    write_manifest(tmp_path, [video("vid00000001", "zh+en", "members", future_field=True)])
    manifest = load_manifest(tmp_path)
    (loaded,) = manifest.videos
    assert loaded.id == "vid00000001"
    assert loaded.access == "members"
    assert loaded.duration_s == 600
    assert loaded.audio_duration_s == 600.5
    assert loaded.captions.listed[0].formats == ("json3", "vtt")
    assert [(c.kind, c.format, c.word_timing) for c in loaded.captions.obtained] == [
        ("auto", "json3", True),
        ("manual", "vtt", None),
    ]


def test_mixed_language_scores_as_chinese(tmp_path: Path) -> None:
    write_manifest(tmp_path, [video("vid00000001", "zh+en"), video("vid00000002", "en"), video("vid00000003", "zh")])
    languages = [v.view_language for v in load_manifest(tmp_path).videos]
    assert languages == ["zh", "en", "zh"]


def test_labels_number_each_scoring_language_in_manifest_order(tmp_path: Path) -> None:
    write_manifest(
        tmp_path,
        [
            video("vid00000001", "zh+en"),
            video("vid00000002", "en"),
            video("vid00000003", "zh+en"),
            video("vid00000004", "en"),
        ],
    )
    assert load_manifest(tmp_path).labels() == {
        "vid00000001": "zh-01",
        "vid00000002": "en-01",
        "vid00000003": "zh-02",
        "vid00000004": "en-02",
    }


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    write_manifest(tmp_path, [video("vid00000001", "en"), video("vid00000001", "en")])
    with pytest.raises(ConfigError, match="more than once"):
        load_manifest(tmp_path)


def test_unknown_language_is_rejected(tmp_path: Path) -> None:
    write_manifest(tmp_path, [video("vid00000001", "fr")])
    with pytest.raises(ValueError, match="language"):
        load_manifest(tmp_path)


def test_eval_dir_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EVAL_DIR_ENV, raising=False)
    with pytest.raises(ConfigError, match=EVAL_DIR_ENV):
        eval_dir()
    with pytest.raises(ConfigError, match=EVAL_DIR_ENV):
        load_manifest()


def test_eval_dir_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_manifest(tmp_path, [video("vid00000001", "en")])
    monkeypatch.setenv(EVAL_DIR_ENV, str(tmp_path))
    assert eval_dir() == tmp_path
    assert [v.id for v in load_manifest().videos] == ["vid00000001"]
