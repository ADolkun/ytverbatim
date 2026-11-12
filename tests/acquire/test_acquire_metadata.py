import json
import stat
from pathlib import Path

import pytest

from ytverbatim.acquire import metadata
from ytverbatim.acquire.creator_note import CreatorNote
from ytverbatim.errors import UnavailableError, YtDlpError
from ytverbatim.ytdlp.types import Signal

VIDEO_ID = "aaaaaaaaaaa"


def info_dict(**overrides: object) -> dict[str, object]:
    info: dict[str, object] = {
        "id": VIDEO_ID,
        "title": "A synthetic talk",
        "channel": "Example Channel",
        "uploader": "example-uploader",
        "channel_id": "UCexample000000000000000",
        "upload_date": "20260101",
        "duration": 600,
        "description": "Synthetic description.",
        "language": "en",
        "availability": "public",
        "live_status": "not_live",
        "chapters": [
            {"start_time": 0.0, "end_time": 120.0, "title": "Intro"},
            {"start_time": 120.0, "end_time": 600.0, "title": "Main"},
        ],
        "subtitles": {"zh-Hans": [{"ext": "json3"}], "en-US": [{"ext": "vtt"}]},
        "automatic_captions": {
            "en-orig": [{"ext": "json3"}],
            "en": [{"ext": "json3"}],
            "de": [{"ext": "json3"}],
            "zh-Hans": [{"ext": "json3"}],
        },
        "formats": [{"url": "https://example.invalid/stream"}],
    }
    info.update(overrides)
    return info


def test_trim_keeps_chapters_and_only_the_original_auto_track() -> None:
    meta = metadata.trim_info(info_dict())
    assert meta.video_id == VIDEO_ID
    assert meta.channel_id == "UCexample000000000000000"
    assert meta.availability == "public"
    assert meta.live_status == "not_live"
    assert meta.duration == 600.0
    assert meta.subtitle_langs == ("en-US", "zh-Hans")
    assert meta.auto_caption_lang == "en-orig"
    assert [(c.title, c.start, c.end) for c in meta.chapters] == [("Intro", 0.0, 120.0), ("Main", 120.0, 600.0)]


def test_auto_track_falls_back_to_the_key_equal_to_language() -> None:
    info = info_dict(automatic_captions={"en": [{}], "de": [{}]})
    assert metadata.trim_info(info).auto_caption_lang == "en"


def test_auto_track_is_none_when_nothing_matches() -> None:
    info = info_dict(language="fr", automatic_captions={"en": [{}]})
    assert metadata.trim_info(info).auto_caption_lang is None


def test_orig_key_matching_the_language_wins() -> None:
    info = info_dict(language="zh", automatic_captions={"en-orig": [{}], "zh-orig": [{}]})
    assert metadata.trim_info(info).auto_caption_lang == "zh-orig"


def test_missing_fields_become_empty_values() -> None:
    meta = metadata.trim_info({"id": VIDEO_ID, "chapters": None, "subtitles": None, "automatic_captions": None})
    assert (meta.title, meta.channel, meta.description) == ("", "", "")
    assert meta.chapters == ()
    assert meta.subtitle_langs == ()
    assert meta.duration is None


def test_malformed_chapters_are_skipped() -> None:
    info = info_dict(chapters=[{"title": "no times"}, {"start_time": 1, "end_time": 2, "title": "kept"}, "junk"])
    assert [c.title for c in metadata.trim_info(info).chapters] == ["kept"]


def test_to_dict_is_json_serializable() -> None:
    payload = metadata.trim_info(info_dict()).to_dict()
    assert json.loads(json.dumps(payload))["chapters"][0]["title"] == "Intro"


def test_fetch_info_runs_one_extraction_and_saves_the_raw_json(fakes, tmp_path: Path) -> None:
    raw = json.dumps(info_dict())
    runner = fakes.runner(lambda _args, _label: fakes.ok(raw))
    cookies = tmp_path / "cookies.txt"

    meta = metadata.fetch_info(runner, VIDEO_ID, tmp_path / "stage", cookies=cookies)

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call.args[:2] == ["-J", "--no-playlist"]
    assert call.args[-1] == f"https://www.youtube.com/watch?v={VIDEO_ID}"
    assert (call.cookies, call.label) == (cookies, "metadata")
    saved = metadata.info_path(tmp_path / "stage")
    assert saved.read_text(encoding="utf-8") == raw
    assert stat.S_IMODE(saved.stat().st_mode) == 0o600
    assert metadata.load_metadata(tmp_path / "stage") == meta
    assert "--get-comments" not in call.args
    assert meta.creator_note is None


PINNED = {"is_pinned": True, "author_is_uploader": True, "text": "Correction: 12%.", "author": "@example"}


def test_creator_note_rides_on_the_same_extraction(fakes, tmp_path: Path) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok(json.dumps(info_dict(comments=[PINNED]))))

    meta = metadata.fetch_info(runner, VIDEO_ID, tmp_path / "stage", creator_note=True, max_comments=5)

    [call] = runner.calls
    assert call.args[:2] == ["-J", "--no-playlist"]
    assert call.args[call.args.index("--extractor-args") + 1] == "youtube:comment_sort=top;max_comments=5,all,0,0"
    assert "--get-comments" in call.args
    assert call.args[-1] == f"https://www.youtube.com/watch?v={VIDEO_ID}"
    assert meta.creator_note == CreatorNote("Correction: 12%.", "@example", None, "")
    assert metadata.load_metadata(tmp_path / "stage").creator_note == meta.creator_note


@pytest.mark.parametrize("signal", [Signal.BOT_CHECK, Signal.RATE_LIMITED, Signal.CAPTCHA, Signal.IP_BLOCKED])
def test_push_back_signals_propagate_when_the_note_is_requested(fakes, tmp_path: Path, signal: Signal) -> None:
    runner = fakes.runner(fakes.raises(fakes.fail(signal)))
    with pytest.raises(YtDlpError) as raised:
        metadata.fetch_info(runner, VIDEO_ID, tmp_path / "stage", creator_note=True)
    assert raised.value.primary is signal


@pytest.mark.parametrize(("live_status", "kind"), [("is_live", "live"), ("is_upcoming", "upcoming")])
def test_live_and_upcoming_videos_are_unavailable(fakes, tmp_path: Path, live_status: str, kind: str) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok(json.dumps(info_dict(live_status=live_status))))
    with pytest.raises(UnavailableError) as raised:
        metadata.fetch_info(runner, VIDEO_ID, tmp_path / "stage")
    assert raised.value.kind == kind
    assert not metadata.info_path(tmp_path / "stage").exists()


def test_info_for_a_different_video_is_rejected(fakes, tmp_path: Path) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok(json.dumps(info_dict(id="bbbbbbbbbbb"))))
    with pytest.raises(YtDlpError):
        metadata.fetch_info(runner, VIDEO_ID, tmp_path / "stage")


@pytest.mark.parametrize("stdout", ["not json", "[1, 2]", ""])
def test_non_object_output_raises(fakes, tmp_path: Path, stdout: str) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok(stdout))
    with pytest.raises(YtDlpError):
        metadata.fetch_info(runner, VIDEO_ID, tmp_path / "stage")
