import itertools
from pathlib import Path

import pytest

from ytverbatim.acquire import captions
from ytverbatim.acquire.captions import CaptionKind, CaptionSegment, CaptionTrack
from ytverbatim.acquire.metadata import VideoMetadata
from ytverbatim.errors import YtDlpError
from ytverbatim.ytdlp.types import Signal, YtDlpResult

FIXTURES = Path(__file__).parent / "fixtures"
AUTO, MANUAL = CaptionKind.AUTO, CaptionKind.MANUAL


def arg_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def make_meta(*, subtitle_langs: tuple[str, ...] = (), auto: str | None = None) -> VideoMetadata:
    return VideoMetadata(
        video_id="aaaaaaaaaaa",
        title="t",
        channel="c",
        channel_id=None,
        upload_date=None,
        duration=600.0,
        description="",
        language="en",
        availability="public",
        live_status="not_live",
        chapters=(),
        subtitle_langs=subtitle_langs,
        auto_caption_lang=auto,
    )


def track(kind: CaptionKind, texts: list[str], *, step: float = 10.0) -> CaptionTrack:
    segments = tuple(CaptionSegment(i * step, (i + 1) * step, text) for i, text in enumerate(texts))
    return CaptionTrack("en", kind, "json3", segments)


# --- json3 ---


def test_auto_json3_builds_monotonic_word_timings() -> None:
    segments = captions.parse_json3(fixture_text("auto_en_orig.json3"), kind=AUTO)

    assert segments[0].text == "Hey everyone and thanks for jumping back"
    assert [w.text for w in segments[0].words][:3] == ["Hey", "everyone", "and"]
    assert segments[1].text == "into the cryptoverse."
    words = [w for segment in segments for w in segment.words]
    assert len(words) > 15
    assert all(a.start <= b.start for a, b in itertools.pairwise(words))
    assert all(w.start <= w.end for w in words)
    for segment in segments:
        assert all(a.end == b.start for a, b in itertools.pairwise(segment.words))
        assert segment.start == segment.words[0].start
        assert segment.end == segment.words[-1].end


def test_auto_json3_ends_a_line_where_the_newline_event_starts() -> None:
    first = captions.parse_json3(fixture_text("auto_en_orig.json3"), kind=AUTO)[0]
    assert (first.start, first.end) == (1.04, 2.79)
    assert first.words[1].start == pytest.approx(1.28)


def test_auto_json3_skips_the_header_and_newline_events() -> None:
    segments = captions.parse_json3(fixture_text("auto_en_orig.json3"), kind=AUTO)
    assert len(segments) == 4
    assert all("\n" not in segment.text and segment.text for segment in segments)


def test_manual_json3_has_segment_timings_only() -> None:
    segments = captions.parse_json3(fixture_text("manual_zh.json3"), kind=MANUAL)
    assert (segments[0].start, segments[0].end, segments[0].text) == (0.533, pytest.approx(1.546), "大家好 欢迎回来")
    assert [s.text for s in segments][1] == "上次视频发布后"
    assert all(s.words == () for s in segments)


def test_auto_json3_clamps_out_of_order_words() -> None:
    content = (
        '{"events": ['
        '{"tStartMs": 1000, "dDurationMs": 500, "segs": [{"utf8": "one"}, {"utf8": " two", "tOffsetMs": 400}]},'
        '{"tStartMs": 1200, "dDurationMs": 500, "segs": [{"utf8": "three"}]}]}'
    )
    words = [w for s in captions.parse_json3(content, kind=AUTO) for w in s.words]
    assert [w.start for w in words] == [1.0, 1.4, 1.4]
    assert all(w.end >= w.start for w in words)


@pytest.mark.parametrize("content", ["not json", "[]", '{"events": 3}'])
def test_invalid_json3_raises(content: str) -> None:
    with pytest.raises(ValueError, match=r"json3|events"):
        captions.parse_json3(content, kind=MANUAL)


# --- vtt ---

ROLLING_VTT = (
    "WEBVTT\nKind: captions\nLanguage: en\n\n"
    "00:00:00.030 --> 00:00:02.929 align:start position:0%\n"
    "hello<00:00:00.659><c> everyone</c><00:00:01.110><c> welcome</c>\n\n"
    "00:00:02.929 --> 00:00:02.939 align:start position:0%\n"
    "hello everyone welcome\n\n"
    "00:00:02.939 --> 00:00:05.299 align:start position:0%\n"
    "hello everyone welcome\n"
    "to<00:00:03.270><c> the</c><00:00:03.480><c> show</c>\n\n"
    "00:00:05.299 --> 00:00:05.309 align:start position:0%\n"
    "to the show\n\n"
    "00:00:05.309 --> 00:00:07.500 align:start position:0%\n"
    "to the show\n"
    "and<00:00:06.000><c> today</c><00:00:06.500><c> we</c><00:00:07.000><c> begin</c>\n"
)


def test_rolling_vtt_collapses_scrolling_repeats() -> None:
    segments = captions.parse_vtt(ROLLING_VTT)
    assert [s.text for s in segments] == ["hello everyone welcome", "to the show", "and today we begin"]
    assert [s.start for s in segments] == [0.03, 2.939, 5.309]


def test_repeated_line_in_a_later_cue_is_kept() -> None:
    content = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\nyes\n\n"
        "00:00:05.000 --> 00:00:06.000\nyes\n\n"
        "00:00:06.000 --> 00:00:07.000\nno\n"
    )
    assert [s.text for s in captions.parse_vtt(content)] == ["yes", "yes", "no"]


def test_dedup_can_be_turned_off_for_manual_tracks() -> None:
    content = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\ngo\n\n00:00:02.000 --> 00:00:03.000\ngo\n"
    assert [s.text for s in captions.parse_vtt(content, dedupe_rolling=False)] == ["go", "go"]
    assert [s.text for s in captions.parse_vtt(content)] == ["go"]


def test_vtt_unescapes_entities_and_joins_wrapped_lines() -> None:
    content = "WEBVTT\n\n00:01.000 --> 00:03.500\nTom &amp; Jerry\nare here&nbsp;now\r\n"
    segments = captions.parse_vtt(content)
    assert [(s.start, s.end, s.text) for s in segments] == [(1.0, 3.5, "Tom & Jerry are here now")]


def test_manual_vtt_matches_the_json3_track() -> None:
    from_vtt = captions.parse_vtt(fixture_text("manual_zh.vtt"), dedupe_rolling=False)
    from_json = captions.parse_json3(fixture_text("manual_zh.json3"), kind=MANUAL)
    assert [s.text for s in from_vtt][:5] == [s.text for s in from_json][:5]
    assert [s.start for s in from_vtt][:5] == pytest.approx([s.start for s in from_json][:5])


# --- quality ---


def test_dense_auto_track_is_ok() -> None:
    quality = captions.caption_quality(track(AUTO, ["a reasonably long caption line"] * 30), video_duration=300.0)
    assert quality.ok
    assert quality.coverage == 1.0
    assert quality.is_auto


def test_sparse_auto_track_fails_but_sparse_manual_passes() -> None:
    texts = ["a reasonably long caption line"] * 3
    assert not captions.caption_quality(track(AUTO, texts, step=30.0)).ok
    assert captions.caption_quality(track(MANUAL, texts, step=30.0)).ok


def test_empty_and_near_empty_tracks_are_degenerate() -> None:
    assert not captions.caption_quality(CaptionTrack("en", MANUAL, "json3", ())).ok
    assert not captions.caption_quality(track(MANUAL, [".", ".", "."], step=30.0)).ok


def test_partial_track_fails_the_coverage_gate() -> None:
    short = track(MANUAL, ["a reasonably long caption line"] * 6)
    assert captions.caption_quality(short, video_duration=3600.0).coverage == pytest.approx(0.017)
    assert not captions.caption_quality(short, video_duration=3600.0).ok
    assert captions.caption_quality(short, video_duration=3600.0, min_coverage=0.01).ok


def test_unknown_duration_applies_no_coverage_gate() -> None:
    quality = captions.caption_quality(track(MANUAL, ["a reasonably long caption line"] * 6))
    assert quality.coverage is None
    assert quality.ok


# --- language selection ---


def test_selection_takes_the_best_manual_variant_per_language_plus_original_auto() -> None:
    meta = make_meta(subtitle_langs=("en-GB", "zh-CN", "zh-TW", "ko"), auto="en-orig")
    assert captions.select_caption_langs(meta) == ("en-GB", "zh-CN", "en-orig")


def test_selection_skips_an_auto_track_in_an_unwanted_language() -> None:
    meta = make_meta(auto="ko-orig")
    assert captions.select_caption_langs(meta) == ()
    assert captions.select_caption_langs(meta, ("ko",)) == ("ko-orig",)


# --- fetch ---


def test_fetch_downloads_one_call_and_parses_both_tracks(fakes, tmp_path: Path) -> None:
    info = tmp_path / "info.json"
    captions_dir = tmp_path / "captions"

    def handler(args: list[str], _label: str) -> YtDlpResult:
        template = arg_after(args, "-o").removeprefix("subtitle:")
        Path(template.replace("%(ext)s", "zh-Hans.json3")).write_text(fixture_text("manual_zh.json3"), encoding="utf-8")
        Path(template.replace("%(ext)s", "en-orig.json3")).write_text(
            fixture_text("auto_en_orig.json3"), encoding="utf-8"
        )
        return fakes.ok()

    runner = fakes.runner(handler)
    cookies = tmp_path / "cookies.txt"
    meta = make_meta(subtitle_langs=("zh-Hans",), auto="en-orig")

    result = captions.fetch_captions(runner, info, captions_dir, meta, cookies=cookies)

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert (call.cookies, call.label) == (cookies, "captions")
    assert arg_after(call.args, "--load-info-json") == str(info)
    assert {"--skip-download", "--write-subs", "--write-auto-subs"} <= set(call.args)
    assert arg_after(call.args, "--sub-langs") == "zh-Hans,en-orig"
    assert arg_after(call.args, "--sub-format") == "json3/vtt/best"
    assert [(t.language, t.kind) for t in result.tracks] == [("zh-Hans", MANUAL), ("en-orig", AUTO)]
    assert result.tracks[1].has_word_timings
    assert not result.tracks[0].has_word_timings
    assert result.warnings == ()


def test_fetch_falls_back_to_vtt(fakes, tmp_path: Path) -> None:
    def handler(args: list[str], _label: str) -> YtDlpResult:
        template = arg_after(args, "-o").removeprefix("subtitle:")
        Path(template.replace("%(ext)s", "zh.vtt")).write_text(fixture_text("manual_zh.vtt"), encoding="utf-8")
        return fakes.ok()

    result = captions.fetch_captions(
        fakes.runner(handler), tmp_path / "i.json", tmp_path / "c", make_meta(subtitle_langs=("zh",))
    )
    assert [(t.language, t.source_format) for t in result.tracks] == [("zh", "vtt")]
    assert result.tracks[0].segments[0].text == "大家好 欢迎回来"


def test_fetch_without_wanted_tracks_makes_no_request(fakes, tmp_path: Path) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok())
    result = captions.fetch_captions(runner, tmp_path / "i.json", tmp_path / "c", make_meta())
    assert (result.tracks, result.warnings, runner.calls) == ((), (), [])


def test_missing_file_is_reported_as_a_warning(fakes, tmp_path: Path) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok())
    result = captions.fetch_captions(runner, tmp_path / "i.json", tmp_path / "c", make_meta(subtitle_langs=("en",)))
    assert result.tracks == ()
    assert "no json3 or vtt caption file for 'en'" in result.warnings[0]


@pytest.mark.parametrize("signal", [Signal.HTTP_429, Signal.PO_TOKEN_SUBS])
def test_a_rate_limited_subtitle_download_drops_captions_with_a_warning(fakes, tmp_path: Path, signal: Signal) -> None:
    captions_dir = tmp_path / "c"
    captions_dir.mkdir()
    (captions_dir / "captions.en.json3").write_text("{}", encoding="utf-8")
    runner = fakes.runner(fakes.raises(fakes.fail(signal)))

    result = captions.fetch_captions(runner, tmp_path / "i.json", captions_dir, make_meta(subtitle_langs=("en",)))

    assert result.tracks == ()
    assert len(result.warnings) == 1
    assert signal.value in result.warnings[0]
    assert list(captions_dir.iterdir()) == []


def test_a_429_warning_on_a_successful_exit_also_drops_captions(fakes, tmp_path: Path) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok(secondary=(Signal.HTTP_429,)))
    result = captions.fetch_captions(runner, tmp_path / "i.json", tmp_path / "c", make_meta(subtitle_langs=("en",)))
    assert result.tracks == ()
    assert "http_429" in result.warnings[0]


def test_other_failures_still_raise(fakes, tmp_path: Path) -> None:
    runner = fakes.runner(fakes.raises(fakes.fail(Signal.MEMBERS_ONLY)))
    with pytest.raises(YtDlpError):
        captions.fetch_captions(runner, tmp_path / "i.json", tmp_path / "c", make_meta(subtitle_langs=("en",)))
