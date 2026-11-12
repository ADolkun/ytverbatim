import pytest

from ytverbatim.ids import RefKind, YouTubeRef, classify, extract_video_id, is_video_id, video_url

VID = "abcDEF_1234"


@pytest.mark.parametrize(
    "text",
    [
        VID,
        f"  {VID}\n",
        f"https://www.youtube.com/watch?v={VID}",
        f"http://youtube.com/watch?v={VID}&t=42s",
        f"https://www.youtube.com/watch?feature=share&v={VID}",
        f"https://www.youtube.com/watch?v={VID}&list=PLabc123&index=2",
        f"youtube.com/watch?v={VID}",
        f"https://m.youtube.com/watch?v={VID}",
        f"https://music.youtube.com/watch?v={VID}&si=xyz",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?t=10",
        f"https://www.youtube.com/shorts/{VID}",
        f"https://www.youtube.com/embed/{VID}",
        f"https://www.youtube-nocookie.com/embed/{VID}",
        f"https://www.youtube.com/live/{VID}?feature=share",
        f"https://www.youtube.com/v/{VID}",
    ],
)
def test_video_id_from_every_form(text: str) -> None:
    assert extract_video_id(text) == VID
    assert classify(text) == YouTubeRef(RefKind.VIDEO, VID, video_url(VID))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "hello",
        "abcDEF_123",
        "abcDEF_12345",
        "https://www.youtube.com/watch?v=short",
        f"https://www.youtube.com/watch?v={VID}x",
        "https://www.youtube.com/watch",
        f"https://example.com/watch?v={VID}",
        f"https://notyoutube.com/watch?v={VID}",
        f"https://youtube.com.evil.example/watch?v={VID}",
        f"ftp://youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}/extra",
        "https://www.youtube.com/",
        "/tmp/some/local/file.mp4",
    ],
)
def test_invalid_inputs_are_rejected(text: str) -> None:
    assert extract_video_id(text) is None


def test_no_hash_fallback_for_non_youtube_input() -> None:
    assert classify("https://vimeo.com/12345678") is None


def test_playlist_urls() -> None:
    expected = YouTubeRef(RefKind.PLAYLIST, "PLabc_123-x", "https://www.youtube.com/playlist?list=PLabc_123-x")
    assert classify("https://www.youtube.com/playlist?list=PLabc_123-x") == expected
    assert classify("https://www.youtube.com/watch?list=PLabc_123-x") == expected
    assert classify("https://music.youtube.com/playlist?list=PLabc_123-x&si=1") == expected
    assert extract_video_id("https://www.youtube.com/playlist?list=PLabc_123-x") is None
    assert classify("https://www.youtube.com/playlist") is None


def test_members_only_playlist_is_a_playlist() -> None:
    ref = classify("https://www.youtube.com/playlist?list=UUMOabcdefgh")
    assert ref is not None
    assert ref.kind is RefKind.PLAYLIST


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("https://www.youtube.com/@some.handle", "@some.handle"),
        ("https://www.youtube.com/@some.handle/videos", "@some.handle"),
        ("https://m.youtube.com/@some-handle/streams", "@some-handle"),
        ("https://www.youtube.com/channel/UCabcdefghijklmnopqrstuv", "channel/UCabcdefghijklmnopqrstuv"),
        ("https://www.youtube.com/c/SomeName", "c/SomeName"),
        ("https://www.youtube.com/user/somename/videos", "user/somename"),
        ("youtube.com/@some.handle", "@some.handle"),
    ],
)
def test_channel_urls(text: str, key: str) -> None:
    assert classify(text) == YouTubeRef(RefKind.CHANNEL, key, f"https://www.youtube.com/{key}")


@pytest.mark.parametrize(
    "text", ["https://www.youtube.com/@", "https://www.youtube.com/channel", "https://youtu.be/@x"]
)
def test_malformed_channel_urls(text: str) -> None:
    assert classify(text) is None


def test_is_video_id() -> None:
    assert is_video_id(VID)
    assert not is_video_id("short")
    assert not is_video_id("has space11")
