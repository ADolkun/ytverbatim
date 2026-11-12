import pytest

from ytverbatim.acquire import expand

LIST_URL = "https://www.youtube.com/playlist?list=PLexample"


def test_a_video_target_needs_no_request(fakes) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok())
    assert expand.expand_target(runner, "https://youtu.be/aaaaaaaaaaa") == ["aaaaaaaaaaa"]
    assert runner.calls == []


def test_playlist_ids_are_deduplicated_in_order_and_filtered(fakes) -> None:
    stdout = "aaaaaaaaaaa\nbbbbbbbbbbb\n\naaaaaaaaaaa\n  ccccccccccc \nUCexampletabplaylist00\nNA\n"
    runner = fakes.runner(lambda _args, _label: fakes.ok(stdout))
    assert expand.expand_target(runner, LIST_URL) == ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"]
    assert runner.calls[0].args == ["--flat-playlist", "--print", "%(id)s", LIST_URL]


def test_limit_becomes_playlist_end(fakes) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok("aaaaaaaaaaa\n"))
    expand.expand_target(runner, LIST_URL, limit=5)
    assert runner.calls[0].args[:5] == ["--flat-playlist", "--print", "%(id)s", "--playlist-end", "5"]


def test_channel_target_uses_the_channel_url(fakes) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok("aaaaaaaaaaa\n"))
    expand.expand_target(runner, "youtube.com/@example")
    assert runner.calls[0].args[-1] == "https://www.youtube.com/@example"


def test_unrecognised_target_raises(fakes) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok())
    with pytest.raises(ValueError, match="not a YouTube"):
        expand.expand_target(runner, "https://example.com/watch?v=aaaaaaaaaaa")


def test_expand_all_deduplicates_across_targets(fakes) -> None:
    outputs = {LIST_URL: "aaaaaaaaaaa\nbbbbbbbbbbb\n"}
    runner = fakes.runner(lambda args, _label: fakes.ok(outputs[args[-1]]))
    ids = expand.expand_targets(runner, ["bbbbbbbbbbb", LIST_URL, "https://youtu.be/ddddddddddd", LIST_URL])
    assert ids == ["bbbbbbbbbbb", "aaaaaaaaaaa", "ddddddddddd"]
    assert len(runner.calls) == 2
