import json
import logging

import pytest

from ytverbatim.acquire import creator_note
from ytverbatim.errors import YtDlpError
from ytverbatim.ytdlp.types import Signal

PINNED = {
    "is_pinned": True,
    "author_is_uploader": True,
    "text": "  Correction: the figure is 12%.  ",
    "author": "@example",
    "like_count": 7,
    "_time_text": "1 day ago",
}


def test_pinned_creator_comment_is_selected() -> None:
    other = {"is_pinned": False, "author_is_uploader": True, "text": "not pinned"}
    viewer = {"is_pinned": True, "author_is_uploader": False, "text": "pinned by someone else"}
    note = creator_note.select_pinned_creator_comment([other, viewer, PINNED])
    assert note == creator_note.CreatorNote("Correction: the figure is 12%.", "@example", 7, "1 day ago")


def test_blank_pinned_comment_gives_none() -> None:
    assert creator_note.select_pinned_creator_comment([{**PINNED, "text": "   "}]) is None


def test_no_matching_comment_gives_none() -> None:
    assert creator_note.select_pinned_creator_comment(["junk", {"text": "plain"}]) is None


def test_fetch_requests_a_bounded_comment_page(fakes) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok(json.dumps({"comments": [PINNED]})))
    note = creator_note.fetch_creator_note(runner, "aaaaaaaaaaa", max_comments=10)
    assert note is not None
    args = runner.calls[0].args
    assert "--write-comments" in args
    assert "youtube:comment_sort=top;max_comments=10,10,0,0" in args
    assert runner.calls[0].label == "creator-note"


def test_ordinary_failure_is_skipped_with_a_warning(fakes, caplog: pytest.LogCaptureFixture) -> None:
    runner = fakes.runner(fakes.raises(fakes.fail(Signal.TRANSIENT)))
    with caplog.at_level(logging.WARNING):
        assert creator_note.fetch_creator_note(runner, "aaaaaaaaaaa") is None
    assert "creator note skipped" in caplog.text


def test_unparseable_output_is_skipped(fakes) -> None:
    runner = fakes.runner(lambda _args, _label: fakes.ok("not json"))
    assert creator_note.fetch_creator_note(runner, "aaaaaaaaaaa") is None


@pytest.mark.parametrize("signal", [Signal.BOT_CHECK, Signal.RATE_LIMITED, Signal.CAPTCHA, Signal.IP_BLOCKED])
def test_push_back_signals_propagate(fakes, signal: Signal) -> None:
    runner = fakes.runner(fakes.raises(fakes.fail(Signal.UNAVAILABLE, signal)))
    with pytest.raises(YtDlpError):
        creator_note.fetch_creator_note(runner, "aaaaaaaaaaa")
