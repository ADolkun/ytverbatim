from ytverbatim.acquire import creator_note

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


def test_note_comes_from_the_info_comments() -> None:
    assert creator_note.creator_note_from_info({"comments": [PINNED]}) is not None
    assert creator_note.creator_note_from_info({"comments": None}) is None
    assert creator_note.creator_note_from_info({}) is None


def test_comment_args_request_a_bounded_page_without_replies() -> None:
    assert creator_note.comment_args(10) == (
        "--get-comments",
        "--extractor-args",
        "youtube:comment_sort=top;max_comments=10,all,0,0",
    )
