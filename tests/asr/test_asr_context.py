from ytverbatim.acquire.metadata import Chapter, VideoMetadata
from ytverbatim.asr.context import MAX_CONTEXT_CHARS, build_context, context_from_metadata


def joined_length(prompt: str, hotwords: tuple[str, ...]) -> int:
    return len(prompt) + (len("\n") + len(", ".join(hotwords)) if hotwords else 0)


def test_urls_are_stripped_from_every_field() -> None:
    context = build_context(
        "Weekly update www.example.test/x",
        "Join at https://example.test/join?ref=1 or HTTP://EXAMPLE.TEST/a.\nFollow https://example.test/@fake",
        ["Part 1 https://example.test/p1"],
    )

    assert "example.test" not in context.system_prompt
    assert "http" not in context.system_prompt.lower()
    assert "Weekly update" in context.system_prompt
    assert "Part 1" in context.system_prompt


def test_prompt_orders_title_then_chapters_then_description() -> None:
    context = build_context("Title line", "Description body", ["Chapter one", "Chapter two"])

    assert context.system_prompt == "Title line\nChapter one\nChapter two\nDescription body"


def test_description_timestamps_and_blank_runs_are_collapsed() -> None:
    context = build_context("T", "0:00 Intro\n\n\n  12:34   Outlook \n1:02:03 End", [])

    assert context.system_prompt == "T\nIntro\nOutlook\nEnd"


def test_total_biasing_text_never_exceeds_the_cap() -> None:
    long_description = " ".join(f"word{i}" for i in range(500))
    context = build_context("Fed meeting SPY QQQ", long_description, ["NVDA earnings"])

    assert context.hotwords
    assert joined_length(context.system_prompt, context.hotwords) <= MAX_CONTEXT_CHARS
    assert context.system_prompt.startswith("Fed meeting SPY QQQ\nNVDA earnings\nword0")
    assert not context.system_prompt.endswith(" ")


def test_truncation_does_not_cut_a_latin_word_in_half() -> None:
    context = build_context("T", "alpha " * 200, [])

    assert context.system_prompt.split()[-1] == "alpha"
    assert len(context.system_prompt) <= MAX_CONTEXT_CHARS


def test_cjk_text_is_cut_at_the_limit_without_hunting_for_a_space() -> None:
    context = build_context("美股 SPY", "涨" * 1000, [])

    assert context.hotwords == ("SPY",)
    assert joined_length(context.system_prompt, context.hotwords) == MAX_CONTEXT_CHARS


def test_tickers_in_both_forms_become_hotwords_in_first_seen_order() -> None:
    context = build_context(
        "美股 科技股！XLP重要位置，SMH、SOXX完成修复！$nvda 和 $TSLA",
        "SPY QQQ and US Fed. Not NASDAQ100, not ABCDEFG, not a$b, not x$ABC9.",
        [],
    )

    assert context.hotwords == ("XLP", "SMH", "SOXX", "NVDA", "TSLA", "SPY", "QQQ", "US")


def test_latin_names_and_mixed_case_terms_are_hotwords_but_plain_words_are_not() -> None:
    context = build_context(
        "Bitcoin outlook with Benjamin Cowen",
        "We cover OpenAI, iPhone and the Federal Reserve. Sentences start with Words like these.",
        ["Cycle lows analysis"],
    )

    assert context.hotwords == ("Benjamin Cowen", "OpenAI", "iPhone", "Federal Reserve")


def test_hotwords_are_deduplicated_ignoring_case_and_capped_by_budget() -> None:
    assert build_context("SPY", "$spy $Spy SPY", []).hotwords == ("SPY",)

    many = " ".join(f"{chr(65 + i // 26)}{chr(65 + i % 26)}X" for i in range(200))
    hotwords = build_context(many, "", []).hotwords
    assert 0 < len(hotwords) < 200
    assert len(", ".join(hotwords)) <= 200


def test_the_same_metadata_always_gives_the_same_context() -> None:
    args = ("SPY 和 QQQ", "Fed Reserve notes https://example.test", ["Part SPY"])

    assert build_context(*args) == build_context(*args)


def test_empty_metadata_gives_an_empty_context() -> None:
    context = build_context("", "", [])

    assert context.system_prompt == ""
    assert context.hotwords == ()


def test_context_from_metadata_reads_title_chapters_and_description() -> None:
    metadata = VideoMetadata(
        video_id="AAAAAAAAAAA",
        title="Market recap $SPY",
        channel="Example Channel",
        channel_id=None,
        upload_date=None,
        duration=600.0,
        description="Notes at https://example.test",
        language=None,
        availability=None,
        live_status=None,
        chapters=(Chapter("Opening", 0.0, 60.0), Chapter("Risks", 60.0, 600.0)),
        subtitle_langs=(),
        auto_caption_lang=None,
    )

    context = context_from_metadata(metadata)

    assert context.system_prompt == "Market recap $SPY\nOpening\nRisks\nNotes at"
    assert context.hotwords == ("SPY",)
