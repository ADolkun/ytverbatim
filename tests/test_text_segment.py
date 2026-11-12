from collections.abc import Sequence

from ytverbatim.text import segment as seg
from ytverbatim.text.segment import Token


def chunk_level(text: str, language: str, *, seconds: float | None = None) -> list[str]:
    tokens = seg.pseudo_tokens(text, 0.0, seconds if seconds is not None else len(text) * 0.1, language=language)
    return [sentence.text for sentence in seg.segment_tokens(tokens, language)]


def timed(pieces: Sequence[str], *, step: float = 0.3, gaps: dict[int, float] | None = None) -> list[Token]:
    """One token per piece, each `step` long, with extra silence after the pieces named in `gaps`."""
    tokens: list[Token] = []
    clock = 0.0
    for index, piece in enumerate(pieces):
        tokens.append(Token(piece, clock, clock + step))
        clock += step + (gaps or {}).get(index, 0.0)
    return tokens


def test_zh_splits_at_terminators_and_keeps_them() -> None:
    text = "今天我们来聊聊财报。营收增长了百分之五十！你怎么看？我觉得不错；但是估值很高……所以要小心。"
    assert chunk_level(text, "zh") == [
        "今天我们来聊聊财报。",
        "营收增长了百分之五十！",
        "你怎么看？",
        "我觉得不错；",
        "但是估值很高……",
        "所以要小心。",
    ]


def test_zh_closing_quotes_stay_with_their_sentence() -> None:
    assert chunk_level("他说“我不去了。”然后就走了。", "zh") == ["他说“我不去了。”", "然后就走了。"]


def test_zh_without_punctuation_is_cut_at_pauses() -> None:
    pieces = list("今天我们聊聊财报") + list("营收增长很快") + list("估值也不低")
    tokens = timed(pieces, gaps={7: 0.7, 13: 0.9})
    sentences = seg.segment_tokens(tokens, "zh")
    assert [s.text for s in sentences] == ["今天我们聊聊财报", "营收增长很快", "估值也不低"]
    assert [round(s.start, 1) for s in sentences] == [0.0, 3.1, 5.8]


def test_zh_without_punctuation_or_pauses_is_cut_near_forty_characters() -> None:
    text = "营收增长很快" * 20
    parts = chunk_level(text, "zh")
    assert "".join(parts) == text
    assert all(30 <= len(part) <= 50 for part in parts[:-1]), [len(part) for part in parts]


def test_a_short_pause_does_not_split_punctuated_text() -> None:
    tokens = timed(["我们", "聊聊", "财报", "，", "然后", "看", "估值", "。"], gaps={2: 0.9})
    assert [s.text for s in seg.segment_tokens(tokens, "zh")] == ["我们聊聊财报，然后看估值。"]


ZH_FILLER = "简短的句子。" * 10
EN_FILLER = "Short one. " * 5


def test_overlong_zh_sentence_splits_at_the_comma_nearest_the_middle() -> None:
    first, middle, last = "甲" * 30, "乙" * 70, "丙" * 40
    parts = chunk_level(f"{ZH_FILLER}{first}，{middle}，{last}。", "zh")
    assert parts[-2:] == [f"{first}，{middle}，", f"{last}。"]


def test_overlong_zh_sentence_picks_the_comma_closest_to_the_centre() -> None:
    parts = chunk_level(f"{ZH_FILLER}{'甲' * 20}，{'乙' * 60}，{'丙' * 62}。", "zh")
    assert parts[-2:] == ["甲" * 20 + "，" + "乙" * 60 + "，", "丙" * 62 + "。"]


def test_a_zh_sentence_of_exactly_the_limit_is_left_alone() -> None:
    parts = chunk_level(f"{ZH_FILLER}{'甲' * 59}，{'乙' * 61}。", "zh")
    assert parts[-1] == "甲" * 59 + "，" + "乙" * 61 + "。"


def test_en_splits_at_terminator_space_capital_or_digit() -> None:
    assert chunk_level("It rose. Then it fell! Why? 3 reasons follow.", "en") == [
        "It rose.",
        "Then it fell!",
        "Why?",
        "3 reasons follow.",
    ]


def test_en_does_not_split_at_lowercase_continuations() -> None:
    assert chunk_level("It rose. and then fell.", "en") == ["It rose. and then fell."]


def test_en_abbreviations_do_not_end_sentences() -> None:
    text = "Dr. Smith met J. Smith in the U.S. last week. Prof. Lee agreed, e.g. Mr. Kim did."
    assert chunk_level(text, "en") == [
        "Dr. Smith met J. Smith in the U.S. last week.",
        "Prof. Lee agreed, e.g. Mr. Kim did.",
    ]


def test_en_never_splits_inside_a_number_or_a_dotted_ticker() -> None:
    text = "BRK.B rose 2.5% to $1,234.50 on Tuesday. AAPL fell 3. Then it recovered."
    assert chunk_level(text, "en") == [
        "BRK.B rose 2.5% to $1,234.50 on Tuesday.",
        "AAPL fell 3.",
        "Then it recovered.",
    ]


def test_en_no_before_a_number_is_an_abbreviation_but_no_alone_is_a_sentence_end() -> None:
    assert chunk_level("It is No. 5 on the list.", "en") == ["It is No. 5 on the list."]
    assert chunk_level("I said no. Then he left.", "en") == ["I said no.", "Then he left."]


def test_en_closing_quote_and_bracket_stay_with_the_sentence() -> None:
    assert chunk_level('He said "go." Then (quietly) left. Okay (really). Fine.', "en") == [
        'He said "go."',
        "Then (quietly) left.",
        "Okay (really).",
        "Fine.",
    ]


def test_en_unpunctuated_text_is_cut_near_twenty_words() -> None:
    text = " ".join(f"word{n}" for n in range(70))
    parts = chunk_level(text, "en")
    assert " ".join(parts) == text
    assert all(15 <= len(part.split()) <= 25 for part in parts[:-1]), [len(p.split()) for p in parts]


def test_en_unpunctuated_text_is_cut_at_pauses_first() -> None:
    words = [f" w{n}" for n in range(12)]
    sentences = seg.segment_tokens(timed(words, gaps={4: 0.8}), "en")
    assert [s.text for s in sentences] == ["w0 w1 w2 w3 w4", "w5 w6 w7 w8 w9 w10 w11"]


def test_overlong_en_sentence_splits_at_the_comma_nearest_the_middle() -> None:
    left = " ".join(f"A{n}" for n in range(10)) + ","
    right = " ".join(f"b{n}" for n in range(35)) + ","
    tail = " ".join(f"c{n}" for n in range(30)) + "."
    parts = chunk_level(f"{EN_FILLER}{left} {right} {tail}", "en")
    assert parts[-2:] == [f"{left} {right}", tail]


def test_overlong_sentence_without_a_comma_is_cut_in_its_middle() -> None:
    words = [f"W{n}" for n in range(90)]
    words[44] += ":"
    parts = chunk_level(EN_FILLER + " ".join(words) + ".", "en")
    long_parts = parts[5:]
    assert len(long_parts) == 2 and len(long_parts[0].split()) == 45


def test_a_bare_terminator_after_a_pause_joins_the_sentence_before_it() -> None:
    pieces = ["很"] * 90 + ["。"]
    sentences = seg.segment_tokens(timed(pieces, gaps={89: 0.9}), "zh")
    assert [s.text for s in sentences] == ["很" * 40, "很" * 50 + "。"]


def test_empty_input_has_no_sentences() -> None:
    assert seg.segment_tokens([], "en") == []
    assert seg.segment_tokens(timed([" ", ""]), "zh") == []


def test_sentence_times_come_from_its_tokens() -> None:
    (sentence,) = seg.segment_tokens(
        [Token(" Hello", 1.0, 1.4), Token(" there.", 1.5, 2.2)],
        "en",
    )
    assert (sentence.start, sentence.end, sentence.text) == (1.0, 2.2, "Hello there.")


def test_span_of_finds_the_characters_of_one_origin() -> None:
    tokens = [Token(" Hello", 0, 1, origin=0), Token(" there", 1, 2, origin=0), Token(" friend.", 2, 3, origin=1)]
    (sentence,) = seg.segment_tokens(tokens, "en")
    assert sentence.text == "Hello there friend."
    assert sentence.span_of(0) == (0, 11)
    assert sentence.span_of(1) == (12, 19)
    assert sentence.span_of(7) is None


def test_pseudo_token_times_are_proportional_to_characters() -> None:
    tokens = seg.pseudo_tokens("ab cdef", 10.0, 16.0, language="en")
    assert [t.text for t in tokens] == ["ab", " cdef"]
    assert [(t.start, t.end) for t in tokens] == [(10.0, 12.0), (12.0, 16.0)]
    assert not any(t.timed for t in tokens)


def test_zh_pseudo_tokens_keep_latin_words_and_numbers_whole() -> None:
    texts = [t.text for t in seg.pseudo_tokens("买NVDA涨了3.5%。", 0, 8, language="zh")]
    assert texts == ["买", "NVDA", "涨", "了", "3.5", "%", "。"]


def test_punctuation_density() -> None:
    assert seg.is_punctuated("今天聊财报。" * 5, "zh")
    assert not seg.is_punctuated("今天聊财报" * 40, "zh")
    assert seg.is_punctuated("It rose. Then it fell.", "en")
    assert not seg.is_punctuated(" ".join(["word"] * 80), "en")
    assert seg.is_punctuated(("word " * 39).strip() + ".", "en")


def spans(sentences: Sequence[tuple[float, float]]) -> tuple[list[float], list[float]]:
    return [s for s, _ in sentences], [e for _, e in sentences]


def test_paragraph_breaks_at_a_two_second_pause() -> None:
    starts, ends = spans([(0, 1), (1.5, 2.5), (4.5, 5), (5.2, 6)])
    assert seg.split_paragraphs(starts, ends, [None] * 4) == [(0, 1), (2, 3)]


def test_paragraph_pause_threshold_is_inclusive() -> None:
    assert seg.split_paragraphs([0, 3.0], [1.0, 4.0], [None, None]) == [(0, 0), (1, 1)]
    assert seg.split_paragraphs([0, 2.9], [1.0, 4.0], [None, None]) == [(0, 1)]


def test_paragraph_breaks_after_six_sentences() -> None:
    starts = [float(n) for n in range(13)]
    ends = [n + 0.5 for n in starts]
    assert seg.split_paragraphs(starts, ends, [None] * 13) == [(0, 5), (6, 11), (12, 12)]


def test_paragraph_breaks_after_ninety_seconds() -> None:
    starts = [0.0, 40.0, 80.0, 95.0, 100.0]
    ends = [39.0, 79.0, 94.0, 99.0, 104.0]
    assert seg.split_paragraphs(starts, ends, [None] * 5) == [(0, 2), (3, 4)]


def test_paragraph_never_crosses_a_chapter() -> None:
    starts = [0.0, 1.0, 2.0, 3.0]
    ends = [0.5, 1.5, 2.5, 3.5]
    assert seg.split_paragraphs(starts, ends, [0, 0, 1, 1]) == [(0, 1), (2, 3)]


def test_no_sentences_means_no_paragraphs() -> None:
    assert seg.split_paragraphs([], [], []) == []


def test_chapter_indexes_use_the_last_chapter_that_has_begun() -> None:
    assert seg.chapter_indexes([0.0, 49.9, 50.0, 120.0], [0.0, 50.0, 100.0]) == [0, 0, 1, 2]
    assert seg.chapter_indexes([1.0, 9.0], [5.0, 20.0]) == [0, 0]


def test_a_long_unpunctuated_stretch_inside_punctuated_text_is_cut_too() -> None:
    stretch = " ".join(f"Word{n}" for n in range(60))
    parts = chunk_level(f"{EN_FILLER}{stretch}. Then it ended.", "en")
    assert parts[:5] == ["Short one."] * 5
    assert parts[-1] == "Then it ended."
    assert all(len(part.split()) <= 25 for part in parts[5:-1] + parts[-2:-1])
    assert len(parts) > 8


def test_a_stray_period_before_a_lowercase_word_does_not_hide_an_unpunctuated_stretch() -> None:
    stretch = "hello everyone see example.com today. " + " ".join(f"word{n}" for n in range(60))
    parts = chunk_level(f"{EN_FILLER}{stretch}", "en")
    assert all(len(part.split()) <= 25 for part in parts[5:])
