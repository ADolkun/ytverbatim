import pytest

from ytverbatim.text.clean import Fillers, Script, normalize, to_simplified


def clean_en(text: str) -> str:
    return normalize(text, "en", script="zh-Hans", fillers="remove").text


def clean_zh(text: str, *, fillers: Fillers = "remove", script: Script = "zh-Hans") -> str:
    return normalize(text, "zh", script=script, fillers=fillers).text


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("so um we bought uh more", "so we bought more"),
        ("Um, the market rose.", "the market rose."),
        ("hmm I think, um, that it held", "I think, that it held"),
        ("errr the rate er moved", "the rate moved"),
    ],
)
def test_english_hesitations_go(before: str, after: str) -> None:
    assert clean_en(before) == after


def test_the_word_err_and_all_caps_acronyms_stay() -> None:
    assert clean_en("To err is human.") == "To err is human."
    assert clean_en("UM and ER are hospital units.") == "UM and ER are hospital units."
    assert clean_en("Umbrella sales erupted.") == "Umbrella sales erupted."


def test_chinese_fillers_go_only_at_boundaries_and_repeats_collapse() -> None:
    assert clean_zh("嗯，今天聊聊估值") == "今天聊聊估值"
    assert clean_zh("我们看，啊，这个数据") == "我们看，这个数据"
    assert clean_zh("就是说就是说市场很好") == "就是说市场很好"
    assert clean_zh("然后然后我们买入") == "然后我们买入"
    assert clean_zh("好啊我们走吧") == "好啊我们走吧", "a particle inside a clause is speech, not a filler"
    assert clean_zh("这个那个都看", fillers="remove") == "这个那个都看", "demonstratives are never fillers"


def test_fillers_are_kept_unless_asked_and_punctuation_is_always_tidied() -> None:
    assert normalize("um, so.. we waited,, then", "en", script="zh-Hans", fillers="keep").text == (
        "um, so. we waited, then"
    )
    assert clean_zh("，，好的。。", fillers="keep") == "好的。"


def test_simplified_script_is_the_default_and_keep_leaves_traditional_text() -> None:
    assert clean_zh("這個軟體很好", fillers="keep") == "这个软体很好"
    assert clean_zh("這個軟體很好", fillers="keep", script="keep") == "這個軟體很好"
    assert to_simplified("Apple 這個") == "Apple 这个"
    assert normalize("這個", "en", script="zh-Hans", fillers="keep").text == "這個", "only zh text is converted"


def test_steps_chain_from_the_input_to_the_output_one_replacement_at_a_time() -> None:
    text = "um the GAAP number um rose"
    result = normalize(text, "en", script="zh-Hans", fillers="remove")
    assert result.text == "the GAAP number rose"
    assert result.steps[0][0] == text and result.steps[-1][1] == result.text
    assert all(a[1] == b[0] for a, b in zip(result.steps, result.steps[1:], strict=False))


def test_a_text_of_only_fillers_is_kept_as_it_was() -> None:
    result = normalize("um uh", "en", script="zh-Hans", fillers="remove")
    assert result.text == "um uh" and result.steps == ()
