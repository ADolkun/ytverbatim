import pytest

from evals.normalize import canonical, join_units, normalize, split_units, to_units


def test_traditional_characters_become_simplified() -> None:
    assert normalize("金額漲跌", "zh", "raw") == "金额涨跌"


def test_nfkc_folds_fullwidth_forms() -> None:
    assert normalize("ＳＰＹ３．５％", "zh", "raw") == "SPY3.5%"


def test_punctuation_and_symbols_are_stripped_but_not_money_or_percent() -> None:
    assert normalize("好，（行）！ a+b ~ $5 ¥6 €7 £8 9%", "zh", "raw") == "好行a b $5 ¥6 €7 £8 9%"


def test_separators_survive_only_between_digits() -> None:
    assert normalize("1,000.5, then 2.", "en", "raw") == "1,000.5 then 2"


def test_apostrophes_and_letter_dots_are_dropped() -> None:
    assert normalize("U.S. don't well-known", "en", "raw") == "US dont well known"


def test_raw_view_preserves_latin_case() -> None:
    assert normalize("NVDA nvda", "zh", "raw") == "NVDA nvda"
    assert to_units("NVDA nvda", "en", "raw") == ["NVDA", "nvda"]


def test_itn_view_casefolds() -> None:
    assert to_units("NVDA nvda", "zh", "itn") == ["nvda", "nvda"]


def test_chinese_fillers_are_removed_only_when_standalone() -> None:
    assert normalize("嗯，我们 啊 好啊。额度 呃", "zh", "raw") == "我们好啊额度"
    assert normalize("嗯哼 嗯嗯 对", "zh", "raw") == "对"


def test_english_fillers_are_removed_in_the_raw_view() -> None:
    assert to_units("Um, I think uh Hmm mhm it works", "en", "raw") == ["I", "think", "it", "works"]


def test_views_differ_on_chinese_quantities() -> None:
    text = "一百五十期和150期"
    assert normalize(text, "zh", "raw") == "一百五十期和150期"
    assert normalize(text, "zh", "itn") == "150期和150期"


def test_itn_leaves_non_quantity_numerals_alone() -> None:
    assert normalize("一样一直", "zh", "itn") == "一样一直"


def test_itn_view_runs_english_through_the_whisper_normalizer() -> None:
    assert to_units("I have twenty five dollars, don't I?", "en", "itn") == ["i", "have", "$25", "do", "not", "i"]


def test_english_itn_and_raw_agree_on_dotted_acronyms() -> None:
    assert to_units("the U.S. economy", "en", "itn") == to_units("the US economy", "en", "itn")


def test_mixed_units() -> None:
    assert split_units("SPY涨了3.5%，QQQ跌了") == ["SPY", "涨", "了", "3.5%", "QQQ", "跌", "了"]
    assert to_units("SPY涨了3.5%，QQQ跌了", "zh", "raw") == ["SPY", "涨", "了", "3.5%", "QQQ", "跌", "了"]


@pytest.mark.parametrize(
    ("units", "text"),
    [
        (["SPY", "涨", "了", "3.5%", "QQQ"], "SPY涨了3.5% QQQ"),
        (["a", "b"], "a b"),
        (["你", "好"], "你好"),
    ],
)
def test_join_units_spaces_only_non_cjk_neighbours(units: list[str], text: str) -> None:
    assert join_units(units) == text


def test_canonical_converts_only_chinese_script() -> None:
    assert canonical("ＡＢ漲", "zh") == "AB涨"
    assert canonical("ＡＢ漲", "en") == "AB漲"
