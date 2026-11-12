import pytest

from ytverbatim.text.numbers import itn, quantity_tokens


def surfaces(text: str) -> list[str]:
    return [t.surface for t in quantity_tokens(text)]


@pytest.mark.parametrize("text", ["一样", "一直", "统一", "不一样", "十字路口", "万事如意", "一般来说"])
def test_lone_numeral_characters_are_not_quantities(text: str) -> None:
    assert quantity_tokens(text) == []


def test_ordinal_follows_di() -> None:
    (token,) = quantity_tokens("第一")
    assert (token.surface, token.value_itn, token.kind) == ("第一", "第1", "cjk_quantity")


@pytest.mark.parametrize(
    ("text", "value", "kind"),
    [
        ("一百五十期", "150期", "cjk_quantity"),
        ("二零二四年", "2024年", "cjk_quantity"),
        ("三十", "30", "cjk_quantity"),
        ("两百块", "200块", "cjk_quantity"),
        ("兩百", "200", "cjk_quantity"),
        ("廿五", "25", "cjk_quantity"),
        ("三点五", "3.5", "cjk_quantity"),
        ("三点五亿美元", "350000000美元", "cjk_quantity"),
        ("一个", "1个", "cjk_quantity"),
        ("五股", "5股", "cjk_quantity"),
        ("三个百分点", "3个百分点", "cjk_quantity"),
        ("百分之五十", "50%", "percent"),
    ],
)
def test_chinese_quantities_are_converted(text: str, value: str, kind: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.surface, token.value_itn, token.kind) == (text, value, kind)


def test_measure_word_makes_a_single_numeral_a_quantity() -> None:
    assert surfaces("买了三股和五个") == ["三股", "五个"]


def test_unconvertible_run_keeps_its_surface() -> None:
    (token,) = quantity_tokens("千万")
    assert token.value_itn == "千万"


@pytest.mark.parametrize(
    ("text", "surface", "value", "kind"),
    [
        ("1,000", "1,000", "1000", "number"),
        ("3.50", "3.50", "3.50", "number"),
        ("0700", "0700", "0700", "number"),
        ("-5%", "-5%", "-5%", "percent"),
        ("5 %", "5 %", "5%", "percent"),
        ("３.５％", "３.５％", "3.5%", "percent"),
        ("$1,500.50", "$1,500.50", "$1500.50", "currency"),
        ("200元", "200元", "200元", "currency"),
        ("3.5亿", "3.5亿", "350000000", "number"),
        ("3万美元", "3万美元", "30000美元", "currency"),
        ("150期", "150期", "150期", "number"),
        ("3.5亿股", "3.5亿股", "350000000股", "number"),
    ],
)
def test_digit_quantities(text: str, surface: str, value: str, kind: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.surface, token.value_itn, token.kind) == (surface, value, kind)


@pytest.mark.parametrize(
    "ticker",
    ["NVDA", "SPY", "Q2", "BRK.B", "0700.HK", "$NVDA", "$F", "NYSE:F"],
)
def test_tickers(ticker: str) -> None:
    (token,) = quantity_tokens(f"看 {ticker} 的")
    assert (token.surface, token.kind) == (ticker, "ticker")


def test_share_class_letter_before_gu() -> None:
    assert [(t.surface, t.kind) for t in quantity_tokens("A股")] == [("A", "ticker")]


def test_offsets_index_the_input() -> None:
    text = "SPY涨了3.5%，QQQ跌了一百五十期"
    for token in quantity_tokens(text):
        assert text[token.char_start : token.char_end] == token.surface
    assert surfaces(text) == ["SPY", "3.5%", "QQQ", "一百五十期"]


def test_dates_and_identifiers_are_not_split_into_signed_numbers() -> None:
    assert surfaces("2024-10-01 abc123") == ["2024", "10", "01"]


def test_digit_and_chinese_forms_of_a_quantity_share_one_value() -> None:
    forms = ["一百五十期", "150期", "三点五亿股", "3.5亿股"]
    values = [quantity_tokens(f)[0].value_itn for f in forms]
    assert values == ["150期", "150期", "350000000股", "350000000股"]


def test_itn_rewrites_tokens_only() -> None:
    assert itn("一百五十期涨了百分之五，一样 3.5亿") == "150期涨了5%，一样 350000000"
