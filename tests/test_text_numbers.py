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
        ("两百块", "200块", "currency"),
        ("兩百", "200", "cjk_quantity"),
        ("廿五", "25", "cjk_quantity"),
        ("三点五", "3.5", "cjk_quantity"),
        ("三点五亿美元", "$350000000", "currency"),
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
        ("3万美元", "3万美元", "$30000", "currency"),
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


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("2万亿美元", "$2000000000000"),
        ("两万亿美元", "$2000000000000"),
        ("1.5万亿", "1500000000000"),
        ("一点五万亿", "1500000000000"),
        ("3.5亿", "350000000"),
        ("三点五亿", "350000000"),
        ("2千万美元", "$20000000"),
        ("5百万", "5000000"),
    ],
)
def test_stacked_magnitudes_are_one_multiplier(text: str, value: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.surface, token.value_itn) == (text, value)


@pytest.mark.parametrize("text", ["两三个", "三五个", "一一对应", "五五开", "三三两两", "四五十个", "三四百", "九七年"])
def test_adjacent_bare_digits_are_a_range_or_an_idiom_not_one_number(text: str) -> None:
    assert quantity_tokens(text) == []
    assert itn(text) == text


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("二〇一八年", "2018年"),
        ("二零二四年", "2024年"),
        ("一九九七年", "1997年"),
        ("零零后", "00后"),
        ("零点五", "0.5"),
    ],
)
def test_digit_by_digit_years_and_codes(text: str, value: str) -> None:
    assert itn(text) == value


@pytest.mark.parametrize(
    "text",
    [
        "千万不要",
        "千万别",
        "千万记住",
        "万一",
        "万万没想到",
        "一一",
        *(f"周{d}" for d in "一二三四五六日天"),
        *(f"星期{d}" for d in "一二三四五六日天"),
        "礼拜三",
        "一些",
        "一点儿",
        "好一点",
        "一点也不",
        "一下",
        "一定",
        "一起",
        "一直",
        "一样",
        "一般",
        "一致",
        "统一",
        "唯一",
        "一旦",
        "一边",
        "一次性",
        "一成不变",
        "一股脑",
        "一个劲",
        "一天到晚",
        "三天两头",
        "一五一十",
    ],
)
def test_idioms_are_not_quantities(text: str) -> None:
    assert quantity_tokens(text) == []
    assert itn(text) == text


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("每周一次", "每周1次"),
        ("一千万元", "10000000元"),
        ("三万一千", "31000"),
        ("一点五", "1.5"),
        ("一点钟", "1点钟"),
        ("一亿两千万", "120000000"),
        ("百分之三点五", "3.5%"),
        ("一成", "1成"),
    ],
)
def test_quantities_next_to_idiom_lookalikes_still_convert(text: str, value: str) -> None:
    assert itn(text) == value


@pytest.mark.parametrize(
    ("text", "surface", "value", "kind"),
    [
        ("$15 billion", "$15 billion", "$15000000000", "currency"),
        ("15 billion", "15 billion", "15000000000", "number"),
        ("$1.5B", "$1.5B", "$1500000000", "currency"),
        ("2.5 Million", "2.5 Million", "2500000", "number"),
        ("15K", "15K", "15000", "number"),
        ("3bn", "3bn", "3000000000", "number"),
        ("15 percent", "15 percent", "15%", "percent"),
        ("15 dollars", "15 dollars", "$15", "currency"),
        ("15 billion dollars", "15 billion dollars", "$15000000000", "currency"),
    ],
)
def test_english_magnitudes_and_unit_words(text: str, surface: str, value: str, kind: str) -> None:
    (token,) = quantity_tokens(text)
    assert (token.surface, token.value_itn, token.kind) == (surface, value, kind)


def test_dollar_amounts_agree_however_they_are_written() -> None:
    values = {
        quantity_tokens(text)[0].value_itn for text in ("$15", "15美元", "十五美元", "15 dollars", "fifteen dollars")
    }
    assert values == {"$15"}
    assert quantity_tokens("$15 billion")[0].value_itn != quantity_tokens("$15 million")[0].value_itn


def test_magnitude_words_need_a_word_boundary() -> None:
    assert surfaces("15 millionaires and 15 percentage points") == ["15", "15"]


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("fifteen", ["15"]),
        ("Twenty-five", ["25"]),
        ("twenty five percent", ["25%"]),
        ("two hundred and ten dollars", ["$210"]),
        ("fifteen hundred", ["1500"]),
        ("one million two hundred thousand", ["1200000"]),
        ("nineteen ninety", ["19", "90"]),
        ("five and six", ["5", "6"]),
        ("one percent", ["1%"]),
    ],
)
def test_spelled_out_english_numbers(text: str, values: list[str]) -> None:
    assert [t.value_itn for t in quantity_tokens(text)] == values


@pytest.mark.parametrize("text", ["no one knows", "one of them", "someone", "the tenant", "a million reasons"])
def test_english_words_that_are_not_quantities(text: str) -> None:
    assert quantity_tokens(text) == []
