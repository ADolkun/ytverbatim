import pytest

from ytverbatim.text.guards import EntityChange, check_fix, osa_distance, symbol_changes


def rejected_by(original: str, fixed: str) -> str | None:
    return check_fix(original, fixed).rejected_by


@pytest.mark.parametrize(
    ("original", "fixed"),
    [
        # Ported: legitimate bilingual prose fixes that keep data intact.
        ("动态适应率", "动态市盈率"),
        ("NVDA 适应率 2024", "NVDA 市盈率 2024"),
        ("第三季度适应率", "第三季度市盈率"),
        ("这家公司的动态适应率比较高", "这家公司的动态市盈率比较高"),
        ("适应率", "市盈率"),
        ("正线", "证券"),
        ("今年的适应率", "今年的市盈率"),
        ("buy BRK.B now thats good", "buy BRK.B now that's good"),
        ("the ¥100 cost is to hihg", "the ¥100 cost is too high"),
        ("NVDA的适应率是−10％", "NVDA的市盈率是−10％"),
        ("the yeild is -10% for BRK.B", "the yield is -10% for BRK.B"),
        ("the ＄１００ cost is to hihg", "the ＄１００ cost is too high"),
        ("the yeild is - 10 %", "the yield is - 10 %"),
        ("现在的适应率是负十", "现在的市盈率是负十"),
        ("reterns were not negative ten percent", "returns were not negative ten percent"),
        ("the 10 million dollars is to hihg", "the 10 million dollars is too high"),
        ("这个地方不是毛利滤", "这个地方不是毛利率"),
        ("价格500亿美元的适应率", "价格500亿美元的市盈率"),
        ("buy $F now thats good", "buy $F now that's good"),
        ("I have a typoo", "I have a typo"),
        # Ported: punctuation and buy/sell-equivalent forms.
        ("我们买了它", "我们买了它。"),
        ("我们買了它", "我们买了它"),
        ("他们賣了它", "他们卖了它"),
        ("we Buy apples", "we buy apples."),
        ("we buy shares", "we bought shares"),
        ("we sell shares", "we sold shares"),
        ("the buyer is here", "the buyer is here."),
        ("the resale is final", "the resale is final."),
        ("the buyback is here", "the buyback is here."),
        ("the buy2 label is here", "the buy2 label is here."),
        ("the _sell_ label is here", "the _sell_ label is here."),
        ("the market is volatil", "the market is volatile"),
        # Loosened: words that merely contain a numeral character.
        ("他一致在说这个问题", "他一直在说这个问题"),
        ("这两个公司的情况依样", "这两个公司的情况一样"),
        ("这个价格是分合理的", "这个价格十分合理的"),
        ("万以出现亏损怎么办", "万一出现亏损怎么办"),
    ],
)
def test_safe_fix_is_accepted(original: str, fixed: str) -> None:
    assert rejected_by(original, fixed) is None


@pytest.mark.parametrize(
    ("original", "fixed", "guard"),
    [
        ("这是一个完整的中文句子用于测试", "", "empty"),
        ("这是一个完整的中文句子用于测试", "短", "shrink"),
        ("短", "这是被模型大幅扩写并远超原始长度的内容不应被接受啊啊啊啊啊啊啊", "balloon"),
        ("目标价 $1000", "目标价 $1500", "digits"),
        ("price １０００", "price ２０００", "digits"),
        ("we buy 10 shares", "we buy 20 shares", "digits"),
        ("revenue was $1.2M", "revenue was $12M", "digits"),
        ("目标价位是一百元", "目标价位是一千元", "cjk_numerals"),
        ("他买了一百股", "他买了一千股", "cjk_numerals"),
        ("我们有三个方案", "我们有四个方案", "cjk_numerals"),
        ("这是第一个问题", "这是第二个问题", "cjk_numerals"),
        ("今天买三送一的活动", "今天买四送一的活动", "cjk_numerals"),
        ("returns fell to -10% this year", "returns fell to 10% this year", "numeric_spans"),
        ("the yield is 10% right now", "the yield is 10 right now", "numeric_spans"),
        ("the price is € 100 right now", "the price is 100 right now", "quantities"),
        ("revenue was 10 million dollars", "revenue was 10 billion dollars", "quantities"),
        ("金额是一百美元", "金额是一百港元", "quantities"),
        ("现在价格是￥一百元", "现在价格是一百元", "quantities"),
        ("价格100块", "价格100元", "quantities"),
        ("we do not buy shares", "we do buy shares", "negation"),
        ("这个地方不是毛利率", "这个地方是毛利率", "negation"),
        ("这个不行那个行", "这个行那个不行", "negation"),
        ("他要買股票", "他要賣股票", "direction"),
        ("we buy then sell", "we sell then buy", "direction"),
        ("we hold shares", "we sell shares", "direction"),
        ("buy BRK.B today", "buy BRK.A today", "numeric_spans"),
        ("the cost is ¥100", "the cost is €100", "numeric_spans"),
        ("buy $F today", "buy $T today", "quantities"),
        ("买入F股", "买入T股", "quantities"),
        ("这是中文句子", "english sentence", "cjk_retention"),
        ("这家公司的动态适应率比较高", "随便讲讲其他的话题内容好", "cjk_similarity"),
        ("这家公司的动态适应率比较高", "这家公司的策略方向相当清楚", "cjk_similarity"),
        # Acronyms and tickers: injected, dropped, replaced, or respelled too far.
        ("买入 NVDA 和 AMD", "买入 NVDA 和 INTC", "symbols"),
        ("今年的适应率", "今年的NVDA适应率", "symbols"),
        ("NVDA 的适应率很高", "的市盈率很高", "symbols"),
        ("AAPL leads while NVDA falls", "NVDA leads while AAPL falls", "symbols"),
        ("BUY", "buy", "symbols"),
        ("the AI trade", "the EV trade", "symbols"),
    ],
)
def test_unsafe_fix_is_rejected_by_the_named_guard(original: str, fixed: str, guard: str) -> None:
    assert rejected_by(original, fixed) == guard


@pytest.mark.parametrize(
    ("original", "fixed"),
    [
        # Ported: every recognized data, modifier, direction and negation change.
        ("returns fell to −10% this year", "returns fell to +10% this year"),
        ("the yield is 10 right now", "the yield is 10% right now"),
        ("the yield is - 10 % right now", "the yield is + 10 % right now"),
        ("the yield is 10 % right now", "the yield is 10 right now"),
        ("the price is 100 € right now", "the price is 100 right now"),
        ("NVDA-10% today", "NVDA+10% today"),
        ("现在价格是＄１００", "现在价格是１００"),
        ("现在价格是￥１００", "现在价格是＄１００"),
        ("现在收益率是负十", "现在收益率是十"),
        ("现在收益率是負１０％", "现在收益率是１０％"),
        ("只有500亿美元", "只有500亿欧元"),
        ("价格100元", "价格100美元"),
        ("价格100美金", "价格100港币"),
        ("金额是人民币100元", "金额是日元100元"),
        ("价格100分", "价格100元"),
        ("AAPL tracks AAPL and NVDA", "AAPL tracks and NVDA"),
        ("AAPL tracks NVDA", "AAPL tracks AAPL and NVDA"),
        ("BRK.A leads while BRK.B falls", "BRK.B leads while BRK.A falls"),
        ("buy NYSE:F today", "buy NYSE:C today"),
        ("buy $F and $T today", "buy $T and $F today"),
        ("buy NYSE:F and NYSE:F", "buy NYSE:F"),
        ("AAPL is 10% while NVDA is 10", "AAPL is 10 while NVDA is 10%"),
        ("returns were negative 10%", "returns were positive 10%"),
        ("returns were minus 10%", "returns were plus 10%"),
        ("the price is 100 dollars", "the price is 100 euros"),
        ("revenue was ten million dollars", "revenue was twenty million dollars"),
        ("the yield is 10 basis points", "the yield is 10 percentage points"),
        ("do not buy at 100", "do buy at 100"),
        ("do buy at 100", "do not buy at 100"),
        ("you cannot buy at 100", "you can buy at 100"),
        ("you can't buy at 100", "you can buy at 100"),
        ("you don’t buy at 100", "you do buy at 100"),
        ("never buy at 100", "buy at 100"),
        ("this is not 10 but 20", "this is 10 but not 20"),
        ("这家公司没有现金流", "这家公司有现金流"),
        ("这家公司尚未盈利", "这家公司已经盈利"),
        ("这个数据并非毛利率", "这个数据就是毛利率"),
        ("你要卖put", "你要买put"),
        ("顶部就慢", "顶部就卖"),
        ("買入之后賣出", "賣出之后買入"),
        ("我们买入之后卖出", "我们买入之后买入"),
        ("we bought shares", "we sold shares"),
        ("we are selling shares", "we are buying shares"),
        ("we buy shares", "we hold shares"),
    ],
)
def test_ported_unsafe_fixes_stay_rejected(original: str, fixed: str) -> None:
    assert rejected_by(original, fixed) is not None


@pytest.mark.parametrize(
    ("original", "fixed", "before", "after"),
    [
        ("under GAP rules the margin fell", "under GAAP rules the margin fell", "GAP", "GAAP"),
        ("这只股票被纳入SMCI指数", "这只股票被纳入MSCI指数", "SMCI", "MSCI"),
        ("the GDP print and the GAP filing", "the GDP print and the GAAP filing", "GAP", "GAAP"),
    ],
)
def test_near_acronym_fix_is_accepted_and_reported(original: str, fixed: str, before: str, after: str) -> None:
    verdict = check_fix(original, fixed)
    assert verdict.ok
    (change,) = verdict.entity_changes
    assert (change.before, change.after) == (before, after)
    assert fixed[change.char_start : change.char_end] == after


@pytest.mark.parametrize(
    ("original", "fixed"),
    [
        ("按照SAP准则计算", "按照GAAP准则计算"),
        ("the GDP print", "the GAAP print"),
        ("AI chips", "EV chips"),
    ],
)
def test_acronym_changes_beyond_one_edit_are_rejected(original: str, fixed: str) -> None:
    assert rejected_by(original, fixed) is not None


def test_unchanged_symbols_report_no_change() -> None:
    assert check_fix("NVDA 适应率 2024", "NVDA 市盈率 2024").entity_changes == ()


def test_symbol_changes_point_into_the_corrected_text() -> None:
    assert symbol_changes("accordng to GAP and SMCI", "according to GAAP and MSCI") == (
        EntityChange("GAP", "GAAP", 13, 17),
        EntityChange("SMCI", "MSCI", 22, 26),
    )


def test_symbol_changes_refuse_a_changed_count() -> None:
    with pytest.raises(ValueError, match="symbol count"):
        symbol_changes("NVDA and AMD", "NVDA")


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [
        ("", "", 0),
        ("GAP", "GAP", 0),
        ("GAP", "GAAP", 1),
        ("SMCI", "MSCI", 1),
        ("CA", "ABC", 3),
        ("NVDA", "", 4),
        ("AAPL", "NVDA", 4),
        ("GDP", "GAAP", 2),
    ],
)
def test_osa_distance(a: str, b: str, distance: int) -> None:
    assert osa_distance(a, b) == distance
    assert osa_distance(b, a) == distance
