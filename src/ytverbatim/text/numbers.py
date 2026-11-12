"""Quantity tokens: numbers, currency, percentages, tickers and Chinese numerals.

The patterns descend from the guards in ytscript's ``analyze/correct.py``. The Chinese rules are
stricter than a plain numeral run, because many words contain numeral characters without being
quantities: idioms (``一样``, ``千万不要``, ``万一``), weekdays (``周一``) and approximations
(``两三个`` is "two or three", not 23).
"""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

import cn2an

QKind = Literal["number", "currency", "percent", "ticker", "cjk_quantity"]

_NUMERALS = "〇零一二两兩三四五六七八九十百千万萬亿億廿卅"

# Longest alternatives first so 美元 wins over 元 and 个月 over 个.
_CURRENCY_WORDS = (
    "美元",
    "美金",
    "美分",
    "欧元",
    "歐元",
    "港元",
    "港币",
    "港幣",
    "人民币",
    "人民幣",
    "日元",
    "英镑",
    "英鎊",
    "新台币",
    "新臺幣",
    "台币",
    "臺幣",
    "澳元",
    "加元",
    "韩元",
    "韓元",
    "元",
    "块",
    "塊",
)
_MEASURE_WORDS = (
    *_CURRENCY_WORDS,
    "个百分点",
    "个月",
    "个",
    "股",
    "年",
    "月",
    "日",
    "号",
    "点钟",
    "点",
    "倍",
    "成",
    "期",
    "次",
    "天",
    "周",
    "季度",
    "季",
)


# Words whose numeral characters do not state a quantity. They are matched before numeral runs, so a
# run that starts earlier (三万一千) still claims its characters, while 万一 on its own is left alone.
_IDIOMS = (
    "一些",
    "一点儿",
    "一点点",
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
    "一一",
    "万一",
    "万万",
    "千千万万",
    "一切",
    "一共",
    "一度",
    "一律",
    "一再",
    "一向",
    "一贯",
    "一味",
    "一举",
    "一时",
    "一同",
    "一并",
    "一路",
    "一流",
    "一线",
    "一方面",
    "一口气",
    "一会儿",
    "一次性",
    "一成不变",
    "一股脑",
    "一股劲",
    "一个劲",
    "一天到晚",
    "一年到头",
    "一年四季",
    "一年半载",
    "三天两头",
    "一五一十",
    "一模一样",
    "一心一意",
    "三三两两",
    "七七八八",
)
_IDIOM_PATTERNS = (
    # "By all means": 千万不要, 千万别, 千万记住. 千万 before a measure word (千万元) stays a number.
    r"千万(?=[不别要记得莫]|小心|注意)",
    # "A little" (好一点, 一点也不); 一点五 is a decimal and 一点钟 or 一点半 a time.
    r"一点(?![〇零一二两兩三四五六七八九十钟半])",
    # Weekdays, not 周1; 每周一次 is "once a week".
    r"(?:周|星期|礼拜)[一二三四五六日天](?![次回遍趟])",
)


def _alternation(words: tuple[str, ...]) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_DIGIT = "[0-9０-９]"
_NUMBER = rf"(?:{_DIGIT}+(?:,{_DIGIT}{{3}})*(?:\.{_DIGIT}+)?|\.{_DIGIT}+)"
_NO_ALNUM_BEFORE = r"(?<![A-Za-z0-9])"
_NO_ALNUM_AFTER = r"(?![A-Za-z0-9])"

_TOKEN = re.compile(
    "|".join(
        (
            rf"(?P<idiom>{'|'.join(_IDIOM_PATTERNS)}|{_alternation(_IDIOMS)})",
            rf"(?P<cjk>(?P<pct>百分之)?(?P<ord>第)?(?P<run>[{_NUMERALS}]+(?:点[{_NUMERALS}]+)?)"
            rf"(?P<unit>{_alternation((*_MEASURE_WORDS, '%'))})?)",
            rf"(?P<cashtag>\$[A-Z]{{1,6}}(?:\.[A-Z]{{1,2}})?{_NO_ALNUM_AFTER})",
            rf"(?P<exchange>(?i:\b(?:NYSE|NASDAQ|AMEX|NYSEARCA|NYSEAMERICAN|BATS|OTC|OTCQX|OTCQB))\s*:\s*"
            rf"[A-Z]{{1,5}}(?:\.[A-Z])?{_NO_ALNUM_AFTER})",
            r"(?P<share_class>(?<![A-Za-z0-9])[A-Z](?=股))",
            rf"(?P<numeric_ticker>{_NO_ALNUM_BEFORE}\d{{2,}}\.[A-Z]{{2,}}{_NO_ALNUM_AFTER})",
            rf"(?P<symbol>{_NO_ALNUM_BEFORE}[A-Z][A-Z0-9]+(?:\.[A-Z]+)?{_NO_ALNUM_AFTER})",
            rf"(?P<number>{_NO_ALNUM_BEFORE}(?P<sign>[-+−])?(?P<currency>[$¥€£])?(?P<sign2>[-+−])?"
            rf"(?P<digits>{_NUMBER})(?:(?P<percent> ?[%％])"
            rf"|(?P<magnitude>[百千][万亿萬億]*|[万亿萬億]+)?(?P<measure>{_alternation(_MEASURE_WORDS)})?))",
        )
    )
)

_MIN_BARE_RUN = 2
_YEAR_DIGITS = 4
_VARIANT_NUMERALS = str.maketrans({"兩": "两", "萬": "万", "億": "亿"})
# Stacked magnitudes multiply: 万亿 is 10^12, so exponents add.
_MAGNITUDE = {"百": 2, "千": 3, "万": 4, "萬": 4, "亿": 8, "億": 8}
_DIGIT_VALUE = {ch: str(i) for i, ch in enumerate("零一二三四五六七八九")} | {"〇": "0"}
# 两 counts things (两个) but is never read out as a digit of a code or decimal.
_COUNTING_DIGITS = _DIGIT_VALUE | {"两": "2"}
_PLACE_WORDS = frozenset("十百千万亿")
_ZEROS = frozenset("〇零")
# Two bare digits side by side (四五十, 三四百) are a range, never a positional numeral.
_ADJACENT_DIGITS = re.compile("[一二两三四五六七八九]{2}")
_FRACTION = re.compile(rf"([{''.join(_DIGIT_VALUE)}]+)([万亿]*)")


@dataclass(frozen=True, slots=True)
class QToken:
    char_start: int
    char_end: int
    surface: str
    value_itn: str
    kind: QKind


def _plain(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _ascii_digits(digits: str) -> str:
    return digits.translate({0xFF10 + i: ord("0") + i for i in range(10)}).replace(",", "")


def _is_quantity(integer: str, unit: str) -> bool:
    """Tell a numeral from bare digits that only look like one.

    Without a place word, digits read one by one form a number only as a code or year, which speech
    marks with 〇 or 零 (二〇一八年) unless it is a four-digit year (一九九七年). Anything else is a
    range or an idiom: 两三个, 一一对应, 五五开.
    """
    if any(ch in _PLACE_WORDS for ch in integer):
        return not _ADJACENT_DIGITS.search(integer)
    if len(integer) == 1:
        return True
    return "两" not in integer and (bool(_ZEROS & set(integer)) or (len(integer) == _YEAR_DIGITS and unit == "年"))


def _integer_value(integer: str) -> str | None:
    if not any(ch in _PLACE_WORDS for ch in integer):
        # Digit by digit, keeping leading zeros as the digit path keeps 0700: 零零后 is 00后.
        return "".join(_COUNTING_DIGITS[ch] for ch in integer)
    try:
        return _plain(Decimal(str(cn2an.cn2an(integer, "smart"))))
    except (ValueError, InvalidOperation):
        return None


def _cjk_value(integer: str, fraction: str) -> str | None:
    whole = _integer_value(integer)
    if whole is None or not fraction:
        return whole
    # The magnitude after a decimal scales the whole number: 一点五万亿 is 1.5 × 10^12.
    parts = _FRACTION.fullmatch(fraction)
    if parts is None:
        return None
    decimals = "".join(_DIGIT_VALUE[ch] for ch in parts.group(1))
    scale = sum(_MAGNITUDE[ch] for ch in parts.group(2))
    return _plain(Decimal(f"{whole}.{decimals}") * Decimal(10) ** scale)


def _cjk_token(match: re.Match[str]) -> QToken | None:
    run, unit = match.group("run"), match.group("unit") or ""
    numeral_count = sum(1 for ch in run if ch != "点")
    percent, ordinal = match.group("pct"), match.group("ord")
    if not (percent or ordinal or unit or numeral_count >= _MIN_BARE_RUN):
        return None
    numerals = run.translate(_VARIANT_NUMERALS).replace("廿", "二十").replace("卅", "三十")
    integer, _, fraction = numerals.partition("点")
    if not _is_quantity(integer, unit):
        return None
    value = _cjk_value(integer, fraction)
    if value is None:
        itn = match.group(0)
    elif percent:
        itn = f"{value}%{unit}"
    else:
        itn = f"{'第' if ordinal else ''}{value}{unit}"
    return QToken(match.start(), match.end(), match.group(0), itn, "percent" if percent else "cjk_quantity")


def _number_token(match: re.Match[str]) -> QToken:
    digits = _ascii_digits(match.group("digits"))
    magnitude, percent = match.group("magnitude"), match.group("percent")
    if magnitude:
        digits = _plain(Decimal(digits) * Decimal(10) ** sum(_MAGNITUDE[ch] for ch in magnitude))
    sign = (match.group("sign") or match.group("sign2") or "").replace("−", "-")
    symbol = match.group("currency") or ""
    measure = match.group("measure") or ""
    itn = f"{sign}{symbol}{digits}{'%' if percent else ''}{measure}"
    kind: QKind = "percent" if percent else "currency" if symbol or measure in _CURRENCY_WORDS else "number"
    return QToken(match.start(), match.end(), match.group(0), itn, kind)


def quantity_tokens(text: str) -> list[QToken]:
    """Return the quantity tokens of ``text`` in order, with offsets into ``text`` itself.

    A Chinese numeral run is a quantity only when it has two or more numeral characters, is
    attached to a measure word, or follows ``第``, and is neither an idiom nor a range of bare
    digits. ``value_itn`` is the inverse text normalization of the token: ``一百五十期`` becomes
    ``150期``, ``3.5亿`` becomes ``350000000`` and ``2万亿`` becomes ``2000000000000``.
    """
    tokens: list[QToken] = []
    for match in _TOKEN.finditer(text):
        if match.group("idiom") is not None:
            continue
        if match.group("cjk") is not None:
            token = _cjk_token(match)
            if token is not None:
                tokens.append(token)
        elif match.group("number") is not None:
            tokens.append(_number_token(match))
        else:
            surface = match.group(0)
            tokens.append(QToken(match.start(), match.end(), surface, surface, "ticker"))
    return tokens


def itn(text: str) -> str:
    """Replace every quantity token of ``text`` by its ``value_itn`` and leave the rest untouched."""
    parts: list[str] = []
    cursor = 0
    for token in quantity_tokens(text):
        parts.extend((text[cursor : token.char_start], token.value_itn))
        cursor = token.char_end
    parts.append(text[cursor:])
    return "".join(parts)
