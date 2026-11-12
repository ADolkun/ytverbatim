"""Quantity tokens: numbers, currency, percentages, tickers, Chinese numerals and spelled-out English numbers.

The patterns descend from the guards in ytscript's ``analyze/correct.py``. The Chinese rules are
stricter than a plain numeral run, because many words contain numeral characters without being
quantities: idioms (``一样``, ``千万不要``, ``万一``), weekdays (``周一``) and approximations
(``两三个`` is "two or three", not 23).
"""

import re
from dataclasses import dataclass, field
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


# The same dollar amount is written $15, 15 dollars or 15美元; all of them normalise to $15.
_USD_WORDS = frozenset({"美元", "美金"})
_EN_MAGNITUDE = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12, "k": 3, "m": 6, "b": 9, "bn": 9}
_EN_UNITS = dict(
    zip(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ],
        range(20),
        strict=True,
    )
)
_EN_TENS = dict(
    zip(["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"], range(20, 100, 10), strict=True)
)
_EN_SCALES = {"thousand": 3, "million": 6, "billion": 9, "trillion": 12}
_EN_NUMBER_WORD = _alternation((*_EN_UNITS, *_EN_TENS, "hundred", *_EN_SCALES))
_NO_SCALE = 99
_TEN, _HUNDRED = 10, 100
_EN_PIECE = re.compile(rf"(?i:\b(?:{_EN_NUMBER_WORD}|and)\b)")

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
            rf"(?P<digits>{_NUMBER})"
            r"(?P<en_magnitude>\s?(?i:thousand|million|billion|trillion)\b|(?:[Kk]|M|B|bn)(?![A-Za-z0-9]))?"
            r"(?:(?P<percent> ?[%％]|\s?(?i:percent)\b)|(?P<dollars>\s?(?i:dollars?)\b)"
            rf"|(?P<magnitude>[百千][万亿萬億]*|[万亿萬億]+)?(?P<measure>{_alternation(_MEASURE_WORDS)})?))",
            rf"(?P<spelled>(?<![A-Za-z0-9-])(?i:{_EN_NUMBER_WORD})(?:(?:[ -]|\s+and\s+)(?i:{_EN_NUMBER_WORD}))*"
            r"(?![A-Za-z0-9-]))(?P<spelled_unit>\s+(?i:percent|dollars?)\b)?",
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
# One digit after 亿 or 万亿 is the next lower unit, as 五 in 一万五: 一亿五 is 1.5亿.
_YI_SHORTHAND = re.compile("(?P<head>.*?(?P<unit>万?亿))(?P<digit>[一二三四五六七八九])")
_FRACTION = re.compile(rf"([{''.join(_DIGIT_VALUE)}]+)([万亿]*)")


def _magnitude_exponent(magnitudes: str) -> int:
    return sum(_MAGNITUDE[ch] for ch in magnitudes)


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


def _cn2an_value(integer: str) -> Decimal | None:
    # ValueError is cn2an's "not one number". Its speaking mode (一万五 is 15000) raises KeyError when the lower
    # unit it wants is missing from its table, as after 十 (七十六十五, 一万十五).
    try:
        return Decimal(str(cn2an.cn2an(integer, "smart")))
    except (ValueError, KeyError, InvalidOperation):
        return None


def _integer_value(integer: str) -> str | None:
    if not any(ch in _PLACE_WORDS for ch in integer):
        # Digit by digit, keeping leading zeros as the digit path keeps 0700: 零零后 is 00后.
        return "".join(_COUNTING_DIGITS[ch] for ch in integer)
    shorthand = _YI_SHORTHAND.fullmatch(integer)
    if shorthand is None:
        value = _cn2an_value(integer)
    else:
        # cn2an's speaking mode has no unit below 亿, so the trailing digit of 一亿五 is placed here: 千万, or 千亿
        # after 万亿.
        head = _cn2an_value(shorthand["head"])
        lower_unit = Decimal(10) ** (_magnitude_exponent(shorthand["unit"]) - 1)
        value = None if head is None else head + int(_DIGIT_VALUE[shorthand["digit"]]) * lower_unit
    return None if value is None else _plain(value)


def _cjk_value(integer: str, fraction: str) -> str | None:
    whole = _integer_value(integer)
    if whole is None or not fraction:
        return whole
    # 一亿五点五 mixes shorthand with a decimal and is not one number.
    if _YI_SHORTHAND.fullmatch(integer):
        return None
    # The magnitude after a decimal scales the whole number: 一点五万亿 is 1.5 × 10^12.
    parts = _FRACTION.fullmatch(fraction)
    if parts is None:
        return None
    decimals = "".join(_DIGIT_VALUE[ch] for ch in parts.group(1))
    scale = _magnitude_exponent(parts.group(2))
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
    elif unit in _USD_WORDS:
        itn = f"{'第' if ordinal else ''}${value}"
    else:
        itn = f"{'第' if ordinal else ''}{value}{unit}"
    kind: QKind = "percent" if percent else "currency" if unit in _CURRENCY_WORDS else "cjk_quantity"
    return QToken(match.start(), match.end(), match.group(0), itn, kind)


def _number_token(match: re.Match[str]) -> QToken:
    digits = _ascii_digits(match.group("digits"))
    magnitude, percent = match.group("magnitude"), match.group("percent")
    en_magnitude = (match.group("en_magnitude") or "").strip().casefold()
    exponent = sum(_MAGNITUDE[ch] for ch in magnitude or "") + _EN_MAGNITUDE.get(en_magnitude, 0)
    if exponent:
        digits = _plain(Decimal(digits) * Decimal(10) ** exponent)
    sign = (match.group("sign") or match.group("sign2") or "").replace("−", "-")
    symbol = match.group("currency") or ""
    measure = match.group("measure") or ""
    if match.group("dollars") or measure in _USD_WORDS:
        symbol = symbol or "$"
        measure = "" if measure in _USD_WORDS else measure
    itn = f"{sign}{symbol}{digits}{'%' if percent else ''}{measure}"
    kind: QKind = "percent" if percent else "currency" if symbol or measure in _CURRENCY_WORDS else "number"
    return QToken(match.start(), match.end(), match.group(0), itn, kind)


def _spelled_tokens(match: re.Match[str]) -> list[QToken]:
    """Split a run of English number words into numbers and convert each: fifteen, twenty-five, two hundred and ten.

    Words that cannot continue the current number start the next one, so nineteen ninety is two numbers. A lone
    "one" is a pronoun or determiner far more often than a quantity (no one, one of them), so it counts only with a
    unit after it.
    """
    groups = _spelled_groups(
        [(piece[0].casefold(), piece.start(), piece.end()) for piece in _EN_PIECE.finditer(match.group("spelled"))]
    )
    offset = match.start()
    unit = match.group("spelled_unit")
    tokens: list[QToken] = []
    for number, group in enumerate(groups):
        # The unit belongs to the last number only, and only when no word was dropped between them.
        last = number == len(groups) - 1 and group.end == len(match.group("spelled"))
        suffix = (unit or "").strip().casefold() if last else ""
        if group.words == ["one"] and not suffix:
            continue
        stop = match.end() if suffix else offset + group.end
        kind: QKind
        if suffix == "percent":
            itn, kind = f"{group.value}%", "percent"
        elif suffix:
            itn, kind = f"${group.value}", "currency"
        else:
            itn, kind = str(group.value), "number"
        tokens.append(QToken(offset + group.start, stop, match.string[offset + group.start : stop], itn, kind))
    return tokens


@dataclass
class _SpelledNumber:
    start: int
    end: int = 0
    total: int = 0
    current: int = 0
    scale: int = _NO_SCALE
    last: str = ""
    words: list[str] = field(default_factory=list[str])

    def accepts(self, word: str) -> bool:
        if word in _EN_UNITS or word in _EN_TENS:
            if self.last == "tens":
                return 0 < _EN_UNITS.get(word, 0) < _TEN
            return self.last in ("hundred", "scale", "and")
        if word == "hundred":
            return self.last in ("unit", "tens") and self.current < _HUNDRED
        return self.last in ("unit", "tens", "hundred") and _EN_SCALES[word] < self.scale

    def add(self, word: str, end: int) -> None:
        self.words.append(word)
        self.end = end
        if word == "hundred":
            self.current *= _HUNDRED
            self.last = "hundred"
        elif word in _EN_SCALES:
            self.scale = _EN_SCALES[word]
            self.total += self.current * 10**self.scale
            self.current = 0
            self.last = "scale"
        else:
            self.current += _EN_UNITS.get(word, 0) + _EN_TENS.get(word, 0)
            self.last = "tens" if word in _EN_TENS else "unit"

    @property
    def value(self) -> int:
        return self.total + self.current


def _spelled_groups(pieces: list[tuple[str, int, int]]) -> list[_SpelledNumber]:
    """Group number words into numbers; ``and`` only joins a hundred or a scale to the words after it."""
    groups: list[_SpelledNumber] = []
    group: _SpelledNumber | None = None
    for word, start, end in pieces:
        if word == "and":
            if group is not None and group.last in ("hundred", "scale"):
                group.last = "and"
            else:
                group = None
            continue
        if group is not None and group.accepts(word):
            group.add(word, end)
            continue
        if word in _EN_UNITS or word in _EN_TENS:
            group = _SpelledNumber(start)
            group.add(word, end)
            groups.append(group)
        else:
            group = None
    return groups


def quantity_tokens(text: str) -> list[QToken]:
    """Return the quantity tokens of ``text`` in order, with offsets into ``text`` itself.

    A Chinese numeral run is a quantity only when it has two or more numeral characters, is
    attached to a measure word, or follows ``第``, and is neither an idiom nor a range of bare
    digits. ``value_itn`` is the inverse text normalization of the token: ``一百五十期`` becomes
    ``150期``, ``3.5亿`` becomes ``350000000`` and ``2万亿`` becomes ``2000000000000``. English
    magnitudes scale the same way (``$1.5 billion`` and ``$1.5B`` become ``$1500000000``), a
    dollar amount is ``$`` however it is written (``15 dollars``, ``15美元``), ``15 percent`` is
    ``15%``, and spelled-out numbers convert (``twenty-five`` is ``25``).
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
        elif match.group("spelled") is not None:
            tokens.extend(_spelled_tokens(match))
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
