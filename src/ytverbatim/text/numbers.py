"""Quantity tokens: numbers, currency, percentages, tickers and Chinese numerals.

The patterns descend from the guards in ytscript's ``analyze/correct.py``. The Chinese rules are
stricter there than a plain numeral run, because ``一样`` and ``一直`` contain numeral characters
without being quantities.
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


def _alternation(words: tuple[str, ...]) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_DIGIT = "[0-9０-９]"
_NUMBER = rf"(?:{_DIGIT}+(?:,{_DIGIT}{{3}})*(?:\.{_DIGIT}+)?|\.{_DIGIT}+)"
_NO_ALNUM_BEFORE = r"(?<![A-Za-z0-9])"
_NO_ALNUM_AFTER = r"(?![A-Za-z0-9])"

_TOKEN = re.compile(
    "|".join(
        (
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
            rf"|(?P<magnitude>[万亿萬億])?(?P<measure>{_alternation(_MEASURE_WORDS)})?))",
        )
    )
)

_MIN_BARE_RUN = 2
_VARIANT_NUMERALS = str.maketrans({"兩": "两", "萬": "万", "億": "亿"})
_MAGNITUDE = {"万": 4, "萬": 4, "亿": 8, "億": 8}


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


def _cjk_value(run: str) -> str | None:
    numerals = run.translate(_VARIANT_NUMERALS).replace("廿", "二十").replace("卅", "三十")
    # cn2an rejects a magnitude after a decimal (三点五亿), so scale the decimal itself.
    scale = _MAGNITUDE[numerals[-1]] if "点" in numerals and numerals[-1] in "万亿" else 0
    try:
        value = Decimal(str(cn2an.cn2an(numerals[:-1] if scale else numerals, "smart")))
    except (ValueError, InvalidOperation):
        return None
    return _plain(value * Decimal(10) ** scale)


def _cjk_token(match: re.Match[str]) -> QToken | None:
    run, unit = match.group("run"), match.group("unit") or ""
    numeral_count = sum(1 for ch in run if ch != "点")
    percent, ordinal = match.group("pct"), match.group("ord")
    if not (percent or ordinal or unit or numeral_count >= _MIN_BARE_RUN):
        return None
    value = _cjk_value(run)
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
        digits = _plain(Decimal(digits) * Decimal(10) ** _MAGNITUDE[magnitude])
    sign = (match.group("sign") or match.group("sign2") or "").replace("−", "-")
    symbol = match.group("currency") or ""
    measure = match.group("measure") or ""
    itn = f"{sign}{symbol}{digits}{'%' if percent else ''}{measure}"
    kind: QKind = "percent" if percent else "currency" if symbol or measure in _CURRENCY_WORDS else "number"
    return QToken(match.start(), match.end(), match.group(0), itn, kind)


def quantity_tokens(text: str) -> list[QToken]:
    """Return the quantity tokens of ``text`` in order, with offsets into ``text`` itself.

    A Chinese numeral run is a quantity only when it has two or more numeral characters, is
    attached to a measure word, or follows ``第``. ``value_itn`` is the inverse text normalization
    of the token: ``一百五十期`` becomes ``150期`` and ``3.5亿`` becomes ``350000000``.
    """
    tokens: list[QToken] = []
    for match in _TOKEN.finditer(text):
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
