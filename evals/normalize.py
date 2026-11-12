import re
import unicodedata
from functools import cache
from typing import Literal

import opencc
from whisper_normalizer.english import EnglishTextNormalizer

from ytverbatim.text.numbers import itn as quantities_to_itn

Language = Literal["zh", "en"]
View = Literal["raw", "itn"]

_ZH_FILLERS = ("嗯哼", "嗯", "啊", "呃", "额", "哦", "唉", "诶")
_EN_FILLERS = ("hmm", "mm", "mhm", "mmm", "uh", "um")
_KEPT_SYMBOLS = frozenset("%$¥€£")
_APOSTROPHES = frozenset("'’‘ʼ`")

_HAN = "㐀-䶿一-鿿豈-﫿\U00020000-\U0002fa1f"
_UNIT = re.compile(rf"[{_HAN}]|[$¥€£]?(?:(?![{_HAN}])[^\W_])+(?:[.,][0-9]+)*%?")
_HAN_UNIT = re.compile(rf"[{_HAN}]")
# Whitespace has already replaced every punctuation mark, so a filler is standalone when blanks or
# the string ends surround it.
_ZH_FILLER = re.compile(rf"(?<!\S)(?:{'|'.join(_ZH_FILLERS)})+(?!\S)")
_EN_FILLER = re.compile(rf"\b(?:{'|'.join(_EN_FILLERS)})\b", re.IGNORECASE)
_LETTER_DOT = re.compile(r"(?<=[A-Za-z])\.(?=[A-Za-z])")


@cache
def _t2s() -> opencc.OpenCC:
    return opencc.OpenCC("t2s")


@cache
def _english() -> EnglishTextNormalizer:
    return EnglishTextNormalizer()


def canonical(text: str, language: Language) -> str:
    """NFKC, plus traditional-to-simplified for Chinese. Offsets stay stable for the quantity scan."""
    folded = unicodedata.normalize("NFKC", text)
    return _t2s().convert(folded) if language == "zh" else folded


def _strip_symbols(text: str) -> str:
    out: list[str] = []
    text = _LETTER_DOT.sub("", text)
    last = len(text) - 1
    for i, ch in enumerate(text):
        if ch in _KEPT_SYMBOLS or unicodedata.category(ch)[0] not in "PS":
            out.append(ch)
        elif ch in _APOSTROPHES:
            continue
        elif ch in ".," and 0 < i < last and text[i - 1].isdecimal() and text[i + 1].isdecimal():
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def split_units(text: str) -> list[str]:
    """Mixed units: every CJK character alone, every Latin or digit run (with its `$` or `%`) as one."""
    return _UNIT.findall(text)


def join_units(units: list[str]) -> str:
    """Join units with a space only where two non-CJK units would otherwise fuse."""
    out: list[str] = []
    for unit in units:
        if out and not _HAN_UNIT.fullmatch(unit) and not _HAN_UNIT.fullmatch(out[-1]):
            out.append(" ")
        out.append(unit)
    return "".join(out)


def to_units(text: str, language: Language, view: View) -> list[str]:
    """Normalize ``text`` and split it into comparison units.

    The raw view keeps surface form and the case of Latin tokens, so ``nvda`` against ``NVDA`` or
    ``150期`` against ``一百五十期`` is an error. The itn view maps Chinese quantities to digits,
    runs English through Whisper's normalizer and casefolds.
    """
    text = canonical(text, language)
    if view == "itn":
        text = _english()(_LETTER_DOT.sub("", text)) if language == "en" else quantities_to_itn(text)
    text = _strip_symbols(text)
    text = _ZH_FILLER.sub(" ", text) if language == "zh" else _EN_FILLER.sub(" ", text)
    units = split_units(text)
    return [u.casefold() for u in units] if view == "itn" else units


def normalize(text: str, language: Language, view: View) -> str:
    return join_units(to_units(text, language, view))
