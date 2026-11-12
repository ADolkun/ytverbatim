"""Deterministic guards on an LLM correction of one transcript segment.

A correction may fix homophones and garbled terms; it must never change what the speaker asserted. Every guard
compares the original with the proposed text and rejects recognized data, negation or direction changes and
destructive rewrites. They are conservative checks, not a proof of semantic equivalence.

Ported from ytscript's ``analyze/correct.py`` with two guards loosened: an uppercase acronym or ticker may move to a
near spelling (GAP→GAAP), and Chinese numerals are compared as quantities, so a word that merely contains a numeral
character (一样, 一直) can be fixed.
"""

import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from ytverbatim.text.numbers import quantity_tokens

GuardName = Literal[
    "empty",
    "shrink",
    "balloon",
    "digits",
    "numeric_spans",
    "quantities",
    "cjk_numerals",
    "negation",
    "direction",
    "symbols",
    "cjk_retention",
    "cjk_similarity",
]

MIN_KEEP_RATIO = 0.5
# A real homophone fix keeps roughly the original length, so a fix longer than max(floor, original × ratio) is a
# hallucination.
_BALLOON_RATIO = 1.6
_BALLOON_FLOOR = 20
# A predominantly-CJK segment's fix must keep this share of its CJK characters, or it was translated or
# transliterated.
_CJK_RETENTION_FLOOR = 0.5
# A wholesale rewrite (or a same-length mass swap) changes many characters. The ratio is noisy when one legitimate
# fix dominates a very short segment, so it applies only from a minimum length.
_CJK_MIN_SIMILARITY = 0.65
_CJK_RATIO_MIN_CHARS = 8
# GAP→GAAP is one edit and SMCI→MSCI one transposition. Two edits already reach a different ticker (SAP→GAAP).
_MAX_SYMBOL_DISTANCE = 1

_CJK = re.compile(r"[\u4e00-\u9fff]")
_CJK_NUMERAL_CHARS = "〇零一二两兩三四五六七八九十百千万萬亿億廿卅"
_CJK_NUMERAL = re.compile(f"[{_CJK_NUMERAL_CHARS}]")
# `\d` is Unicode-aware and already matches full-width ０-９; spelling them out keeps a later narrowing to ASCII from
# dropping protection for the full-width digits common on CJK finance slides.
_DIGIT_RUN = re.compile(r"[\d０-９]+(?:\.[\d０-９]+)?")
_SIGN_OR_CURRENCY = r"(?:[-+−负負正](?:\s*[$¥€£])?\s*|[$¥€£](?:\s*[-+−负負正])?\s*)"
# Digit quantities keep their sign, currency and unit. Chinese numerals are compared as quantity tokens instead, so
# only a sign or currency symbol in front of them is tracked here (¥一百元 → 一百元 drops the currency).
_QUANTITY = (
    rf"{_SIGN_OR_CURRENCY}?(?:\d+(?:,\d{{3}})*(?:\.\d+)?|\.\d+)(?:\s*[%$¥€£元块塊分])?"
    rf"|{_SIGN_OR_CURRENCY}(?=[{_CJK_NUMERAL_CHARS}])"
)
# Bare uppercase runs (NVDA, GAAP, SPX): acronyms and tickers, the one symbol class a fix may respell.
_BARE_SYMBOL = re.compile(r"[A-Z]{2,}")
# Dotted, exchange-prefixed and share-class tickers stay exact: BRK.B→BRK.A or $F→$T is a different security even
# though it is one letter away. Bare single letters (F, C, T) collide with English words, so only their explicit
# financial forms are protected.
_DOTTED_TICKER = re.compile(r"[A-Z]{2,}\.[A-Z]+|\d{2,}\.[A-Z]{2,}")
_SINGLE_FINANCIAL_TICKER = re.compile(
    r"\$\s*[A-Z](?:\.[A-Z])?(?![A-Za-z0-9])"
    r"|(?i:\b(?:NYSE|NASDAQ|AMEX|NYSEARCA|NYSEAMERICAN|BATS|OTC|OTCQX|OTCQB))\s*:\s*[A-Z](?:\.[A-Z])?(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])[A-Z](?=股)"
)
# The major non-$ symbols too, so a ¥100→€100 swap with unchanged digits is still a changed amount.
_CURRENCY_AMOUNT = re.compile(r"[$¥€£][\d０-９][\d０-９.,]*")
# Spoken units carry value even when the recognizer writes the number as digits. A finite lexicon is deliberately
# conservative; it is not a semantic proofreader.
_NUMBER_WORDS = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundreds?|thousands?|millions?|billions?|trillions?|half|halves|quarters?"
)
_QUANTITY_WORDS = (
    r"negative|positive|minus|plus|dollars?|cents?|euros?|pounds?|pence|pennies|yen|yuan|renminbi|"
    r"sterling|rupees?|won|basis\s+points?|percentage\s+points?|percent(?:age)?|bps"
)
_CJK_CURRENCY_WORDS = (
    r"美元|美金|美分|欧元|歐元|欧分|歐分|港元|港币|港幣|人民币|人民幣|日元|日圆|日圓|"
    r"英镑|英鎊|新台币|新臺幣|台币|臺幣|澳元|澳币|澳幣|加元|新西兰元|紐西蘭元|瑞士法郎|韩元|韓元|韩币|韓幣"
)
_QUANTITY_TOKEN = re.compile(
    rf"{_QUANTITY}|(?i:\b(?:{_NUMBER_WORDS}|{_QUANTITY_WORDS})\b)|"
    rf"{_CJK_CURRENCY_WORDS}|{_SINGLE_FINANCIAL_TICKER.pattern}|{_DOTTED_TICKER.pattern}"
)
# Ordered numeric surface tokens: alphanumeric and dotted tickers (Q2, BRK.B, 0700.HK) and numbers with their sign,
# currency and percent. Letter-only symbols and Chinese numerals have their own guards.
_NUMERIC_SPAN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z][A-Z0-9]+\.[A-Z]+|[A-Z][A-Z0-9]*[0-9][A-Z0-9]*|\d{2,}\.[A-Z]{2,})(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])[-+−]?(?:[$¥€£][-+−]?)?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)%?"
)
_NEGATION = re.compile(
    r"(?i:\b(?:not|no|never|neither|nor|without|cannot|"
    r"(?:ca|could|do|does|did|has|have|had|is|are|was|were|should|would|must|need|dare|wo|sha|ai)n'?t)\b)"
    r"|[不没沒无無未非勿别別莫甭否负負]"
)
_LEXICAL_UNIT = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]")
_ACTION_DIRECTION = re.compile(
    r"[买買卖賣]|(?<![A-Za-z0-9_])(?:buying|bought|buys|buy|selling|sells|sold|sell)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_BUY_FORMS = frozenset({"买", "買", "buy", "buys", "buying", "bought"})


@dataclass(frozen=True, slots=True)
class EntityChange:
    """One respelled acronym or ticker; the offsets are half-open into the corrected text."""

    before: str
    after: str
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    rejected_by: GuardName | None
    entity_changes: tuple[EntityChange, ...] = ()

    @property
    def ok(self) -> bool:
        return self.rejected_by is None


def check_fix(original: str, fixed: str, *, min_keep_ratio: float = MIN_KEEP_RATIO) -> GuardVerdict:
    """Accept ``fixed`` as a correction of ``original``, or name the first guard it breaks.

    An accepted fix lists every acronym or ticker it respelled, so the caller can flag each one.
    """
    checks: tuple[tuple[GuardName, Callable[[], bool]], ...] = (
        ("empty", lambda: not fixed),
        ("shrink", lambda: len(fixed) < min_keep_ratio * len(original)),
        ("balloon", lambda: len(fixed) > max(_BALLOON_FLOOR, int(len(original) * _BALLOON_RATIO))),
        ("digits", lambda: _DIGIT_RUN.findall(fixed) != _DIGIT_RUN.findall(original)),
        ("numeric_spans", lambda: _numeric_spans(fixed) != _numeric_spans(original)),
        ("quantities", lambda: _quantity_runs(fixed) != _quantity_runs(original)),
        ("cjk_numerals", lambda: _cjk_numerals_changed(original, fixed)),
        ("negation", lambda: _negation_signature(fixed) != _negation_signature(original)),
        ("direction", lambda: _direction_signature(fixed) != _direction_signature(original)),
        ("symbols", lambda: _exact_symbols(fixed) != _exact_symbols(original)),
        ("cjk_retention", lambda: _cjk_retention_broken(original, fixed)),
        ("cjk_similarity", lambda: _cjk_similarity_broken(original, fixed)),
    )
    for name, broken in checks:
        if broken():
            return GuardVerdict(name)
    if len(_BARE_SYMBOL.findall(fixed)) != len(_BARE_SYMBOL.findall(original)):
        return GuardVerdict("symbols")
    changes = symbol_changes(original, fixed)
    if any(not _is_near_symbol(change.before, change.after) for change in changes):
        return GuardVerdict("symbols")
    return GuardVerdict(None, changes)


def symbol_changes(original: str, fixed: str) -> tuple[EntityChange, ...]:
    """Acronyms and tickers that differ position by position; both texts must hold the same number of them."""
    before = list(_BARE_SYMBOL.finditer(original))
    after = list(_BARE_SYMBOL.finditer(fixed))
    if len(before) != len(after):
        raise ValueError(f"symbol count changed from {len(before)} to {len(after)}")
    return tuple(
        EntityChange(old[0], new[0], new.start(), new.end())
        for old, new in zip(before, after, strict=True)
        if old[0] != new[0]
    )


def osa_distance(a: str, b: str) -> int:
    """Optimal string alignment distance: insertions, deletions, substitutions and adjacent transpositions."""
    previous2: list[int] = []
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i, *([0] * len(b))]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if i > 1 and j > 1 and char_a == b[j - 2] and a[i - 2] == char_b:
                current[j] = min(current[j], previous2[j - 2] + 1)
        previous2, previous = previous, current
    return previous[-1]


def _is_near_symbol(before: str, after: str) -> bool:
    limit = min(_MAX_SYMBOL_DISTANCE, min(len(before), len(after)) - 1)
    return osa_distance(before, after) <= limit


def _cjk_numerals_changed(original: str, fixed: str) -> bool:
    """Quantities must keep their values in order, and no numeral may be swapped for another.

    Tickers are left to the symbol guard. Quantity tokens skip words that merely contain a numeral (一样, 一直), so
    the substitution check catches a bare numeral that changed value without forming a token, as in 买三送一→买四送一.
    """

    def values(text: str) -> list[tuple[str, str]]:
        return [(token.value_itn, token.kind) for token in quantity_tokens(text) if token.kind != "ticker"]

    if values(original) != values(fixed):
        return True
    opcodes = SequenceMatcher(None, original, fixed, autojunk=False).get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != "replace":
            continue
        old, new = _CJK_NUMERAL.findall(original[i1:i2]), _CJK_NUMERAL.findall(fixed[j1:j2])
        if old and new and old != new:
            return True
    return False


def _numeric_spans(text: str) -> list[str]:
    return _NUMERIC_SPAN.findall(unicodedata.normalize("NFKC", text))


def _quantity_runs(text: str) -> list[str]:
    return [token.casefold() for token in _QUANTITY_TOKEN.findall(unicodedata.normalize("NFKC", text))]


def _negation_signature(text: str) -> list[tuple[int, str]]:
    normalized = unicodedata.normalize("NFKC", text).replace("’", "'").casefold()
    # Positions count lexical units, so punctuation fixes pass while moving 不 between otherwise equal spans does not.
    return [
        (len(_LEXICAL_UNIT.findall(normalized[: match.start()])), match[0]) for match in _NEGATION.finditer(normalized)
    ]


def _direction_signature(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return ["buy" if match[0] in _BUY_FORMS else "sell" for match in _ACTION_DIRECTION.finditer(normalized)]


def _exact_symbols(text: str) -> Counter[str]:
    return Counter(
        [
            *_DOTTED_TICKER.findall(text),
            *_SINGLE_FINANCIAL_TICKER.findall(text),
            *_CURRENCY_AMOUNT.findall(text),
        ]
    )


def _is_predominantly_cjk(text: str) -> bool:
    non_space = len(re.sub(r"\s+", "", text))
    return non_space > 0 and len(_CJK.findall(text)) >= 0.5 * non_space


def _cjk_retention_broken(original: str, fixed: str) -> bool:
    if not _is_predominantly_cjk(original):
        return False
    return len(_CJK.findall(fixed)) < _CJK_RETENTION_FLOOR * len(_CJK.findall(original))


def _cjk_similarity_broken(original: str, fixed: str) -> bool:
    if not _is_predominantly_cjk(original) or len(_CJK.findall(original)) < _CJK_RATIO_MIN_CHARS:
        return False
    return SequenceMatcher(None, original, fixed).ratio() < _CJK_MIN_SIMILARITY
