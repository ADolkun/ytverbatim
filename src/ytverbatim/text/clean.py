"""Deterministic text normalization: Simplified script, optional filler removal and punctuation tidying.

The filler and punctuation rules are ported from ytscript's cleaner. Every rule is conservative: English drops only
pure hesitation sounds, never discourse markers, and Chinese drops a filler only where its boundaries show a
disfluency. A rule that would empty a text keeps the original instead, so no stretch loses its time range.

`normalize` returns each single replacement as a (before, after) step of the whole text, so flag offsets can be
moved through every step exactly rather than across one coarse span.
"""

import functools
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import opencc

from ytverbatim.text.segment import is_zh

Script = Literal["zh-Hans", "keep"]
Fillers = Literal["keep", "remove"]

# "er" and its drawn-out forms ("errr") are hesitations; "err", the verb, is a word and stays.
_EN_FILLERS = re.compile(r"\b(?:um+|uh+|hmm+|hm+|er(?:rr+)?)\b", re.IGNORECASE)
_ZH_FILLER_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"就是说就是说"), "就是说"),
    (re.compile(r"然后然后"), "然后"),
    (re.compile(r"(?:^|(?<=[，。！？、\s]))嗯+(?=[，。！？、\s]|$)"), ""),
    (re.compile(r"(?:^|(?<=[，。！？、\s]))啊+(?=[，。！？、\s]|$)"), ""),
)
# Demonstratives (这个/那个) may refer to source material even at a boundary, so they are never fillers.
_PUNCTUATION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # A clause never opens on a comma. Trailing marks stay: Whisper ends each Mandarin clause with its boundary.
    (re.compile(r"^[，,、]+\s*"), ""),
    (re.compile(r"\.{2,}"), "."),
    (re.compile(r",{2,}"), ","),
    (re.compile(r"。{2,}"), "。"),
    (re.compile(r"，{2,}"), "，"),
)
_SPACES = re.compile(r"\s{2,}")
_REPEATED_COMMA = re.compile(r"([,，])(?:\s*[,，])+")


@dataclass(frozen=True, slots=True)
class Normalized:
    text: str
    # Successive whole-text states, one per replacement, in order: steps[0][0] is the input.
    steps: tuple[tuple[str, str], ...]


@functools.cache
def _t2s() -> Callable[[str], str]:
    converter = opencc.OpenCC("t2s")

    def convert(text: str) -> str:
        return str(converter.convert(text))

    return convert


def to_simplified(text: str) -> str:
    return _t2s()(text)


def normalize(text: str, language: str, *, script: Script, fillers: Fillers) -> Normalized:
    steps: list[tuple[str, str]] = []
    current = text

    def step(after: str) -> None:
        nonlocal current
        if after != current:
            steps.append((current, after))
            current = after

    def apply(pattern: re.Pattern[str], replace: str | Callable[[re.Match[str]], str]) -> None:
        # One match at a time, so each step is a single contiguous replacement.
        position = 0
        while (match := pattern.search(current, position)) is not None:
            after = replace if isinstance(replace, str) else replace(match)
            step(current[: match.start()] + after + current[match.end() :])
            position = match.start() + len(after)
            if match.end() == match.start() and not after:
                position += 1

    if is_zh(language) and script == "zh-Hans":
        step(to_simplified(current))
    if fillers == "remove":
        if is_zh(language):
            for pattern, replacement in _ZH_FILLER_RULES:
                apply(pattern, replacement)
        else:
            # A capitalised hesitation ("Um,") still goes; an all-caps one is more likely an acronym ("UM", "ER").
            apply(_EN_FILLERS, lambda match: match[0] if match[0].isupper() else "")
        # "I think, um, that" would otherwise keep both commas around the removed filler.
        apply(_REPEATED_COMMA, lambda match: match[1])
        apply(_SPACES, " ")
    for pattern, replacement in _PUNCTUATION_RULES:
        apply(pattern, replacement)
    step(current.strip())
    if not current:
        return Normalized(text, ())
    return Normalized(current, tuple(steps))
