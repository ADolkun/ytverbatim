"""Move character offsets from a text to its edited version, and describe an edit as small replaced runs.

Both work from a diff of the two texts rather than their common prefix and suffix, so an edit that touches several
places (punctuation inserted throughout a stretch, two fillers removed) moves an offset between those places exactly.
"""

import re
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from ytverbatim.contract.models import Flag

# A Latin word or number, a run of spaces, or any other single character (CJK, punctuation). CJK: Extension A, the
# unified block, the compatibility ideographs, and Extension B onward with the compatibility supplement.
_CJK = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f"
_UNIT = re.compile(rf"[^\W_{_CJK}]+|\s+|.", re.DOTALL)


@dataclass(frozen=True, slots=True)
class OffsetMap:
    """Map offsets in ``before`` to ``after``; text inserted at a span's edge stays outside that span."""

    _opcodes: tuple[tuple[str, int, int, int, int], ...]
    _starts: tuple[int, ...]

    @classmethod
    def between(cls, before: str, after: str) -> "OffsetMap":
        ops = tuple(
            op
            for op in SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
            # Insertions cover no offset of `before`; their effect shows in the next opcode's target start.
            if op[1] != op[2]
        )
        return cls(ops, tuple(op[1] for op in ops))

    def start(self, offset: int) -> int:
        """Where a span starting at ``offset`` starts after the edit."""
        index = bisect_right(self._starts, offset) - 1
        if index < 0:
            return offset
        tag, i1, i2, j1, j2 = self._opcodes[index]
        if offset >= i2:
            return j2
        return j1 + offset - i1 if tag == "equal" else j1

    def end(self, offset: int) -> int:
        """Where a span ending at ``offset`` (exclusive) ends after the edit."""
        index = bisect_right(self._starts, offset - 1) - 1
        if index < 0:
            return offset
        tag, i1, i2, j1, j2 = self._opcodes[index]
        if offset > i2:
            return j2
        return j1 + offset - i1 if tag == "equal" else j2


def remap_flags(flags: Sequence[Flag], changes: Iterable[tuple[str, str]]) -> tuple[Flag, ...]:
    """Move ``flags`` through successive (before, after) versions of their text; a flag whose text is gone is dropped.

    A flag overlapping a replaced run stretches over the replacement.
    """
    moved = list(flags)
    for before, after in changes:
        if not moved:
            break
        mapping = OffsetMap.between(before, after)
        moved = [
            flag.model_copy(update={"char_start": start, "char_end": end})
            for flag in moved
            if (end := mapping.end(flag.char_end)) > (start := mapping.start(flag.char_start))
        ]
    return tuple(moved)


@dataclass(frozen=True, slots=True)
class ChangedRun:
    """One replaced run of an edit: ``before`` became ``after``, which sits at ``[start, end)`` of the new text."""

    start: int
    end: int
    before: str
    after: str


def changed_runs(before: str, after: str) -> list[ChangedRun]:
    """Split the edit into the runs of whole words (CJK: characters) that changed, in order.

    Two word-changing runs one unit apart are one change: a diff writes 十五→五十 as an insert and a delete around
    the shared 十, which no reader would call two edits.
    """
    old = _UNIT.findall(before)
    new = _UNIT.findall(after)
    spans: list[list[int]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if spans and _wordy(old, new, spans[-1]) and _wordy(old, new, [i1, i2, j1, j2]):
            previous = spans[-1]
            if i1 - previous[1] <= 1 and j1 - previous[3] <= 1:
                spans[-1] = [previous[0], i2, previous[2], j2]
                continue
        spans.append([i1, i2, j1, j2])
    positions = [0]
    for unit in new:
        positions.append(positions[-1] + len(unit))
    return [
        ChangedRun(positions[j1], positions[j2], "".join(old[i1:i2]), "".join(new[j1:j2])) for i1, i2, j1, j2 in spans
    ]


def _wordy(old: list[str], new: list[str], span: list[int]) -> bool:
    i1, i2, j1, j2 = span
    return any(char.isalnum() for unit in (*old[i1:i2], *new[j1:j2]) for char in unit)
