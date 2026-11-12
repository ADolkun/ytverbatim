"""Sentences, paragraphs and chapter indexes from timed tokens; pure functions, no I/O.

A token is the smallest timed piece of text: a Whisper word, or a character or word of a chunk whose engine gave no
timings (its times are spread over the chunk in proportion to characters, and `timed` is False). Everything downstream
works on tokens, so engines with and without word timings share one set of rules.
"""

import bisect
import difflib
import itertools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

SENTENCE_GAP_S = 0.6
PARAGRAPH_PAUSE_S = 2.0
PARAGRAPH_MAX_SENTENCES = 6
PARAGRAPH_MAX_S = 90.0

# Unpunctuated runs are cut near this many units: CJK characters (Latin words count as one) or English words.
_ZH_RUN_UNITS = 40
_EN_RUN_UNITS = 20
_RUN_WINDOW = (0.75, 1.25)
_ZH_OVERLONG_UNITS = 120
_EN_OVERLONG_UNITS = 60
# Whisper's Mandarin often has no punctuation at all; below this density a document is treated as unpunctuated.
_ZH_CHARS_PER_TERMINATOR = 80
_EN_WORDS_PER_TERMINATOR = 40

_CJK = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CLOSERS = "”’\"')）】」』》〕\\]"
_ZH_TERMINATORS = re.compile(f"[。！？；…!?;]+[{_CLOSERS}]*")
_EN_OPENERS = "\"'“‘(\\["
# Any terminator before a capital or digit ends a sentence. Before a lowercase word only a plain period, "!" or "?"
# does (Whisper lowercases whole stretches of speech); an ellipsis there is a trailing-off pause mid-sentence.
_EN_TERMINATORS = re.compile(
    f"[.!?…]+[{_CLOSERS}]*(?=\\s+[{_EN_OPENERS}]*[A-Z0-9])"
    f"|(?<![.…])(?:\\.|[!?]+)[{_CLOSERS}]*(?=\\s+[{_EN_OPENERS}]*[a-z])"
)
_EN_DENSITY_MARKS = re.compile(r"[.!?…]+(?=\s|$)")
_ZH_DENSITY_MARKS = re.compile(r"[。！？；…!?;]")
_ZH_UNITS = re.compile(f"[{_CJK}]|[A-Za-z0-9]+")
_CJK_CHAR = re.compile(f"[{_CJK}]")
_ZH_PSEUDO_TOKEN = re.compile(f"\\s*(?:[{_CJK}]|[0-9A-Za-z]+(?:[.,'’][0-9A-Za-z]+)*|\\S)")
_EN_PSEUDO_TOKEN = re.compile(r"\s*\S+")
_COMMAS = ",，、"
# What word alignment compares: a CJK character, or a run of other letters and digits (a "run").
_ALIGN_RUNS = re.compile(f"[{_CJK}]|[^\\W_{_CJK}]+")
# The longest stretch of unmatched characters, per side, that word alignment diffs character by character. Pieces of a
# split run are a few characters; this is a sentence or two, and a diff of it takes about a millisecond.
_MAX_CHARACTER_DIFF = 300
# Punctuation between two aligned words goes to the word before it, except opening marks, which lead the next word.
_OPENERS = "“‘(（「『《【〔["
# A period inside a word ("example.com") is no break. Inside a sentence range a period before a lowercase word is an
# abbreviation or an ellipsis, since every other one is already a terminator cut.
_INNER_MARKS = re.compile(r"[,，、;；:：。！？]|[.!?…](?=\s+[^a-z\s]|\s*$)")
_LEADING_PUNCTUATION = "([“\"'$"

# Lowercase, without the final period. "no" is left out on purpose: it is an ordinary word that ends sentences too.
_ABBREVIATIONS = frozenset(
    (  # noqa: SIM905
        "mr mrs ms dr prof sr jr st vs inc ltd co corp fig eg ie cf al approx est vol ch pp mt gen sen rep gov lt col "
        "capt sgt jan feb mar apr jun jul aug sep sept oct nov dec e.g i.e etc"
    ).split()
)
_DOTTED_INITIALS = re.compile(r"(?:[A-Za-z]\.)+[A-Za-z]")


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: float
    end: float
    conf: float | None = None
    timed: bool = True
    # Opaque tag of where the token came from (the chunk index); `Sentence.span_of` finds a tag's characters.
    origin: int = 0


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    start: float
    end: float
    tokens: tuple[Token, ...]
    # Offset into `text` where each token begins.
    offsets: tuple[int, ...]

    def span_of(self, origin: int) -> tuple[int, int] | None:
        """Half-open character range of the tokens tagged `origin`, or None when it has none in this sentence."""
        hits = [i for i, token in enumerate(self.tokens) if token.origin == origin]
        if not hits:
            return None
        first, last = hits[0], hits[-1]
        end = self.offsets[last + 1] if last + 1 < len(self.offsets) else len(self.text)
        start = self.offsets[first]
        covered = self.text[start:end]
        start += len(covered) - len(covered.lstrip())
        end -= len(covered) - len(covered.rstrip())
        return (start, end) if end > start else None


def is_zh(language: str) -> bool:
    return language.lower().startswith("zh")


def uses_zh_rules(text: str, language: str) -> bool:
    """Whether Chinese sentence rules apply: a zh-tagged video whose text has no CJK at all reads as English."""
    return is_zh(language) and _CJK_CHAR.search(text) is not None


def is_punctuated(text: str, language: str) -> bool:
    """Whether the text carries enough sentence-final marks to trust its own punctuation."""
    if uses_zh_rules(text, language):
        cjk = len(_CJK_CHAR.findall(text))
        return len(_ZH_DENSITY_MARKS.findall(text)) * _ZH_CHARS_PER_TERMINATOR >= cjk
    words = len(text.split())
    return words > 0 and len(_EN_DENSITY_MARKS.findall(text)) * _EN_WORDS_PER_TERMINATOR >= words


def pseudo_tokens(text: str, start: float, end: float, *, language: str, origin: int = 0) -> list[Token]:
    """Split a chunk's text into tokens whose times are proportional to characters across `[start, end]`."""
    pattern = _ZH_PSEUDO_TOKEN if is_zh(language) else _EN_PSEUDO_TOKEN
    pieces = [match.group() for match in pattern.finditer(text.strip())]
    weights = [len(piece.strip()) for piece in pieces]
    total = sum(weights)
    tokens: list[Token] = []
    done = 0
    for piece, weight in zip(pieces, weights, strict=True):
        fraction = (done / total, (done + weight) / total) if total else (0.0, 0.0)
        tokens.append(
            Token(
                piece,
                start + (end - start) * fraction[0],
                start + (end - start) * fraction[1],
                timed=False,
                origin=origin,
            )
        )
        done += weight
    return tokens


class TimedWord(Protocol):
    @property
    def text(self) -> str: ...
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...
    @property
    def conf(self) -> float | None: ...


def aligned_tokens(
    text: str, words: Sequence[TimedWord], start: float, end: float, *, language: str, origin: int = 0
) -> tuple[list[Token], bool]:
    """Tokens of `text` timed by the engine's `words` wherever a sequence diff matches them; True when all matched.

    Letters and digits are compared casefolded, one character at a time, so punctuation and where a Latin or digit run
    is split into pieces do not matter. A word matches when every one of its characters lines up with consecutive text
    without a space between Latin or digit characters; its token takes the text's spelling and the word's times,
    clamped to `[start, end]`. Pieces of one run ("X" and "LP", or "171" and "6" around a decimal point) become one
    token timed from the first piece to the last. A Latin or digit run is never cut: it is timed by words that cover
    all of it, or by none. Text that no word matches is spread over the gap between the timed words around it, and
    those tokens are not `timed`.
    """
    text = text.strip()
    text_runs = [m.span() for m in _ALIGN_RUNS.finditer(text)]
    text_units = [(i, i + 1) for a, b in text_runs for i in range(a, b)]
    word_units: list[str] = []
    word_run_sizes: list[int] = []
    word_run_of: list[int] = []
    owners: list[int] = []
    for index, word in enumerate(words):
        for run in _ALIGN_RUNS.findall(word.text):
            word_run_of += [len(word_run_sizes)] * len(run)
            word_run_sizes.append(len(run))
            word_units += (char.casefold() for char in run)
            owners += [index] * len(run)
    pairs = _matched_units(
        [text[a:b].casefold() for a, b in text_units], [b - a for a, b in text_runs], word_units, word_run_sizes
    )
    hits: dict[int, list[int]] = {}
    for text_unit, word_unit in pairs:
        hits.setdefault(owners[word_unit], []).append(text_unit)
    sizes = [0] * len(words)
    firsts = [0] * len(words)
    for position, owner in enumerate(owners):
        if not sizes[owner]:
            firsts[owner] = position
        sizes[owner] += 1
    candidates = {
        index: units
        for index, units in hits.items()
        if len(units) == sizes[index]
        and units[-1] - units[0] == len(units) - 1
        and not _spans_a_space(text, text_units, units, word_run_of[firsts[index] : firsts[index] + sizes[index]])
    }
    accepted = _whole_runs(candidates, [b - a for a, b in text_runs])

    tokens: list[Token] = []
    complete = True
    cursor = 0
    left = start
    joinable = False
    for index, word in enumerate(words):
        units = accepted.get(index)
        if not units:
            continue
        first, last = text_units[units[0]][0], text_units[units[-1]][1]
        word_start = min(max(word.start, start), end)
        word_end = min(max(word.end, word_start), end)
        if joinable and _continues_run(text, cursor, first):
            previous = tokens[-1]
            conf = [c for c in (previous.conf, word.conf) if c is not None]
            tokens[-1] = replace(
                previous,
                text=previous.text + text[cursor:last],
                end=max(previous.end, word_end),
                conf=min(conf) if conf else None,
            )
        else:
            lead, interpolated = _fill(tokens, text[cursor:first], left, word_start, language, origin)
            complete = complete and not interpolated
            tokens.append(Token(lead + text[first:last], word_start, word_end, word.conf, origin=origin))
        cursor, left, joinable = last, word_end, True
    rest, interpolated = _fill(tokens, text[cursor:], left, end, language, origin)
    if rest and tokens:
        tokens[-1] = replace(tokens[-1], text=tokens[-1].text + rest)
    elif rest:
        # Text without a letter or digit, which nothing can be matched against.
        return pseudo_tokens(rest, start, end, language=language, origin=origin), False
    return tokens, complete and not interpolated


def _spans_a_space(
    text: str, text_units: Sequence[tuple[int, int]], units: Sequence[int], word_runs: Sequence[int]
) -> bool:
    """Whether the text puts a space inside one of the word's own runs ("ETFS" over "ETF S").

    `word_runs` is the run each of the word's characters belongs to. A space where the word itself has a run boundary
    ("AT&T" over "AT & T") is only spacing.
    """
    for k, (before, after) in enumerate(itertools.pairwise(units)):
        if word_runs[k] != word_runs[k + 1]:
            continue
        gap = text[text_units[before][1] : text_units[after][0]]
        if any(char.isspace() for char in gap):
            return True
    return False


def _whole_runs(candidates: dict[int, list[int]], run_sizes: Sequence[int]) -> dict[int, list[int]]:
    """Drop every candidate word that touches a text run its accepted words do not cover whole, until none does.

    `candidates` maps a word to the text units it matched; `run_sizes` are the lengths of the text's runs in order.
    """
    run_of = [run for run, size in enumerate(run_sizes) for _ in range(size)]
    touching: dict[int, set[int]] = {}
    covered = [0] * len(run_sizes)
    for word, units in candidates.items():
        for unit in units:
            touching.setdefault(run_of[unit], set()).add(word)
            covered[run_of[unit]] += 1
    accepted = dict(candidates)
    queue = [run for run in touching if covered[run] != run_sizes[run]]
    while queue:
        run = queue.pop()
        for word in touching[run]:
            if word not in accepted:
                continue
            for unit in accepted.pop(word):
                other = run_of[unit]
                covered[other] -= 1
                if other != run and covered[other] != run_sizes[other]:
                    queue.append(other)
    return accepted


def _matched_units(
    a: Sequence[str], a_runs: Sequence[int], b: Sequence[str], b_runs: Sequence[int]
) -> list[tuple[int, int]]:
    """Index pairs of equal units in order. `a_runs` and `b_runs` are the lengths of the runs the units fall in.

    Whole runs are diffed first, which is cheap and anchors the alignment. Characters are then diffed only in the
    stretches between anchors that hold unmatched runs on both sides, so a misheard word cannot take characters from
    its neighbours. A stretch longer than `_MAX_CHARACTER_DIFF` on either side is a rewritten passage, not a split
    word, and is left unmatched: the diff is quadratic, and with no anchors a whole transcript would be one stretch.
    """
    a_at, b_at = _starts(a_runs), _starts(b_runs)
    a_keys = [tuple(a[a_at[i] : a_at[i + 1]]) for i in range(len(a_runs))]
    b_keys = [tuple(b[b_at[i] : b_at[i + 1]]) for i in range(len(b_runs))]
    pairs: list[tuple[int, int]] = []
    done_a = done_b = 0
    for block in difflib.SequenceMatcher(None, a_keys, b_keys, autojunk=False).get_matching_blocks():
        lo_a, hi_a, lo_b, hi_b = a_at[done_a], a_at[block.a], b_at[done_b], b_at[block.b]
        if 0 < hi_a - lo_a <= _MAX_CHARACTER_DIFF and 0 < hi_b - lo_b <= _MAX_CHARACTER_DIFF:
            for inner in difflib.SequenceMatcher(
                None, a[lo_a:hi_a], b[lo_b:hi_b], autojunk=False
            ).get_matching_blocks():
                pairs += ((lo_a + inner.a + k, lo_b + inner.b + k) for k in range(inner.size))
        length = a_at[block.a + block.size] - a_at[block.a]
        pairs += ((a_at[block.a] + k, b_at[block.b] + k) for k in range(length))
        done_a, done_b = block.a + block.size, block.b + block.size
    return pairs


def _starts(lengths: Sequence[int]) -> list[int]:
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    return starts


def _continues_run(text: str, cursor: int, first: int) -> bool:
    """Whether the two matched pieces around `text[cursor:first]` are parts of one run.

    That holds with nothing between them, a point or comma between digits, or an apostrophe between letters.
    """
    before, after = text[cursor - 1], text[first]
    if _CJK_CHAR.match(before) or _CJK_CHAR.match(after):
        return False
    gap = text[cursor:first]
    if not gap:
        return True
    if gap in (".", ","):
        return before.isdigit() and after.isdigit()
    return gap in ("'", "’") and before.isalpha() and after.isalpha()


def _fill(tokens: list[Token], stretch: str, left: float, right: float, language: str, origin: int) -> tuple[str, bool]:
    """Place the text between two matched words; return what leads the next word and whether anything was timed."""
    if not any(char.isalnum() for char in stretch):
        space = next((i for i, char in enumerate(stretch) if char.isspace()), None)
        split = space if space is not None else len(stretch.rstrip(_OPENERS))
        if not tokens:
            return stretch, False
        tokens[-1] = replace(tokens[-1], text=tokens[-1].text + stretch[:split])
        return stretch[split:], False
    head = stretch[: len(stretch) - len(stretch.lstrip())]
    piece = pseudo_tokens(stretch, left, max(right, left), language=language, origin=origin)
    piece[0] = replace(piece[0], text=head + piece[0].text)
    tokens += piece
    return stretch[len(stretch.rstrip()) :], True


def segment_tokens(tokens: Sequence[Token], language: str) -> list[Sentence]:
    """Group tokens into sentences.

    Sentence-final punctuation decides first. Unpunctuated text (a whole document, or a long stretch with no marks)
    is also cut at pauses of 0.6 s and at about 40 CJK units or 20 English words. Whatever is still longer than 120
    CJK units or 60 words is split at the comma nearest its middle.
    """
    kept = [token for token in tokens if token.text.strip()]
    if not kept:
        return []
    text = "".join(token.text for token in kept)
    cutter = _Cutter(kept, zh=uses_zh_rules(text, language))
    cuts = cutter.terminator_cuts(text)
    ranges = _ranges(len(kept), cuts)
    unpunctuated = not is_punctuated(text, language)
    ranges = [
        piece for r in ranges for piece in (cutter.unpunctuated(*r) if unpunctuated or cutter.sparse(*r) else [r])
    ]
    ranges = [piece for r in ranges for piece in cutter.overlong(*r)]
    return [_sentence(kept[a:b]) for a, b in _attach_bare_punctuation(ranges, cutter)]


def split_paragraphs(
    starts: Sequence[float],
    ends: Sequence[float],
    chapters: Sequence[int | None],
    *,
    pause_s: float = PARAGRAPH_PAUSE_S,
    max_sentences: int = PARAGRAPH_MAX_SENTENCES,
    max_seconds: float = PARAGRAPH_MAX_S,
) -> list[tuple[int, int]]:
    """Inclusive sentence index ranges. A paragraph ends at a pause, at a size limit, or at a chapter boundary."""
    ranges: list[tuple[int, int]] = []
    first = 0
    for i in range(1, len(starts)):
        pause = starts[i] - ends[i - 1]
        if (
            pause >= pause_s
            or i - first >= max_sentences
            or ends[i - 1] - starts[first] >= max_seconds
            or chapters[i] != chapters[i - 1]
        ):
            ranges.append((first, i - 1))
            first = i
    if starts:
        ranges.append((first, len(starts) - 1))
    return ranges


def chapter_indexes(starts: Sequence[float], chapter_starts: Sequence[float]) -> list[int]:
    """Index of the last chapter that began at or before each start; times before the first chapter get chapter 0."""
    return [max(bisect.bisect_right(chapter_starts, start) - 1, 0) for start in starts]


def _sentence(tokens: Sequence[Token]) -> Sentence:
    raw = "".join(token.text for token in tokens)
    lead = len(raw) - len(raw.lstrip())
    text = raw.strip()
    offsets: list[int] = []
    position = 0
    for token in tokens:
        offsets.append(min(max(position - lead, 0), len(text)))
        position += len(token.text)
    return Sentence(text, tokens[0].start, max(token.end for token in tokens), tuple(tokens), tuple(offsets))


def _ranges(count: int, cuts: set[int]) -> list[tuple[int, int]]:
    """Half-open token ranges; a cut at i ends a range after token i."""
    ranges: list[tuple[int, int]] = []
    first = 0
    for i in sorted(cuts):
        if i + 1 < count:
            ranges.append((first, i + 1))
            first = i + 1
    ranges.append((first, count))
    return ranges


def _attach_bare_punctuation(ranges: list[tuple[int, int]], cutter: "_Cutter") -> list[tuple[int, int]]:
    """Merge ranges with nothing to read (a lone "。" after a pause) into their neighbour."""
    merged: list[tuple[int, int]] = []
    carry: int | None = None
    for a, b in ranges:
        if carry is not None:
            a, carry = carry, None
        if cutter.units(a, b) == 0:
            if merged:
                merged[-1] = (merged[-1][0], b)
            else:
                carry = a
            continue
        merged.append((a, b))
    return merged


class _Cutter:
    def __init__(self, tokens: Sequence[Token], *, zh: bool) -> None:
        self._tokens = tokens
        self._zh = zh
        count: Callable[[str], int] = _zh_units if self._zh else _en_units
        self._prefix = [0]
        for token in tokens:
            self._prefix.append(self._prefix[-1] + count(token.text))
        self._bounds: list[int] = []
        position = 0
        for token in tokens:
            position += len(token.text)
            self._bounds.append(position)
        self._run_units = _ZH_RUN_UNITS if self._zh else _EN_RUN_UNITS
        self._overlong_units = _ZH_OVERLONG_UNITS if self._zh else _EN_OVERLONG_UNITS

    def units(self, a: int, b: int) -> int:
        return self._prefix[b] - self._prefix[a]

    def gap(self, i: int) -> float:
        """Silence after token i; never negative, because timed tokens may overlap slightly."""
        if i + 1 >= len(self._tokens):
            return 0.0
        return max(self._tokens[i + 1].start - self._tokens[i].end, 0.0)

    def terminator_cuts(self, text: str) -> set[int]:
        pattern = _ZH_TERMINATORS if self._zh else _EN_TERMINATORS
        cuts: set[int] = set()
        for match in pattern.finditer(text):
            if not self._zh and _is_abbreviation(text, match):
                continue
            cuts.add(min(bisect.bisect_left(self._bounds, match.end()), len(self._tokens) - 1))
        return cuts

    def sparse(self, a: int, b: int) -> bool:
        """Whether a long stretch has no punctuation at all inside, as Whisper leaves the start of some videos."""
        if self.units(a, b) <= round(self._run_units * _RUN_WINDOW[1]):
            return False
        inner = "".join(token.text for token in self._tokens[a:b])[:-1]
        return _INNER_MARKS.search(inner) is None

    def unpunctuated(self, a: int, b: int) -> list[tuple[int, int]]:
        pieces: list[tuple[int, int]] = []
        first = a
        for i in range(a, b - 1):
            if self.gap(i) >= SENTENCE_GAP_S:
                pieces.append((first, i + 1))
                first = i + 1
        pieces.append((first, b))
        return [part for piece in pieces for part in self._peel(*piece)]

    def _peel(self, a: int, b: int) -> list[tuple[int, int]]:
        """Cut a long unpunctuated run near the target size, preferring the longest pause in the window.

        A single token over the cap cannot be cut and stays a part of its own.
        """
        low, high = (round(self._run_units * f) for f in _RUN_WINDOW)
        parts: list[tuple[int, int]] = []
        first = a
        while b - first > 1 and self.units(first, b) > high:
            window = [i for i in range(first, b - 1) if low <= self.units(first, i + 1) <= high]
            if window:
                cut = max(window, key=lambda i, f=first: self._run_cut_rank(f, i))
            else:
                cut = next((i for i in range(first, b - 1) if self.units(first, i + 1) >= self._run_units), b - 2)
                if cut > first and self.units(cut, cut + 1) > high:
                    cut -= 1
            parts.append((first, cut + 1))
            first = cut + 1
        parts.append((first, b))
        return parts

    def _run_cut_rank(self, first: int, i: int) -> tuple[bool, float, float]:
        """Prefer a comma, then the longest pause, then the cut nearest the target size."""
        return (
            self._ends_with_comma(i),
            self.gap(i),
            -abs(self.units(first, i + 1) - self._run_units),
        )

    def _ends_with_comma(self, i: int) -> bool:
        return self._tokens[i].text.rstrip().endswith(tuple(_COMMAS))

    def overlong(self, a: int, b: int) -> list[tuple[int, int]]:
        if self.units(a, b) <= self._overlong_units or b - a < 2:  # noqa: PLR2004
            return [(a, b)]
        cut = self._comma_cut(a, b)
        if cut is None:
            cut = self._middle_cut(a, b)
        return [*self.overlong(a, cut + 1), *self.overlong(cut + 1, b)]

    def _comma_cut(self, a: int, b: int) -> int | None:
        commas = [i for i in range(a, b - 1) if self._ends_with_comma(i)]
        return min(commas, key=lambda i: self._distance_from_middle(a, b, i)) if commas else None

    def _middle_cut(self, a: int, b: int) -> int:
        """Without a comma, cut in the middle half of the sentence at its longest pause."""
        total = self.units(a, b)
        middle = [i for i in range(a, b - 1) if total * 0.25 <= self.units(a, i + 1) <= total * 0.75]
        if not middle:
            return min(range(a, b - 1), key=lambda i: self._distance_from_middle(a, b, i))
        return max(middle, key=lambda i: (self.gap(i), -self._distance_from_middle(a, b, i)))

    def _distance_from_middle(self, a: int, b: int, i: int) -> float:
        return abs(self.units(a, i + 1) - self.units(a, b) / 2)


def _zh_units(text: str) -> int:
    return len(_ZH_UNITS.findall(text))


def _en_units(text: str) -> int:
    return 1 if any(char.isalnum() for char in text) else 0


def _is_abbreviation(text: str, match: re.Match[str]) -> bool:
    if match.group().rstrip(_CLOSERS) != ".":
        return False
    before = text[: match.start()].split()
    if not before:
        return False
    word = before[-1].lstrip(_LEADING_PUNCTUATION)
    if word.lower() in _ABBREVIATIONS or _DOTTED_INITIALS.fullmatch(word):
        return True
    # "No. 5" is a number sign, while "I said no. Then" ends a sentence.
    if word.lower() == "no":
        return text[match.end() :].lstrip()[:1].isdigit()
    # A lone capital or a ticker ("Plan B.", "AAPL.") ends a sentence; an initial ("John F. Kennedy") is the price.
    return False
