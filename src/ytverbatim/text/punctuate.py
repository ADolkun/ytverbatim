"""LLM pass that restores punctuation where an ASR stretch has none.

Whisper large-v3 often returns Mandarin with no punctuation at all (sometimes with a space at each pause), and English
with an unpunctuated, lowercase first stretch. Stretches that already carry enough sentence-final marks pass through
untouched. For the rest, the LLM may add or replace punctuation, turn pause spaces into punctuation and capitalise
English sentence starts; it may not add, drop or change a letter, CJK character or digit. `guard_violation` enforces
that on every reply: a violating item is retried once with the diff, and when it fails again the original text is kept
and its index is reported in `PunctuationResult.failed`. A reply that is not the expected JSON raises `LLMOutputError`
rather than shipping unpunctuated text as if it had been handled.

Each batch that passes the guard is checkpointed, so a resume never pays twice for a finished batch.
"""

import hashlib
import json
import logging
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ytverbatim import atomic
from ytverbatim.contract.models import Edit, LLMUsage
from ytverbatim.errors import LLMOutputError
from ytverbatim.llm.jsonio import complete_json
from ytverbatim.llm.protocol import LLMClient, LLMRequest
from ytverbatim.llm.usage import UsageMeter
from ytverbatim.text.segment import is_punctuated, uses_zh_rules

_log = logging.getLogger(__name__)

# Part of every checkpoint and stage key: bump it on any prompt, guard or logic change so stored results are redone.
PROMPT_VERSION = "punctuation-v2"
BATCH_CHARS = 1500
STAGE = "punctuation"

# Replies echo their input, and the claude CLI also counts thinking tokens here.
_MAX_OUTPUT_TOKENS = 8192
# Marks that are data in a transcript (S&P, 3.5%, #1) and so count as content for the guard.
_DATA_MARKS = frozenset("%&#@")
_MAX_DIFF_PARTS = 6
_DIFF_CONTEXT = 12

Style = Literal["zh", "en"]

_RULES = (
    "Keep the SAME language and script (never translate, never convert between Traditional and Simplified). "
    "Keep every letter, CJK character and digit exactly as given, in the same order: never add, drop, repeat, reorder "
    "or change one, and do not fix mis-recognised words. Keep tickers and numbers intact (S&P 500, 3.5%, NVDA). "
    "Never add, remove or change punctuation inside or next to a number (1,000, 3.5, -5%, 1/2, 3:30, 10-K stay "
    "exactly as given). Never lowercase a letter. "
    "Never put quotation marks around arbitrary text. Never merge or split segments; every segment keeps its index. "
)
_STYLE_RULES: dict[Style, str] = {
    "zh": (
        "Use full-width Chinese punctuation: ，。？！；：、 . A space inside the text marks a pause: replace it with "
        "the punctuation that fits (usually ， or 。) and leave no pause spaces. A space between Latin words or "
        "numbers stays. Leave the case of Latin letters as given. "
    ),
    "en": (
        "Use ASCII punctuation. Capitalise the first letter of each sentence, and proper nouns and acronyms that are "
        "unmistakable (the I pronoun, NVDA, Fed); change nothing else about the letters. "
    ),
}
_SYSTEM = (
    "You restore punctuation in speech-to-text segments that have little or none. You are given numbered segments "
    "and return each one with sentence punctuation added so it reads naturally: periods or the Chinese equivalent at "
    "sentence ends, commas at clause breaks, question and exclamation marks where the speech asks for them. " + _RULES
)
_SCHEMA_HINT = 'Return JSON {"items": [{"i": <index>, "text": "<punctuated text>"}]} with EVERY segment given.'


class _Item(BaseModel):
    model_config = ConfigDict(strict=True)
    i: int
    text: str


class _Items(BaseModel):
    model_config = ConfigDict(strict=True)
    items: list[_Item]


class _Checkpoint(BaseModel):
    """A batch's guarded reply and what it cost, so a reused batch still counts toward the document's LLM usage."""

    items: list[_Item]
    usage: tuple[LLMUsage, ...]


@dataclass(frozen=True, slots=True)
class PunctuationResult:
    """Results aligned with the input; ``failed`` lists the indices whose original text was kept unpunctuated."""

    texts: tuple[str, ...]
    edits: tuple[tuple[Edit, ...], ...]
    failed: tuple[int, ...]
    usage: tuple[LLMUsage, ...]


@dataclass(frozen=True, slots=True)
class _Rejected:
    reply: str
    problem: str


def needs_punctuation(text: str, language: str) -> bool:
    return bool(text.strip()) and not is_punctuated(text, language)


@dataclass(frozen=True, slots=True)
class _Reduced:
    skeleton: str
    # The skeleton before lowercasing, so a letter the reply lowercased can be told from one it capitalised.
    cased: str
    # The text, and for each skeleton character where it came from in it, so a diff can quote readable text.
    source: str
    positions: tuple[int, ...]

    def quote(self, start: int, end: int) -> str:
        if end <= start:
            return ""
        first, last = self.positions[start], self.positions[end - 1] + 1
        # Whole Latin words read better than the fragment a character diff lands on; CJK is quoted per character.
        while first > 0 and _is_latin_word_char(self.source[first - 1]) and _is_latin_word_char(self.source[first]):
            first -= 1
        while (
            last < len(self.source)
            and _is_latin_word_char(self.source[last])
            and _is_latin_word_char(self.source[last - 1])
        ):
            last += 1
        return self.source[first:last]

    def lead(self, start: int) -> str:
        stop = self.positions[start - 1] + 1 if start > 0 else 0
        return self.source[max(0, stop - _DIFF_CONTEXT) : stop].strip()


def _is_latin_word_char(char: str) -> bool:
    return char.isalnum() and not "\u3400" <= char <= "\u9fff"


def skeleton(text: str) -> str:
    """Reduce `text` to its lowercased letters, CJK characters, digits and data marks, dropping punctuation and spaces.

    Punctuation that belongs to a number stays, exactly as written: any mark before a digit (so between two digits too,
    and a sign), and a mark that joins a digit to a following Latin letter or digit (10-K). 1,000 → 1，000 or 3.5 → 35
    therefore changes the skeleton, while a period or comma after a number at a clause end does not.
    """
    return _reduce(text).skeleton


def _numeric_mark(text: str, position: int) -> bool:
    after = text[position + 1] if position + 1 < len(text) else ""
    before = text[position - 1] if position > 0 else ""
    return after.isdigit() or (before.isdigit() and after.isascii() and after.isalnum())


def _reduce(text: str) -> _Reduced:
    kept: list[str] = []
    cased: list[str] = []
    positions: list[int] = []
    for position, raw in enumerate(text):
        if raw.isspace():
            continue
        if not raw.isalnum() and _numeric_mark(text, position):
            # Raw, not NFKC-folded: a full-width ，between digits is not the , it would fold to.
            kept.append(raw)
            cased.append(raw)
            positions.append(position)
            continue
        char = unicodedata.normalize("NFKC", raw)
        if all(unicodedata.category(c).startswith("P") and c not in _DATA_MARKS for c in char):
            continue
        for c in char:
            if c.isspace() or (unicodedata.category(c).startswith("P") and c not in _DATA_MARKS):
                continue
            lowered = c.lower()
            kept.append(lowered)
            cased.append(c if len(lowered) == 1 else lowered)
            positions.extend([position] * len(lowered))
            if len(lowered) > 1:
                cased.extend(lowered[1:])
    return _Reduced("".join(kept), "".join(cased), text, tuple(positions))


def guard_violation(before: str, after: str) -> str | None:
    """Describe how `after` changes more than punctuation, spacing and capitalisation, or None when it does not.

    Letters may only be capitalised, never lowercased, and punctuation that belongs to a number must stay as written.
    """
    old, new = _reduce(before), _reduce(after)
    if old.skeleton == new.skeleton:
        lowered = [i for i, (a, b) in enumerate(zip(old.cased, new.cased, strict=True)) if a != b and a.isupper()]
        if not lowered:
            return None
        runs: list[list[int]] = []
        for i in lowered:
            if runs and runs[-1][1] == i:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
        return "lowercased " + ", ".join(f'"{old.quote(a, b)}"' for a, b in runs[:_MAX_DIFF_PARTS])
    parts: list[str] = []
    for tag, a1, a2, b1, b2 in SequenceMatcher(None, old.skeleton, new.skeleton, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        lead = old.lead(a1)
        place = f' after "{lead}"' if lead else " at the start"
        match tag:
            case "delete":
                parts.append(f'dropped "{old.quote(a1, a2)}"{place}')
            case "insert":
                parts.append(f'added "{new.quote(b1, b2)}"{place}')
            case _:
                parts.append(f'changed "{old.quote(a1, a2)}" to "{new.quote(b1, b2)}"{place}')
    if len(parts) > _MAX_DIFF_PARTS:
        parts = [*parts[:_MAX_DIFF_PARTS], f"and {len(parts) - _MAX_DIFF_PARTS} more differences"]
    return "; ".join(parts)


def punctuate(
    texts: Sequence[str],
    *,
    language: str,
    client: LLMClient,
    checkpoint_dir: Path,
) -> PunctuationResult:
    """Restore punctuation in the stretches of ``texts`` that lack it; the rest pass through unchanged."""
    run = _Run(texts, client=client, checkpoint_dir=checkpoint_dir)
    pending: dict[Style, list[int]] = {"zh": [], "en": []}
    for i, text in enumerate(texts):
        if needs_punctuation(text, language):
            pending["zh" if uses_zh_rules(text, language) else "en"].append(i)
    for style, indices in pending.items():
        for batch in _batches(texts, indices, BATCH_CHARS):
            run.punctuate_batch(style, batch)
    result = PunctuationResult(
        texts=tuple(run.texts),
        edits=tuple(tuple(edits) for edits in run.edits),
        failed=tuple(sorted(run.failed)),
        usage=run.meter.summary(),
    )
    _log.info(
        "punctuation: %d of %d stretches needed it, %d edited, %d kept unpunctuated",
        sum(len(batch) for batch in pending.values()),
        len(texts),
        sum(1 for edits in result.edits if edits),
        len(result.failed),
    )
    return result


class _Run:
    def __init__(self, texts: Sequence[str], *, client: LLMClient, checkpoint_dir: Path) -> None:
        self.texts = list(texts)
        self.edits: list[list[Edit]] = [[] for _ in texts]
        self.failed: list[int] = []
        self.meter = UsageMeter()
        self._client = client
        self._checkpoint_dir = checkpoint_dir

    def punctuate_batch(self, style: Style, indices: list[int]) -> None:
        # Batch-local indices keep the checkpoint key independent of where the batch sits in the document.
        originals = [self.texts[i] for i in indices]
        path = self._checkpoint_path(style, originals)
        loaded = self._load(path, originals)
        if loaded is not None:
            outputs, usage = loaded
        else:
            batch = UsageMeter()
            outputs = self._ask_with_retry(style, originals, batch)
            usage = batch.summary()
            # A batch with a kept-original item is not cached, so a resume gets another try at it.
            if all(text is not None for text in outputs):
                self._save(path, [text or "" for text in outputs], usage)
        self.meter.merge(usage)
        for index, original, output in zip(indices, originals, outputs, strict=True):
            if output is None:
                self.failed.append(index)
            elif output != original:
                self.texts[index] = output
                self.edits[index].append(Edit(stage=STAGE, before=original, after=output, llm=self._client.id))

    def _ask_with_retry(self, style: Style, originals: list[str], meter: UsageMeter) -> list[str | None]:
        outputs: list[str | None] = [None] * len(originals)
        rejected = self._guarded(style, originals, list(range(len(originals))), None, outputs, meter)
        if rejected:
            rejected = self._guarded(style, originals, sorted(rejected), rejected, outputs, meter)
        return outputs

    def _guarded(
        self,
        style: Style,
        originals: list[str],
        asked: list[int],
        previous: dict[int, _Rejected] | None,
        outputs: list[str | None],
        meter: UsageMeter,
    ) -> dict[int, _Rejected]:
        """Ask for ``asked``, store the replies that pass the guard in ``outputs`` and return the ones that do not."""
        rejected: dict[int, _Rejected] = {}
        for index, text in self._ask(style, originals, asked, previous, meter).items():
            problem = guard_violation(originals[index], text)
            if problem is None:
                outputs[index] = text
            else:
                rejected[index] = _Rejected(text, problem)
        return rejected

    def _ask(
        self,
        style: Style,
        originals: list[str],
        asked: list[int],
        previous: dict[int, _Rejected] | None,
        meter: UsageMeter,
    ) -> dict[int, str]:
        client = meter.metered(STAGE, self._client)
        expected = sorted(asked)
        complaint = ""
        for _ in range(2):
            request = LLMRequest(
                system=_SYSTEM + _STYLE_RULES[style],
                user=_prompt(originals, asked, previous) + complaint,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
                temperature=0.0,
            )
            reply = complete_json(client, request, _Items)
            got = sorted(item.i for item in reply.items)
            if got == expected:
                return {item.i: item.text.strip() for item in reply.items}
            complaint = f"\n\nYour previous reply had the indices {got}; it must contain exactly {expected}, once each."
        raise LLMOutputError(f"{self._client.id} did not return exactly the requested indices {expected} twice")

    def _checkpoint_path(self, style: Style, originals: list[str]) -> Path:
        batch_text = json.dumps(_batch_items(originals, range(len(originals))), ensure_ascii=False)
        key = json.dumps([PROMPT_VERSION, self._client.id, style, batch_text], ensure_ascii=False)
        return self._checkpoint_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def _load(self, path: Path, originals: list[str]) -> tuple[list[str | None], tuple[LLMUsage, ...]] | None:
        if not path.exists():
            return None
        try:
            checkpoint = _Checkpoint.model_validate_json(path.read_bytes())
        except ValidationError:
            # Only this module writes these files, atomically; an unreadable one is stale, and redoing it is safe.
            _log.warning("punctuation: ignoring an unreadable checkpoint %s", path.name)
            return None
        items = checkpoint.items
        by_index = {item.i: item.text for item in items}
        if sorted(by_index) != list(range(len(originals))) or len(items) != len(originals):
            return None
        # The guard is re-applied so a stricter guard never trusts a reply an older one let through.
        if any(guard_violation(original, by_index[i]) is not None for i, original in enumerate(originals)):
            return None
        return [by_index[i] for i in range(len(originals))], checkpoint.usage

    def _save(self, path: Path, outputs: list[str], usage: tuple[LLMUsage, ...]) -> None:
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        items = [_Item(i=i, text=text) for i, text in enumerate(outputs)]
        atomic.write_json(path, _Checkpoint(items=items, usage=usage).model_dump(mode="json"))


def _batch_items(originals: Sequence[str], indices: Sequence[int]) -> list[dict[str, object]]:
    return [{"i": i, "text": originals[i]} for i in indices]


def _prompt(originals: list[str], asked: list[int], previous: dict[int, _Rejected] | None) -> str:
    if previous is None:
        return (
            "Restore the punctuation in these segments. "
            + _SCHEMA_HINT
            + "\n\nSEGMENTS:\n"
            + json.dumps(_batch_items(originals, asked), ensure_ascii=False)
        )
    items: list[dict[str, object]] = [
        {"i": i, "text": originals[i], "rejected_reply": previous[i].reply, "problem": previous[i].problem}
        for i in asked
    ]
    return (
        "Your punctuation for these segments was rejected because it changed letters, characters or digits, not only "
        "punctuation. Redo each one from its original `text`: touch only punctuation, spacing and sentence-initial "
        "capitals. " + _SCHEMA_HINT + "\n\nSEGMENTS:\n" + json.dumps(items, ensure_ascii=False)
    )


def _batches(texts: Sequence[str], indices: list[int], batch_chars: int) -> list[list[int]]:
    """Group indices into batches of about ``batch_chars`` characters; one oversized text is a batch on its own."""
    batches: list[list[int]] = []
    current: list[int] = []
    running = 0
    for i in indices:
        current.append(i)
        running += len(texts[i])
        if running >= batch_chars:
            batches.append(current)
            current = []
            running = 0
    if current:
        batches.append(current)
    return batches
