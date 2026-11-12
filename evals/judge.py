"""The blind text-only LLM judge that screens challengers; humans make the final call in the A/B session.

Blinding: the judge sees each span's distinct readings in a seeded random order, never an engine id or how many
engines produced a reading, and a context made only of units every engine agrees on (``[?]`` marks other disputes).
Verdicts are cached on disk per span, keyed by the prompt version, the model and exactly what the judge saw, so a
run interrupted by quota resumes where it stopped.
"""

import hashlib
import json
import logging
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ytverbatim import atomic
from ytverbatim.errors import LLMOutputError
from ytverbatim.llm import LLMClient, LLMRequest, LLMResponse, complete_json

log = logging.getLogger(__name__)

PROMPT_VERSION = "judge-v1"
JUDGE_MODEL = "claude-opus-5-5"
BATCH_SIZE = 25
NEEDS_AUDIO = "needs_audio"
EMPTY_READING = "(nothing)"
VERDICTS_FILE = "verdicts.jsonl"
_MAX_OUTPUT_TOKENS = 8192

SYSTEM = f"""You review a speech transcript where automatic transcribers disagreed.
Each span shows the text around one stretch of speech, with the stretch marked ⟪?⟫, and two or more candidate \
readings of that stretch in random order.
Pick the reading a careful listener most likely heard, judging by fluency, grammar, meaning and fit with the \
surrounding text.
Readings are normalized: punctuation is removed and Latin letters keep their case. Prefer correct capitalization of \
names, tickers and acronyms. {EMPTY_READING} means the stretch holds no words.
[?] in the surrounding text is another uncertain stretch; it is not evidence for or against any reading.
If the text cannot decide between readings that differ only in how they sound, answer "{NEEDS_AUDIO}".
Give a reason of at most fifteen words."""


@dataclass(frozen=True)
class JudgeItem:
    span_id: str
    left: str
    right: str
    options: tuple[str, ...]

    def key(self, model: str) -> str:
        payload = [PROMPT_VERSION, model, SYSTEM, self.left, self.right, list(self.options)]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    span: str
    choice: int | Literal["needs_audio"] = Field(description="1-based index of the most plausible reading")
    reason: str


class Decisions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[Decision]


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    choice: str | None
    needs_audio: bool
    reason: str
    model: str
    prompt_version: str


def make_item(span_id: str, left: str, right: str, readings: Iterable[str], seed: int) -> JudgeItem:
    """Distinct readings in an order fixed by ``seed`` and the span alone, so reruns show the same prompt."""
    options = sorted(dict.fromkeys(readings))
    random.Random(f"{seed}:{span_id}").shuffle(options)
    return JudgeItem(span_id, left, right, tuple(options))


def render_batch(items: Sequence[JudgeItem]) -> str:
    blocks = []
    for number, item in enumerate(items, start=1):
        lines = [f"[s{number}] {item.left} ⟪?⟫ {item.right}".rstrip()]
        lines.extend(f"  {index}. {option or EMPTY_READING}" for index, option in enumerate(item.options, start=1))
        blocks.append("\n".join(lines))
    return (
        'Answer every span. Use its label without brackets ("s1") as "span", and the reading number as "choice".\n\n'
        + "\n\n".join(blocks)
    )


def load_verdicts(directory: Path) -> dict[str, Verdict]:
    path = directory / VERDICTS_FILE
    if not path.exists():
        return {}
    verdicts = (Verdict.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {v.key: v for v in verdicts}


class _Recorder:
    """Keeps every raw reply so a surprising verdict can be traced back to what the model actually wrote."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.replies: list[str] = []

    @property
    def id(self) -> str:
        return self._inner.id

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._inner.complete(request)
        self.replies.append(response.text)
        return response


def run_judge(
    client: LLMClient,
    items: Sequence[JudgeItem],
    directory: Path,
    *,
    model: str,
    batch_size: int = BATCH_SIZE,
    max_calls: int | None = None,
    progress: Callable[[str], None] = log.info,
) -> dict[str, Verdict]:
    """Return verdicts keyed by span id, calling the judge only for spans not already cached.

    Verdicts are appended after every call, so an interruption loses at most one batch.
    """
    directory.mkdir(parents=True, exist_ok=True)
    cached = load_verdicts(directory)
    pending = [item for item in items if item.key(model) not in cached]
    for calls, start in enumerate(range(0, len(pending), batch_size)):
        if max_calls is not None and calls >= max_calls:
            progress(f"stopped after {calls} judge calls; {len(pending) - start} spans remain for the next run")
            break
        batch = pending[start : start + batch_size]
        verdicts = _judge_batch(client, batch, directory, model)
        with (directory / VERDICTS_FILE).open("a", encoding="utf-8") as handle:
            handle.writelines(v.model_dump_json() + "\n" for v in verdicts)
        cached.update((v.key, v) for v in verdicts)
        progress(f"judge call {calls + 1}: {min(start + batch_size, len(pending))}/{len(pending)} uncached spans done")
    return {item.span_id: cached[item.key(model)] for item in items if item.key(model) in cached}


def _judge_batch(client: LLMClient, batch: Sequence[JudgeItem], directory: Path, model: str) -> list[Verdict]:
    user = render_batch(batch)
    recorder = _Recorder(client)
    request = LLMRequest(system=SYSTEM, user=user, max_output_tokens=_MAX_OUTPUT_TOKENS)
    decisions = complete_json(recorder, request, Decisions).decisions
    digest = hashlib.sha256(user.encode("utf-8")).hexdigest()[:16]
    raw_dir = directory / "raw"
    raw_dir.mkdir(exist_ok=True)
    atomic.write_json(raw_dir / f"{digest}.json", {"user": user, "replies": recorder.replies})
    return _verdicts(batch, decisions, model)


def _verdicts(batch: Sequence[JudgeItem], decisions: Sequence[Decision], model: str) -> list[Verdict]:
    by_label: Mapping[str, Decision] = {d.span.strip("[] "): d for d in decisions}
    expected = {f"s{n}" for n in range(1, len(batch) + 1)}
    if set(by_label) != expected or len(decisions) != len(batch):
        raise LLMOutputError(f"the judge answered spans {sorted(by_label)} instead of {sorted(expected)}")
    out = []
    for number, item in enumerate(batch, start=1):
        decision = by_label[f"s{number}"]
        if decision.choice == NEEDS_AUDIO:
            choice, needs_audio = None, True
        elif isinstance(decision.choice, int) and 1 <= decision.choice <= len(item.options):
            choice, needs_audio = item.options[decision.choice - 1], False
        else:
            raise LLMOutputError(f"the judge chose {decision.choice!r} for a span with {len(item.options)} readings")
        out.append(
            Verdict(
                key=item.key(model),
                choice=choice,
                needs_audio=needs_audio,
                reason=decision.reason,
                model=model,
                prompt_version=PROMPT_VERSION,
            )
        )
    return out
