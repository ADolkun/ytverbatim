# ASR selection results

Each `asr-<date>.md` here is written by `uv run python -m evals.run_asr decide`. It holds aggregates only:

- files appear as generic labels (`zh-01`, `en-01`), never by video id or title;
- no transcript, reference, caption or context text;
- the screening table, the absolute error against the one manual reference, the blind A/B win rates, speed and
  memory, the quality-gate inputs, and the decision with every check that led to it.

Everything else (audio, plans, hypotheses, judge verdicts, A/B sessions and decisions, the full report) stays in
the private eval directory, `$YTVERBATIM_EVAL_DIR`.

## How a result is produced

```bash
export YTVERBATIM_EVAL_DIR=~/.local/share/ytverbatim/eval
uv run python -m evals.run_asr vad                        # shared speech regions, ≤30 s chunk plan, windows
uv run python -m evals.run_asr asr --hyp w-prod           # one hypothesis at a time; resumable per chunk
uv run python -m evals.run_asr cleanup --hyp q-base       # after its outputs are complete
uv run python -m evals.run_asr captions                   # yt-auto pseudo-hypothesis and ref-manual reference
uv run python -m evals.run_asr screen                     # disputes, automatic metrics, blind LLM judge sample
uv run python -m evals.run_asr session --control w-prod --challenger q-ctx   # both must be complete on every file
uv run python -m evals.run_asr listen                     # the owner's blind terminal A/B
uv run python -m evals.run_asr report
uv run python -m evals.run_asr decide
```

The LLM judge only screens challengers. The decision comes from the blind two-option human A/B, and only once every
span has a decision (`listen` brings skipped spans back). A challenger replaces Whisper anywhere only if its overall
win rate's Wilson 95% lower bound is above 0.5, and in a language only if that language's lower bound is above 0.5
too, it is non-inferior on
numeric and entity spans (point estimate at least 0.5, lower bound above 0.40), its quality gates hold, and its
CER on the manual reference is within one point of the control's. Otherwise Whisper stays. See
`evals/decide.py` for the exact rule and tolerances.
