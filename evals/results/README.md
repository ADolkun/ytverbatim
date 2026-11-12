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

# Correction-LLM results

Each `correction-<date>.md` here is written by `uv run python -m evals.run_correction decide`. It follows the same
privacy rules as the ASR results: aggregates only, files as generic labels (`zh-01`, `en-01`), candidates by id
(`c-sonnet`, `l-qwen35-9b`), and no transcript, reading, caption or context text.

It holds the judge screening of every candidate (fix precision, over-correction and net errors removed per hour, with
95% intervals), the guard false-reject shares, the owner's labels of the picked candidate with the judge-human
agreement, the change in error against the one manual reference, cost, and the decision with every check behind it.

## How a result is produced

```bash
export YTVERBATIM_EVAL_DIR=~/.local/share/ytverbatim/eval
uv run python -m evals.run_correction base --hyp w-prod             # normalize and punctuate once: the shared base
uv run python -m evals.run_correction correct --candidate c-sonnet  # the production correction, per candidate
uv run python -m evals.run_correction correct --candidate c-haiku
uv run python -m evals.run_correction correct --candidate l-qwen35-9b
uv run python -m evals.run_correction cleanup --candidate l-qwen35-9b   # local weights, once its run is complete
uv run python -m evals.run_correction screen                        # edit disputes, blind judge sample, the pick
uv run python -m evals.run_correction session                       # the picked candidate's edits, blind A/B
uv run python -m evals.run_correction listen                        # the owner's blind terminal A/B
uv run python -m evals.run_correction report
uv run python -m evals.run_correction decide                        # prints the decision JSON, writes the file
```

Correction changes the text only where it edits, so a candidate's net change in errors is the sum over its edits of
+1 (a fix), -1 (a break) and 0. The blind LLM judge labels a uniform sample of those edits to rank the candidates and
pick one to test (`screen --judge-n`, `--max-calls` and the judge cache resume across quota windows). `session` builds
on `screen.json` and refuses while the judge sample is incomplete. The owner's labels of that one candidate, weighted
back by stratum, decide.

Correction ships default-on only if every session span is decided, over-correction is at most 15%, the upper 95%
bound of the change in errors per hour is below 0, no number changed, and no correction call returned invalid JSON
after its retry (the first-try validity is reported, not gated). Otherwise it stays default-off. See
`evals/correction_metrics.py` for the exact rule.

A candidate whose call failed twice keeps its `failure.json` and stays ineligible, as production would have failed on
it too. To try it again, move `correction/runs/<candidate>` aside and rerun `correct` from scratch.
