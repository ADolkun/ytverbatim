# Correction-LLM eval

Generated 2026-11-14T05:27:11+00:00. Aggregates only; files carry generic labels.

## Files

| label | language view | access | minutes | manual reference |
|---|---|---|---|---|
| en-01 | en | public | 20.5 | no |
| en-02 | en | public | 18.7 | no |
| en-03 | en | public | 19.2 | no |
| zh-01 | zh | members | 39.4 | no |
| zh-02 | zh | members | 41.2 | no |
| zh-03 | zh | members | 35.8 | no |
| zh-04 | zh | public | 26.4 | no |
| zh-05 | zh | public | 13.6 | yes |
| zh-06 | zh | public | 32.0 | no |
| zh-07 | zh | public | 35.6 | no |

## Judge screening

313 disputed spans over the base and the candidates' texts (2 differ only in written form and are excluded). Judge: claude-opus-5-5 (judge-v1), seed 0, 311 of 311 sampled spans answered. The judge ranks the candidates; it does not decide.

| candidate | run | audible edits | judged | needs audio | fix precision [95% CI] | over-correction [95% CI] | net errors removed/h [95% CI] | number violations | first-try valid JSON | invalid JSON after retry |
|---|---|---|---|---|---|---|---|---|---|---|
| c-sonnet | complete | 233 | 233 | 0 | 0.918 [0.876, 0.947] | 0.009 [0.002, 0.031] | 45.05 [42.92, 46.96] | 2 | 100.0% | no |
| c-haiku | complete | 180 | 180 | 0 | 0.900 [0.847, 0.936] | 0.017 [0.006, 0.048] | 33.78 [31.45, 35.70] | 0 | 100.0% | no |
| l-qwen35-9b (local) | complete | 127 | 127 | 0 | 0.858 [0.787, 0.908] | 0.024 [0.008, 0.067] | 22.52 [20.40, 24.44] | 1 | 98.6% | no |

Judge pick: **c-haiku** (c-haiku has the best judge net (33.78/h)).
- c-sonnet is ineligible: 2 number violations.
- l-qwen35-9b is ineligible: 1 number violations.

## Guard false-rejects

Disputes sampled from rejected proposals: 181, 181 answered. A false reject is a proposal whose every audible dispute went to the proposed reading. Numeric guards are informational: a text judge cannot verify numbers.

| guard | sampled proposals | judged | false rejects | share [95% CI] | note |
|---|---|---|---|---|---|
| balloon | 3 | 3 | 1 | 0.333 [0.061, 0.792] |  |
| cjk_numerals | 4 | 3 | 2 | 0.667 [0.208, 0.939] | informational |
| digits | 6 | 6 | 1 | 0.167 [0.030, 0.564] | informational |
| direction | 1 | 1 | 1 | 1.000 [0.207, 1.000] |  |
| negation | 11 | 11 | 7 | 0.636 [0.354, 0.848] |  |
| numeric_spans | 1 | 1 | 1 | 1.000 [0.207, 1.000] | informational |
| shrink | 2 | 2 | 0 | 0.000 [0.000, 0.658] |  |
| symbols | 12 | 12 | 10 | 0.833 [0.552, 0.953] |  |

## Owner's labels

Session of **c-haiku** against the base: 80 of 80 spans decided.
Fix precision and over-correction are weighted back from the stratified sample; Δ errors per hour is negative when the candidate leaves fewer errors.

| slice | decided | effective n | fix precision [95% CI] | over-correction [95% CI] | both fine | both wrong | audible edits | hours | Δ errors/h [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| overall | 80 | 58.1 | 0.812 [0.694, 0.892] | 0.011 [0.001, 0.081] | 0.011 | 0.165 | 180 | 4.71 | -30.64 [-34.41, -26.31] |
| en | 15 | 15.0 | 0.667 [0.417, 0.848] | 0.133 [0.037, 0.379] | 0.133 | 0.067 | 15 | 0.97 | -8.22 [-13.37, -3.08] |
| zh | 65 | 50.1 | 0.826 [0.699, 0.906] | 0.000 [0.000, 0.071] | 0.000 | 0.174 | 165 | 3.73 | -36.48 [-41.26, -31.23] |

Judge-human agreement on decisive spans: 65 of 68 (0.956 [0.878, 0.985]).

## Manual reference check

Errors against the manual captions, before → after correction. Correction never touches the regions the captions rewrite, so the difference isolates its edits. One file only, so it validates the judge and gates nothing.

| candidate | CER (itn) | CER (raw, case kept) | MER (itn) | judge net errors removed/h on these files |
|---|---|---|---|---|
| c-sonnet | 215 → 177 (+17.7% removed) | 235 → 193 (+17.9% removed) | 184 → 169 (+8.2% removed) | 92.59 [83.77, 97.00] |
| c-haiku | 215 → 191 (+11.2% removed) | 235 → 207 (+11.9% removed) | 184 → 175 (+4.9% removed) | 66.14 [44.09, 83.77] |
| l-qwen35-9b | 215 → 196 (+8.8% removed) | 235 → 215 (+8.5% removed) | 184 → 176 (+4.3% removed) | 57.32 [48.50, 61.73] |

## Cost

| candidate | calls | batches | input tokens | output tokens | wall time s | peak footprint GB |
|---|---|---|---|---|---|---|
| c-sonnet | 73 | 73 | 204059 | 26017 | 327.8 | — |
| c-haiku | 73 | 73 | 203913 | 128129 | 571.4 | — |
| l-qwen35-9b | 74 | 73 | 110029 | 20273 | 2309.1 | 12.01 |

## Limits

- Punctuation parity: production punctuates with the same `[llm]` model that corrects, but here every candidate corrects a base punctuated once by claude-cli:claude-sonnet-5-5, so a local winner's own punctuation is not measured.
- The intervals cover edit sampling and labelling, not variation between videos; the set is a few videos from a few channels. Recall (errors correction missed) is not measured.

## Decision

**correction ships default-on with c-haiku.**

| check | result | detail |
|---|---|---|
| eligible candidate | pass | c-haiku: c-haiku has the best judge net (33.78/h) |
| every span decided | pass | 80 spans |
| enough labels | pass | 80 decided spans (needs >= 30) |
| over-correction | pass | 0.011 (needs <= 0.15) |
| errors per hour | pass | -30.64 [-34.41, -26.31] (upper bound needs < 0.0) |
| number preservation | pass | 0 violations (needs <= 0) |
| JSON after the retry | pass | no LLMOutputError |

Rule: the judge picks the best eligible candidate by net errors removed per hour (a local one whose interval overlaps it is preferred); only that candidate's edits go to the owner. It ships default-on only if every session span is decided and at least 30 are, over-correction is at most 15%, the upper 95% bound of the change in errors per hour is below 0, no number changed and no correction call raised `LLMOutputError`. Otherwise correction stays default-off.

## Owner judging

As in the ASR evaluation, the owner judged the session by kind rather than in the terminal loop. The spans were sorted
by kind before the key was opened, and each kind got one owner policy. Only real content differences were judged one by
one, with each clip's YouTube timestamp at hand.

| kind | spans | owner policy | outcome |
|---|---|---|---|
| same sound, different characters or Latin case | 66 | the correctly written reading wins | c-haiku 66 |
| small words that leave the meaning unchanged | 2 | both fine | — |
| content differences | 12 | judged one by one | both wrong 10, base 2 |

Most content differences were words the ASR misheard so badly that the correction swapped in a different wrong word.

## External candidates

`x-gpt61-medium` and `x-gpt61-high` are gpt-6.1-sol at medium and high reasoning effort. An external client served them
through `YTVERBATIM_EVAL_CLIENT_FACTORY`, with the same prompts and guards. They were screened after the owner's session,
so they are absent from the tables above. Adding candidates changes the disputed spans, and with them every span id the
session is keyed by. Like every other candidate, they corrected the base punctuated by claude-cli:claude-sonnet-5-5.

This screen covers all five candidates. The judge's estimates move with the span set, so compare rows within this table
only. 382 disputed spans (2 differ only in written form and are excluded); 380 of 380 sampled spans answered.

| candidate | audible edits | judged | needs audio | fix precision [95% CI] | over-correction [95% CI] | net errors removed/h [95% CI] | number violations | first-try valid JSON |
|---|---|---|---|---|---|---|---|---|
| c-sonnet | 233 | 233 | 0 | 0.871 [0.822, 0.908] | 0.009 [0.002, 0.031] | 42.71 [40.16, 45.05] | 2 | 100.0% |
| c-haiku | 180 | 180 | 0 | 0.844 [0.784, 0.890] | 0.017 [0.006, 0.048] | 31.66 [29.11, 34.00] | 0 | 100.0% |
| l-qwen35-9b (local) | 127 | 127 | 0 | 0.795 [0.717, 0.856] | 0.016 [0.004, 0.056] | 21.04 [18.91, 23.16] | 1 | 98.6% |
| x-gpt61-medium | 294 | 294 | 0 | 0.929 [0.893, 0.953] | 0.007 [0.002, 0.024] | 57.58 [55.24, 59.49] | 1 | 100.0% |
| x-gpt61-high | 300 | 299 | 1 | 0.920 [0.883, 0.945] | 0.013 [0.005, 0.034] | 57.77 [55.00, 60.12] | 1 | 100.0% |

Against the manual captions, both efforts remove the same errors: CER (itn) 215 → 166 (+22.8% removed), CER (raw, case
kept) 235 → 182 (+22.6% removed), MER (itn) 184 → 163 (+11.4% removed). Cost: both made 73 calls with 112936 input
tokens. Medium produced 20870 output tokens in 626.7 s; high produced 35157 in 1040.2 s.

Every number violation in this screen is one of two checker false positives, and no number changed:

- In one, the correction split a figure fused to a ticker, so the checker saw one more ticker. This accounts for one of
  c-sonnet's two and for l-qwen35-9b's one.
- In the other, the correction joined a ticker-like word into a company name without recording an `entity_corrected`
  flag, so the checker counted a ticker as removed. This accounts for c-sonnet's other violation and for both gpt-6.1-sol
  runs.

## Owner decision

**Correction ships default-on, and `claude-cli:claude-sonnet-5-5` is the recommended LLM, not c-haiku.** The rule picked
c-haiku only because c-sonnet's two number violations made it ineligible, and both are the false positives described
above. c-sonnet leads c-haiku and l-qwen35-9b on every judge metric and against the manual captions. The owner's labels
are for c-haiku's edits; the judge agreed with them on 65 of 68 decisive spans, which supports its ranking of the other
candidates.

gpt-6.1-sol at medium effort, run through an external client, scored higher than every candidate on a built-in provider.
High effort added nothing measurable and took 1.7 times as long. Two follow-ups would let the checker pass these runs on
its own: splitting figures fused to tickers, and having the correction guard record a merged ticker-like word.
