# AI disclosure: working notes for the report

These are notes for the report's AI-disclosure section. Verify each item and edit before quoting it.

## What AI wrote

| Part | Written by | Model |
|---|---|---|
| Treatments 1–6: all code, tests and per-step metrics under `experiments/runs/treatment1-6/` | LLM agent (Codex extension), one fresh context per step | gpt-5.6-luna; treatment 4 trial 1 ran on gpt-6.1-sol |
| Treatment 7: shared core `src/`, `experiments/run_treatment7.py` | Human team. *[State any AI assistance here.]* | — |
| Evaluation harness (`evaluation/`), `scripts/build_ladder_tags.py`, `scripts/tag_histogram.py`, the presentation deck and the cost/LOC analysis | Claude Code, reviewed by the team | Claude Opus 5.5 |

## Errors in LLM-generated code or metrics that we caught

1. **Invalid race comparison.** Treatment 1, trial 1, x264, step 10: the race used 56 evaluations against the baseline's 32. The run flagged it as invalid itself, and it stays in the data, flagged.
2. **Wrong LOC count.** Treatment 3, trial 3, NASA93, step 6 self-reported 8,744 new LOC. Medians are used to limit the damage.
3. **Tuner sorted the wrong way.** Treatment 1, trial 3, x264 race: successive halving kept the *slowest* configurations, so it lost to random search on every seed (found by `evaluation/`). It also only accepts power-of-two budgets.
4. **Linear model without an intercept.** Treatment 6, trial 1, x264 guess predicts negative times (MAE about 456 s against 131 for guessing the average). Its own log reported MAE 466, but the step was marked done.
5. **"Predictors" that only recall the past.** Treatment 2 trial 1 (x264), treatment 5 trial 3 (Health) and treatment 6 trials 2–3 (Health) fall back to the average for any unseen configuration, so they are lookup tables, not models.
6. **Selectors that read the answers.** Most Health race steps in treatments 2, 4, 5 and 6 picked the best row after reading every recorded outcome. That is a lookup, not a budgeted search.
7. **Model drift.** Treatment 4 trial 1 used a different, larger model from all other runs, which breaks the "same model" protocol.
8. **Inconsistent self-reported metrics.** `error_count` was counted differently by each arm (treatments 1–3 report about 0, treatments 4–6 report 200+), and `test_pass` used mixed formats ("14/14", "1", "true").

9. **Over-reported new LOC.** Treatments 1 and 3 logged about 1.5× more new lines than their saved source contains. T1 trial 1 x264 logged 96 / 132 / 107 for steps 1–3, but the source files hold 80 / 81 / 65 non-blank lines: step 1 counted blank lines, later steps apparently added test lines. Treatments 2, 4, 5 and 6 match the tags to within about 7%. The report uses the tag-derived counts. The task talk's "30–40% saving" was based on the inflated counts and is corrected in the report (8% / 8% / 24% per ladder).

## Errors in AI-assisted analysis that we caught

1. **Token estimate.** Tokens were first estimated as saved prompt+response text ÷ 4. The real Codex logs showed about 200× more, because the agent rereads files every turn. Replaced with per-call usage from the logs.
2. **Archive token script.** A teammate's script first gave token totals about 5× too high and impossible median LOC (461 for T1, when every run is 918–1,159). Recomputed from per-call `last_token_usage`, with medians from the repo.
3. **Arithmetic slip.** One cost slide said ~$68; the correct sum was ~$59. Fixed before publishing.
4. **Scoring adapters.** Several first-pass adapters picked the wrong output field (e.g. a stddev instead of the prediction), or fed the wrong input format. Each odd score was checked against the program's own logged result before it was accepted.
