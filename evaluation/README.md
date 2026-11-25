# External metrics: shared test split for all 7 treatments

This folder scores every treatment's **saved** code on one shared split. It makes no new LLM calls, and none of the generated code under `experiments/runs/` is edited. Each program is imported read-only and called through a small adapter.

## Rerun

```
python3 evaluation/make_splits.py      # freeze the 80/20 split per dataset (seed 2026, test capped at 200 rows)
python3 evaluation/run_eval.py guess   # run every treatment's step-2 guess code on the split
python3 evaluation/score_guess.py      # MAE, hit rate, Spearman rho; 7-way ranks with `same`
python3 evaluation/run_race.py         # run every treatment's step-10 race code (x264 and Health)
python3 evaluation/score_race.py       # regret, win rate vs random; 7-way ranks with `same`
RACE_BUDGET=10 python3 evaluation/run_race.py x264 && RACE_BUDGET=10 python3 evaluation/score_race.py
```

Results are written to `evaluation/results/` (`guess_by_run.csv`, `guess_ranks.csv`, `race_by_run.csv`, `race_ranks.csv`).

## Protocol

- **guess.** Each program trains on `splits/<dataset>/train.csv` and predicts every row of `test.csv`. Targets are Performance− (x264), Effort− (NASA93) and MRE− (Health). Metrics:
  - MAE;
  - hit rate: within 5% of the real value (x264), within 25% (NASA93, i.e. PRED(25)), or within 0.05 MRE (Health);
  - Spearman ρ.

  The floor is a predictor that always guesses the training-set average.
- **race on x264.** Every tuner searches the same 200 unseen test configurations, for seeds 1–10. The score is the real encode time of its pick. It is run at two budgets: 20 evaluations (`out/race`) and 10 (`RACE_BUDGET=10`, `out/race_b10`). At 20, even random search gets close to the best, so budget 10 is the headline result.
- **race on Health.** History is `train.csv`. The score is the real MRE− of the recommended configuration. Programs that read every recorded result are labelled `full history`: that is a lookup, not a search, and is not comparable.
- **Regret.** (pick − best) / (median − best) over the candidate pool: 0 is the best pick, 1 is a median pick.
- **Ranks.** Each treatment's values are pooled over its trials and sorted by median. A new rank starts only when `src/core/certify.same` says a treatment differs from the one above it. Rank 0 is best.

## Deviations, recorded in `adapters.py`

- **T1 trial 3, x264 race:** the program only accepts power-of-two budgets, so it ran at 16 (for budget 20) and 8 (for budget 10).
- **T2 trial 2, x264 race:** the program has no seed option, so its one deterministic pick is repeated for all 10 seeds.
- **Programs that refit on every prediction:** they were fitted once with their own functions and then predicted the whole test set in one batch.
- **Programs that split their input internally:** where a split-size argument existed, it was set to near zero so they train on all of `train.csv`. T5 trial 3 (Health guess) has a fixed 80% internal split.
- **Not scorable:** T1 trial 1 and T3 trial 1 (Health race) never recommend a single configuration.
