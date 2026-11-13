# Experiments

Judged on `python -m src.evaluate`: average log loss over the validation
seasons 2015-16, 2016-17 and 2017-18. Keep a change only if it improves the
average by at least 0.002 AND improves at least 2 of 3 seasons.
Holdout seasons 2018-19 and 2019-20 are never used by the loop.

**Current best: exp 0, validation LL 0.6067** (market 0.5894, Elo 0.6101)

---

## Exp 0: harness + baseline (2026-11-13)

- **Setup, not an experiment.** EXPERIMENTS.md, IDEAS.md, `src/evaluate.py`
  and `tests/locked/` didn't exist, so this run created them. Decisions made
  without a human:
  - **Validation:** the last 3 full seasons before the holdout (2015-16 to 2017-18).
  - **Holdout:** 2018-19 and 2019-20 (partial), removed before anything
    runs. This keeps a clean test set after many loop iterations.
  - **No market data for the model:** `evaluate` strips the market columns.
    A model that sees the closing line would trivially "beat" it.
  - **Model contract:** `src/model.py: predict_season(history, test_season)`.
    Experiments edit the model and features, never the harness.
  - **Locked tests** (`tests/locked/`) cover the fixed split, the 2025-26
    refusal, no market or holdout data reaching the model, valid
    predictions, and leakage. The leakage test flips results and checks
    that predictions dated on or before the flipped game don't change.
    It applies to whatever model future experiments write.
- **Baseline model:** logistic regression (C=1) on Elo difference,
  season-to-date form, rest, back-to-backs, 3-in-4 nights, travel, time
  zones, altitude, neutral site and playoffs. It trains on every season
  from 2008-09 up to the validation season.

| Season | LL | Brier |
|---|---|---|
| 2015-16 | 0.5803 | 0.1987 |
| 2016-17 | 0.6249 | 0.2178 |
| 2017-18 | 0.6149 | 0.2132 |
| **Average** | **0.6067** | 0.2099 |

Learned: these numbers match the earlier backtest for the same seasons,
which cross-checks the harness.

---

## Exp 1: tune Elo settings inside the training seasons (2026-11-13). REVERTED

- **Idea:** Elo difference is the strongest feature by far, and its settings
  (K, season carryover, home advantage) were never tuned.
- **Change:** in `src/model.py`, a per-fold grid search (36 settings):
  - K ∈ {15, 20, 25, 30}
  - share reverted to the mean each season ∈ {0.25, 0.40, 0.55}
  - home advantage ∈ {60, 80, 100}

  Each setting is scored by Elo's own log loss on that fold's training
  seasons only. Elo forecasts each game before seeing it, so this is
  out-of-sample (time-series CV). The chosen Elo then feeds the usual
  features. A leakage test confirmed the choice doesn't depend on
  test-season results.
- **Chosen in every fold:** K=20, revert 0.25 (both the existing defaults)
  and home advantage 80 (was 100).

| Season | LL before → after | Brier before → after |
|---|---|---|
| 2015-16 | 0.5803 → 0.5805 | 0.1987 → 0.1988 |
| 2016-17 | 0.6249 → 0.6249 | 0.2178 → 0.2178 |
| 2017-18 | 0.6149 → 0.6153 | 0.2132 → 0.2134 |
| **Average** | **0.6067 → 0.6069** | 0.2099 → 0.2100 |

- **Decision: REVERTED.** The average got worse; no season improved.
- **Learned:** the public Elo defaults are already optimal on these
  seasons. Only home advantage moved, and the logistic regression absorbs
  that through its intercept. Better Elo settings won't help; gains have to
  come from new information or a better way to use margins.
