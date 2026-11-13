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
