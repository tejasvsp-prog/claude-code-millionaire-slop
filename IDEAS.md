# Ideas

Ordered by expected value. Mark done with the experiment number.

## Untried
1. **Predict point margin, then convert it to a win probability.** Moved up:
   exp 1 showed the Elo signal is saturated, so the next lever is using
   margins better. A linear margin model plus a normal CDF with a fitted
   spread uses more information per game than win/loss.
2. **Tune the regularization strength (C, L1 vs L2) by season-wise CV inside
   the training seasons.** 15 partly redundant features with C=1 may be
   fitting noise. (New, after exp 1.)
3. **Drop the schedule features** (travel, time zones, altitude, 3-in-4).
   In the earlier backtest they made the model slightly worse (0.6036 vs
   0.6028). Simpler, but the gain is probably below the 0.002 bar.
4. **Exponentially weighted net rating** (margin, opponent-adjusted) that
   carries over between seasons, instead of season-to-date margin, which
   starts at zero each October.
5. **Opponent-adjusted ratings by recency-weighted ridge regression on
   margins** (SRS-style), refit daily from earlier games only. A different
   estimator from Elo that may not be redundant with it. (New, after exp 1.)
6. **Elo: bigger K early in the season**, when ratings are most stale. Lower
   priority now that exp 1 found the defaults optimal.
7. **Calibration** (Platt or isotonic). Logistic regression is usually
   already calibrated, so expect little.
8. **Player availability.** Not possible with current data (no player
   minutes offline); needs an nba_api pull from a home connection.

## Done
- ~~Tune Elo (K, season carryover, home advantage) inside the training
  seasons.~~ Exp 1, REVERTED (0.6067 → 0.6069): the defaults were already optimal.
