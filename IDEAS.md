# Ideas

Ordered by expected value. Mark done with the experiment number.

## Untried
1. **Tune Elo (K, season carryover, home advantage) inside the training
   seasons.** Elo difference is by far the strongest feature (weight 0.70 vs
   ≤0.25 for the rest), and its settings were never tuned.
2. **Drop the schedule features** (travel, time zones, altitude, 3-in-4).
   In the earlier backtest they made the model slightly worse (0.6036 vs
   0.6028). Simpler, but the gain is probably below the 0.002 bar.
3. **Exponentially weighted net rating** (margin, opponent-adjusted) that
   carries over between seasons, instead of season-to-date margin, which
   starts at zero each October.
4. **Predict point margin, then convert it to a win probability.** Margin
   carries more information per game than win/loss.
5. **Elo: bigger K early in the season**, when ratings are most stale.
6. **Calibration** (Platt or isotonic). Logistic regression is usually
   already calibrated, so expect little.
7. **Player availability.** Not possible with current data (no player
   minutes offline); needs an nba_api pull from a home connection.

## Done
(none yet)
