# NBA Betting Model

Predict NBA game outcomes from past data and backtest whether the model
beats the betting market's no-vig closing line. Backtest only; no live betting.

**Result so far:** the model beats a standard Elo baseline but not the
market. Adding its features on top of the market's probability makes
predictions worse, and betting at closing lines loses about the vig. See
[RESULTS.md](RESULTS.md).

## Run it

```bash
pip install -r requirements.txt
python -m src.sbr        # download + clean 2007-08..2019-20 games and closing odds
python -m src.backtest   # walk-forward backtest, writes reports/
python -m pytest         # 85 tests
```

## Pipeline

| Step | Module | Output |
|---|---|---|
| Raw data | `src/sbr.py` (Sports Book Reviews odds archive) | `data/raw/sbr/*.csv` (cached) |
| Clean games + no-vig market probability | `src/sbr.py` | `data/processed/games.csv` |
| Elo baseline | `src/elo.py` | `data/processed/elo_probs.csv` |
| Pre-game features (form, rest, travel, altitude) | `src/features.py`, `src/geo.py` | (in memory) |
| Walk-forward model, metrics, betting sim | `src/backtest.py` | `reports/`, `data/processed/backtest_predictions.csv` |
| Odds math | `src/odds.py` | |
| Log loss / Brier | `src/metrics.py` | |

`src/data_fetch.py` is an alternative results source via `nba_api`
(stats.nba.com). It works from a home connection but stats.nba.com does not
answer cloud servers, and it has no odds, so the pipeline uses the SBR data.

## Guard rails

- **No leakage:** Elo probabilities and every feature use only games before
  tip-off. Tests flip each game's result and check nothing for that game or
  earlier changes.
- **Walk-forward:** each test season is predicted by a model trained only on
  earlier seasons.
- **ROI alarm:** the backtest warns if any strategy shows ROI above 10%. A
  test confirms the alarm fires when a result is deliberately leaked.
