# NBA Betting Model

## Goal
Predict NBA game outcomes from past data and backtest whether the
model beats the betting market (no-vig closing lines). Backtest only.

## Stack
- Python, pandas, numpy, scikit-learn, nba_api, matplotlib
- Raw data cached in data/raw/, cleaned data in data/processed/
- Real logic in src/, exploring in notebooks/, tests in tests/

## Rules
- NO DATA LEAKAGE: features for a game can only use info from before tip-off.
- Split train/test by season (walk-forward). Never use random splits.
- Compare every model to (1) the Elo baseline and (2) the market's
  no-vig probability, using log loss and Brier score.
- Cache every API pull to CSV. Add a sleep between nba_api calls.
- API keys live in .env, and .env is never committed.
- Write a pytest test for any function that does odds math.
- If a backtest shows ROI above ~10%, assume a bug or leakage first.
