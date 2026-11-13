"""Fixed evaluation harness for the experiment loop. DO NOT EDIT.

    python -m src.evaluate

For each validation season V, calls src.model.predict_season with all
games up to and including V, with the market columns removed (the model
must not see the betting line) and the holdout seasons removed (they are
never used by the loop). Scores log loss and Brier on V's games that have
a closing line, next to the Elo baseline and the no-vig market.

Writes reports/evaluate.json and prints the average validation log loss,
the number experiments are judged on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.elo import run_elo
from src.metrics import brier_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
GAMES_CSV = ROOT / "data" / "processed" / "games.csv"
REPORT = ROOT / "reports" / "evaluate.json"

VALIDATION_SEASONS = ("2015-16", "2016-17", "2017-18")
HOLDOUT_SEASONS = ("2018-19", "2019-20")  # untouched by the loop
FORBIDDEN_FROM_START_YEAR = 2025  # never load 2025-26 or later
MARKET_COLS = ("HOME_ML", "AWAY_ML", "OVERROUND", "MARKET_PROB_HOME")
BEATS_MARKET_ALARM = 0.01


def load_games(path: Path = GAMES_CSV) -> pd.DataFrame:
    games = pd.read_csv(path, dtype={"GAME_ID": str})
    start_years = games["SEASON"].str[:4].astype(int)
    if (start_years >= FORBIDDEN_FROM_START_YEAR).any():
        raise ValueError("games file contains 2025-26 or later; refusing to load")
    return games[~games["SEASON"].isin(HOLDOUT_SEASONS)].reset_index(drop=True)


def evaluate(games: pd.DataFrame, predict_season: Callable | None = None) -> dict:
    if predict_season is None:
        from src.model import predict_season
    if games["SEASON"].isin(HOLDOUT_SEASONS).any():
        raise ValueError("holdout seasons must be removed before evaluation")

    elo = run_elo(games).set_index("GAME_ID")["ELO_PROB_HOME"]
    model_input = games.drop(columns=[c for c in MARKET_COLS if c in games.columns])

    seasons = {}
    for v in VALIDATION_SEASONS:
        history = model_input[model_input["SEASON"] <= v].copy()
        preds = predict_season(history, v)
        test = games[games["SEASON"] == v].set_index("GAME_ID")
        if set(preds.index) != set(test.index):
            raise ValueError(f"{v}: predictions must cover exactly the season's games")
        p = preds.reindex(test.index).astype(float)
        if p.isna().any() or ((p <= 0) | (p >= 1)).any():
            raise ValueError(f"{v}: predictions must be in (0, 1)")
        scored = test["MARKET_PROB_HOME"].notna()
        y = test.loc[scored, "HOME_WIN"]
        seasons[v] = {
            "n": int(scored.sum()),
            "model_logloss": log_loss(y, p[scored]),
            "model_brier": brier_score(y, p[scored]),
            "elo_logloss": log_loss(y, elo.reindex(test.index)[scored]),
            "market_logloss": log_loss(y, test.loc[scored, "MARKET_PROB_HOME"]),
            "market_brier": brier_score(y, test.loc[scored, "MARKET_PROB_HOME"]),
        }

    avg = {k: float(np.mean([s[k] for s in seasons.values()]))
           for k in ("model_logloss", "model_brier", "elo_logloss", "market_logloss", "market_brier")}
    result = {"seasons": seasons, "average": avg,
              "beats_market_alarm": avg["model_logloss"] < avg["market_logloss"] - BEATS_MARKET_ALARM}
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--games", type=Path, default=GAMES_CSV)
    p.add_argument("--out", type=Path, default=REPORT)
    args = p.parse_args(argv)

    result = evaluate(load_games(args.games))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    rows = [{"season": s, **v} for s, v in result["seasons"].items()]
    rows.append({"season": "AVERAGE", **result["average"]})
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.4f"))
    print(f"\nVALIDATION LOG LOSS: {result['average']['model_logloss']:.4f}")
    if result["beats_market_alarm"]:
        print("WARNING: model beats the market by > 0.01 log loss. Assume a bug or leakage.")


if __name__ == "__main__":
    main()
