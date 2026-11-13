"""Elo baseline for NBA games.

Follows the well-known public NBA Elo recipe: K=20, home-court bonus of
100 Elo points, a margin-of-victory multiplier that is damped for heavy
favorites (so blowouts by favorites don't inflate ratings), and a
regression of 1/4 toward the mean at the start of each season.

Leakage guard: each game's probability is recorded BEFORE that game's
result updates the ratings, so `elo_prob_home` only uses earlier games.

Defaults are fixed, not tuned. If you tune them, tune only on seasons
before the first test season (walk-forward), never on the test seasons.

Usage:
    python -m src.elo            # reads data/processed/games.csv
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.metrics import accuracy, brier_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    home_adv: float = 100.0
    initial: float = 1500.0
    season_revert: float = 0.25  # fraction pulled back toward `initial` each new season
    use_mov: bool = True


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B (any home bonus already folded into rating_a)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def mov_multiplier(margin: float, winner_elo_diff: float) -> float:
    """Margin-of-victory multiplier.

    `margin` is the absolute point margin; `winner_elo_diff` is the winner's
    pre-game Elo (incl. home bonus) minus the loser's. Big wins by favorites
    get damped, upsets get amplified.
    """
    if margin <= 0:
        raise ValueError("margin must be positive (NBA games cannot tie)")
    return ((margin + 3.0) ** 0.8) / (7.5 + 0.006 * winner_elo_diff)


def elo_update(
    home_elo: float, away_elo: float, margin: float, params: EloParams = EloParams()
) -> tuple[float, float, float]:
    """Return (pre-game home win prob, new home Elo, new away Elo).

    `margin` = home points minus away points.
    """
    p_home = expected_score(home_elo + params.home_adv, away_elo)
    home_won = 1.0 if margin > 0 else 0.0
    mult = 1.0
    if params.use_mov:
        diff = home_elo + params.home_adv - away_elo
        winner_diff = diff if home_won else -diff
        mult = mov_multiplier(abs(margin), winner_diff)
    shift = params.k * mult * (home_won - p_home)
    return p_home, home_elo + shift, away_elo - shift


def revert_to_mean(rating: float, params: EloParams = EloParams()) -> float:
    return params.initial + (1.0 - params.season_revert) * (rating - params.initial)


def run_elo(games: pd.DataFrame, params: EloParams = EloParams()) -> pd.DataFrame:
    """Walk through games in date order, adding pre-game Elo columns.

    Requires GAME_ID, GAME_DATE, SEASON, HOME_TEAM_ID, AWAY_TEAM_ID, MARGIN.
    Adds HOME_ELO_PRE, AWAY_ELO_PRE, ELO_PROB_HOME.
    """
    required = {"GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM_ID", "AWAY_TEAM_ID", "MARGIN"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"games is missing columns: {sorted(missing)}")
    if (games["MARGIN"] == 0).any():
        raise ValueError("found games with MARGIN == 0")

    df = games.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["GAME_DATE", "GAME_ID"], kind="stable").reset_index(drop=True)

    ratings: dict[int, float] = {}
    current_season = None
    home_pre, away_pre, probs = [], [], []

    for season, h, a, margin in zip(
        df["SEASON"], df["HOME_TEAM_ID"], df["AWAY_TEAM_ID"], df["MARGIN"]
    ):
        if season != current_season:
            if current_season is not None:
                ratings = {t: revert_to_mean(r, params) for t, r in ratings.items()}
            current_season = season
        rh = ratings.get(h, params.initial)
        ra = ratings.get(a, params.initial)
        p, new_h, new_a = elo_update(rh, ra, margin, params)
        home_pre.append(rh)
        away_pre.append(ra)
        probs.append(p)
        ratings[h], ratings[a] = new_h, new_a

    df["HOME_ELO_PRE"] = home_pre
    df["AWAY_ELO_PRE"] = away_pre
    df["ELO_PROB_HOME"] = probs
    return df


def season_metrics(df: pd.DataFrame, prob_col: str = "ELO_PROB_HOME") -> pd.DataFrame:
    """Log loss, Brier score and accuracy per season (and overall)."""
    rows = []
    for season, g in list(df.groupby("SEASON", sort=True)) + [("ALL", df)]:
        rows.append(
            {
                "SEASON": season,
                "N": len(g),
                "LOG_LOSS": log_loss(g["HOME_WIN"], g[prob_col]),
                "BRIER": brier_score(g["HOME_WIN"], g[prob_col]),
                "ACCURACY": accuracy(g["HOME_WIN"], g[prob_col]),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run the Elo baseline over games.csv")
    p.add_argument("--games", type=Path, default=PROCESSED_DIR / "games.csv")
    p.add_argument("--out", type=Path, default=PROCESSED_DIR / "elo_probs.csv")
    p.add_argument(
        "--include-first-season",
        action="store_true",
        help="score the first season too (excluded by default: ratings still warming up)",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    games = pd.read_csv(args.games, dtype={"GAME_ID": str})
    out = run_elo(games)
    out.to_csv(args.out, index=False)
    log.info("wrote %s (%d games)", args.out, len(out))

    scored = out
    if not args.include_first_season and out["SEASON"].nunique() > 1:
        scored = out[out["SEASON"] != out["SEASON"].min()]
    print(season_metrics(scored).to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
