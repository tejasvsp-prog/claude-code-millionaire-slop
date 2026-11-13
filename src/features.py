"""Pre-game features for each game.

Every feature for a game uses only games that finished BEFORE that game's
date, so nothing from the game itself (or later) leaks in. A team plays
at most once per day, so "earlier rows for this team" == "before tip-off".

Per-team, within the current season:
    REST_DAYS     days since the team's previous game (capped, first game = cap)
    B2B           1 if the team played yesterday
    GP            games already played this season
    WIN_PCT       season-to-date win share (prior 0.5 for GP=0, shrunk early)
    MARGIN_AVG    season-to-date average point margin (0 for GP=0, shrunk early)
    MARGIN_L10    average point margin over the last 10 games (0 if none)
    GAMES_L3D     games played in the previous 3 days (2 => 3rd game in 4 nights)
    TRAVEL_KM     distance from the previous game's arena to this one (0 for game 1)
    TZ_CHANGE     |time-zone change| in hours since the previous game

Where a game is played (the home arena) is fixed by the schedule, so
travel features are known before tip-off.

Game-level (home minus away where it makes sense):
    ELO_DIFF, ELO_PROB_HOME (pre-game, from src.elo), NEUTRAL, PLAYOFFS
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.geo import HIGH_ALTITUDE, arena, haversine_km

REST_CAP = 7
SHRINK_GAMES = 5  # pseudo-games of league-average used to shrink early-season stats

TEAM_FEATURES = [
    "REST_DAYS", "B2B", "GP", "WIN_PCT", "MARGIN_AVG", "MARGIN_L10",
    "GAMES_L3D", "TRAVEL_KM", "TZ_CHANGE",
]

MODEL_FEATURES = [
    "ELO_DIFF",
    "NEUTRAL",
    "PLAYOFFS",
    "REST_DIFF",
    "HOME_B2B",
    "AWAY_B2B",
    "WIN_PCT_DIFF",
    "MARGIN_AVG_DIFF",
    "MARGIN_L10_DIFF",
]

SCHEDULE_FEATURES = [
    "HOME_3IN4",
    "AWAY_3IN4",
    "TRAVEL_DIFF_1000KM",
    "AWAY_TZ_CHANGE",
    "HOME_TZ_CHANGE",
    "ALTITUDE",
]

ALL_FEATURES = MODEL_FEATURES + SCHEDULE_FEATURES


def team_game_view(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) with the team's points for/against."""
    home = pd.DataFrame(
        {
            "GAME_ID": games["GAME_ID"],
            "GAME_DATE": games["GAME_DATE"],
            "SEASON": games["SEASON"],
            "TEAM_ID": games["HOME_TEAM_ID"],
            "VENUE_TEAM": games["HOME_TEAM_ID"],
            "IS_HOME": 1,
            "MARGIN": games["MARGIN"],
        }
    )
    away = home.assign(TEAM_ID=games["AWAY_TEAM_ID"], IS_HOME=0, MARGIN=-games["MARGIN"])
    tg = pd.concat([home, away], ignore_index=True)
    tg["GAME_DATE"] = pd.to_datetime(tg["GAME_DATE"])
    tg["WIN"] = (tg["MARGIN"] > 0).astype(int)
    return tg.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def team_pregame_features(tg: pd.DataFrame) -> pd.DataFrame:
    """Add TEAM_FEATURES to a team_game_view frame using only prior games."""
    tg = tg.copy()
    if tg.duplicated(["TEAM_ID", "GAME_DATE"]).any():
        raise ValueError("a team has two games on the same date; leak-free ordering is ambiguous")
    g = tg.groupby(["TEAM_ID", "SEASON"], sort=False)

    prev_date = g["GAME_DATE"].shift(1)
    rest = (tg["GAME_DATE"] - prev_date).dt.days
    tg["REST_DAYS"] = rest.fillna(REST_CAP).clip(upper=REST_CAP)
    tg["B2B"] = (rest == 1).astype(int)

    tg["GP"] = g.cumcount()
    prior_wins = g["WIN"].cumsum() - tg["WIN"]
    prior_margin = g["MARGIN"].cumsum() - tg["MARGIN"]
    tg["WIN_PCT"] = (prior_wins + 0.5 * SHRINK_GAMES) / (tg["GP"] + SHRINK_GAMES)
    tg["MARGIN_AVG"] = prior_margin / (tg["GP"] + SHRINK_GAMES)
    tg["MARGIN_L10"] = (
        g["MARGIN"].transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean()).fillna(0.0)
    )

    # Games in the 3 days before this one (within season).
    def _l3d(dates: pd.Series) -> np.ndarray:
        d = dates.to_numpy(dtype="datetime64[D]")
        lo = np.searchsorted(d, d - np.timedelta64(3, "D"), side="left")
        return np.arange(len(d)) - lo

    tg["GAMES_L3D"] = g["GAME_DATE"].transform(_l3d).astype(int)

    # Travel: where this game is vs where the previous game was.
    start_year = tg["SEASON"].str[:4].astype(int)
    locs = [arena(t, y) for t, y in zip(tg["VENUE_TEAM"], start_year)]
    tg["LAT"] = [l[0] for l in locs]
    tg["LON"] = [l[1] for l in locs]
    tg["TZ"] = [l[2] for l in locs]
    prev = g[["LAT", "LON", "TZ"]].shift(1)
    has_prev = prev["LAT"].notna()
    tg["TRAVEL_KM"] = 0.0
    tg.loc[has_prev, "TRAVEL_KM"] = [
        haversine_km(a, b, c, d)
        for a, b, c, d in zip(prev.loc[has_prev, "LAT"], prev.loc[has_prev, "LON"],
                              tg.loc[has_prev, "LAT"], tg.loc[has_prev, "LON"])
    ]
    tg["TZ_CHANGE"] = (tg["TZ"] - prev["TZ"]).abs().fillna(0.0)
    return tg


def build_features(games_with_elo: pd.DataFrame) -> pd.DataFrame:
    """Return games with pre-game feature columns (see MODEL_FEATURES)."""
    df = games_with_elo.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    tg = team_pregame_features(team_game_view(df))

    for side, is_home in (("HOME", 1), ("AWAY", 0)):
        part = tg[tg["IS_HOME"] == is_home][["GAME_ID", *TEAM_FEATURES]]
        part = part.rename(columns={c: f"{side}_{c}" for c in TEAM_FEATURES})
        df = df.merge(part, on="GAME_ID", how="left", validate="one_to_one")

    df["ELO_DIFF"] = df["HOME_ELO_PRE"] - df["AWAY_ELO_PRE"]
    if "NEUTRAL" not in df.columns:
        df["NEUTRAL"] = 0
    if "SEASON_TYPE" in df.columns:
        df["PLAYOFFS"] = (df["SEASON_TYPE"] == "Playoffs").astype(int)
    else:
        df["PLAYOFFS"] = 0
    df["REST_DIFF"] = df["HOME_REST_DAYS"] - df["AWAY_REST_DAYS"]
    df["WIN_PCT_DIFF"] = df["HOME_WIN_PCT"] - df["AWAY_WIN_PCT"]
    df["MARGIN_AVG_DIFF"] = df["HOME_MARGIN_AVG"] - df["AWAY_MARGIN_AVG"]
    df["MARGIN_L10_DIFF"] = df["HOME_MARGIN_L10"] - df["AWAY_MARGIN_L10"]
    df["HOME_3IN4"] = (df["HOME_GAMES_L3D"] >= 2).astype(int)
    df["AWAY_3IN4"] = (df["AWAY_GAMES_L3D"] >= 2).astype(int)
    df["TRAVEL_DIFF_1000KM"] = (df["HOME_TRAVEL_KM"] - df["AWAY_TRAVEL_KM"]) / 1000.0
    df["ALTITUDE"] = (df["HOME_TEAM_ID"].isin(HIGH_ALTITUDE) & (df["NEUTRAL"] == 0)).astype(int)
    if df[ALL_FEATURES].isna().any().any():
        raise ValueError("NaN in model features")
    return df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
