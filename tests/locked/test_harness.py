"""Locked guard-rail tests for the experiment loop. DO NOT EDIT.

These hold for ANY version of src/model.py: fixed split, no holdout or
market data reaching the model, no leakage, valid predictions.
"""

from math import erf, sqrt

import numpy as np
import pandas as pd
import pytest

from src import evaluate as ev
from src import model

TEAMS = ["BOS", "NYK", "LAL", "DEN", "MIA", "CHI", "DAL", "POR", "UTA", "ATL"]


def synthetic_games(n_seasons=4, days=60, seed=0):
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 5, len(TEAMS))
    rows = []
    for s in range(n_seasons):
        year = 2012 + s
        strength = 0.7 * strength + rng.normal(0, 3, len(TEAMS))
        for day in range(days):
            for h, a in rng.permutation(len(TEAMS)).reshape(-1, 2):
                exp = strength[h] - strength[a] + 3
                margin = int(round(rng.normal(exp, 12))) or 1
                p = 0.5 * (1 + erf(exp / (12 * sqrt(2))))
                rows.append({
                    "GAME_ID": f"{year}_{day:03d}_{TEAMS[h]}",
                    "GAME_DATE": (pd.Timestamp(f"{year}-11-01") + pd.Timedelta(days=day)).strftime("%Y-%m-%d"),
                    "SEASON": f"{year}-{(year + 1) % 100:02d}",
                    "SEASON_TYPE": "Regular Season",
                    "HOME_TEAM_ID": TEAMS[h], "AWAY_TEAM_ID": TEAMS[a],
                    "HOME_PTS": 100 + max(margin, 0), "AWAY_PTS": 100 + max(-margin, 0),
                    "MARGIN": margin, "HOME_WIN": int(margin > 0), "NEUTRAL": 0,
                })
    return pd.DataFrame(rows)


def test_split_is_fixed():
    assert ev.VALIDATION_SEASONS == ("2015-16", "2016-17", "2017-18")
    assert ev.HOLDOUT_SEASONS == ("2018-19", "2019-20")
    assert ev.FORBIDDEN_FROM_START_YEAR == 2025
    assert ev.BEATS_MARKET_ALARM == 0.01


def test_load_refuses_2025_26(tmp_path):
    g = synthetic_games(1, days=2)
    g.loc[0, "SEASON"] = "2025-26"
    path = tmp_path / "g.csv"
    g.to_csv(path, index=False)
    with pytest.raises(ValueError):
        ev.load_games(path)


def test_load_drops_holdout(tmp_path):
    g = synthetic_games(1, days=2)
    g["SEASON"] = ["2018-19"] * 5 + ["2017-18"] * (len(g) - 5)
    path = tmp_path / "g.csv"
    g.to_csv(path, index=False)
    assert set(ev.load_games(path)["SEASON"]) == {"2017-18"}


def test_model_never_sees_market_holdout_or_future_seasons():
    seen = []

    def spy(history, test_season):
        seen.append((test_season, history.copy()))
        test = history[history["SEASON"] == test_season]
        return pd.Series(0.6, index=test["GAME_ID"].values)

    g = synthetic_games(n_seasons=6, days=6)
    g["SEASON"] = g["SEASON"].map(dict(zip(sorted(g.SEASON.unique()),
                                           ["2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18"])))
    g["MARKET_PROB_HOME"], g["HOME_ML"], g["AWAY_ML"], g["OVERROUND"] = 0.6, -150.0, 130.0, 0.03
    ev.evaluate(g, predict_season=spy)
    assert [s for s, _ in seen] == list(ev.VALIDATION_SEASONS)
    for season, hist in seen:
        assert not set(ev.MARKET_COLS) & set(hist.columns)
        assert hist["SEASON"].max() == season
        assert not hist["SEASON"].isin(ev.HOLDOUT_SEASONS).any()


def test_model_predictions_valid():
    g = synthetic_games()
    test_season = sorted(g.SEASON.unique())[-1]
    p = model.predict_season(g, test_season)
    test_ids = set(g.loc[g.SEASON == test_season, "GAME_ID"])
    assert set(p.index) == test_ids
    assert ((p > 0) & (p < 1)).all()


def test_model_has_no_leakage():
    """Flipping one test-season result must not change the prediction for any
    game dated on or before that game (including the game itself)."""
    g = synthetic_games()
    test_season = sorted(g.SEASON.unique())[-1]
    base = model.predict_season(g, test_season)
    test = g[g.SEASON == test_season].reset_index(drop=True)
    dates = test.set_index("GAME_ID")["GAME_DATE"]
    for j in (0, len(test) // 3, len(test) // 2, len(test) - 1):
        gid = test.loc[j, "GAME_ID"]
        flipped = g.copy()
        m = flipped.GAME_ID == gid
        flipped.loc[m, "MARGIN"] *= -1
        flipped.loc[m, ["HOME_PTS", "AWAY_PTS"]] = flipped.loc[m, ["AWAY_PTS", "HOME_PTS"]].values
        flipped["HOME_WIN"] = (flipped.MARGIN > 0).astype(int)
        p = model.predict_season(flipped, test_season)
        upto = dates[dates <= dates[gid]].index
        np.testing.assert_allclose(p[upto], base[upto], err_msg=f"leak when flipping {gid}")
