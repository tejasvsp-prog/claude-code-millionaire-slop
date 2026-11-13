import numpy as np
import pandas as pd
import pytest

from src import backtest as bt
from src.odds import american_to_implied_prob


def test_walk_forward_splits():
    seasons = ["2009-10", "2007-08", "2008-09", "2010-11"]
    splits = bt.walk_forward_splits(seasons, "2009-10", warmup=1)
    assert splits == [(["2008-09"], "2009-10"), (["2008-09", "2009-10"], "2010-11")]
    for train, test in splits:
        assert all(s < test for s in train)
    with pytest.raises(ValueError):
        bt.walk_forward_splits(seasons, "2008-09", warmup=1)


def _bet_frame(p, home_ml, away_ml, home_win):
    return pd.DataFrame(
        {"P": p, "HOME_ML": home_ml, "AWAY_ML": away_ml, "HOME_WIN": home_win,
         "GAME_DATE": "2020-01-01"}
    )


def test_pick_bets_side_ev_and_profit():
    df = _bet_frame(
        p=[0.60, 0.30, 0.50],
        home_ml=[100.0, -150.0, -110.0],
        away_ml=[-120.0, 130.0, -110.0],
        home_win=[1, 1, 0],
    )
    bets = bt.pick_bets(df, "P", threshold=0.0)  # keeps the input index
    # Game 0: home EV = 0.6*2 - 1 = 0.2 -> bet home at +100, wins 1.
    assert bets.loc[0, "BET_SIDE"] == "HOME"
    assert bets.loc[0, "EV"] == pytest.approx(0.2)
    assert bets.loc[0, "PROFIT"] == pytest.approx(1.0)
    # Game 1: away EV = 0.7*2.3 - 1 = 0.61 -> bet away at +130, loses 1.
    assert bets.loc[1, "BET_SIDE"] == "AWAY"
    assert bets.loc[1, "EV"] == pytest.approx(0.61)
    assert bets.loc[1, "PROFIT"] == -1.0
    # Game 2: coin flip at -110 both ways -> negative EV, no bet.
    assert 2 not in bets.index


def test_pick_bets_threshold_and_missing_lines():
    df = _bet_frame([0.6, 0.6], [100.0, np.nan], [-120.0, -120.0], [1, 1])
    assert len(bt.pick_bets(df, "P", 0.0)) == 1
    assert len(bt.pick_bets(df, "P", 0.25)) == 0


def test_no_vig_market_probs_never_bet():
    """With fair (no-vig) probabilities every side has negative EV at the vigged price."""
    rng = np.random.default_rng(1)
    home = rng.choice([-300.0, -200.0, -150.0, -110.0, 120.0, 180.0], 200)
    away = -home - 20  # mirror line shifted by 20 cents of vig (-150 -> +130)
    away = np.where(np.abs(away) < 100, 100.0, away)
    ih = np.array([american_to_implied_prob(x) for x in home])
    ia = np.array([american_to_implied_prob(x) for x in away])
    df = _bet_frame(ih / (ih + ia), home, away, rng.integers(0, 2, 200))
    over = ih + ia - 1
    assert (over > 0).all()
    assert len(bt.pick_bets(df, "P", 0.0)) == 0


def test_roi_summary():
    bets = pd.DataFrame({"PROFIT": [1.0, -1.0, 0.5, -1.0], "WON": [True, False, True, False]})
    s = bt.roi_summary(bets)
    assert s["BETS"] == 4 and s["PROFIT"] == pytest.approx(-0.5)
    assert s["ROI"] == pytest.approx(-0.125) and s["WIN_RATE"] == 0.5
    assert s["ROI_LO"] <= s["ROI"] <= s["ROI_HI"]
    assert bt.roi_summary(bets.iloc[:0])["BETS"] == 0


def _synthetic_games(n_seasons=4, seed=0):
    """Seasons of 10 teams with latent strength; market = true prob with ~4% vig."""
    from math import erf, sqrt

    from src.odds import prob_to_american

    rng = np.random.default_rng(seed)
    teams = list("ABCDEFGHIJ")
    strength = rng.normal(0, 5, len(teams))
    rows = []
    for s in range(n_seasons):
        year = 2010 + s
        strength = 0.7 * strength + rng.normal(0, 3, len(teams))
        for day in range(80):
            order = rng.permutation(len(teams))
            for h, a in order.reshape(-1, 2):
                exp = strength[h] - strength[a] + 3
                margin = int(round(rng.normal(exp, 12))) or 1
                p = 0.5 * (1 + erf(exp / (12 * sqrt(2))))
                p = min(max(p, 0.03), 0.97)
                rows.append({
                    "GAME_ID": f"{year}_{day}_{h}", "GAME_DATE": pd.Timestamp(f"{year}-11-01") + pd.Timedelta(days=day),
                    "SEASON": f"{year}-{(year + 1) % 100:02d}", "SEASON_TYPE": "Regular Season",
                    "HOME_TEAM_ID": teams[h], "AWAY_TEAM_ID": teams[a], "MARGIN": margin,
                    "HOME_WIN": int(margin > 0), "NEUTRAL": 0,
                    "HOME_ML": prob_to_american(min(p * 1.02, 0.99)),
                    "AWAY_ML": prob_to_american(min((1 - p) * 1.02, 0.99)),
                    "MARKET_PROB_HOME": p,
                })
    return pd.DataFrame(rows)


def test_end_to_end_market_unbeatable_on_synthetic_data():
    """Market = true probabilities. A leak-free model must not beat it or profit."""
    preds, metrics, betting = bt.run(_synthetic_games(), "2012-13", warmup=1, thresholds=[0.0])
    allrow = metrics[metrics.SEASON == "ALL"].iloc[0]
    assert allrow.MARKET_LOGLOSS < allrow.MODEL_LOGLOSS
    assert allrow.MARKET_LOGLOSS < allrow.ELO_LOGLOSS
    model = betting[(betting.PROBS == "MODEL")].iloc[0]
    assert model.ROI < bt.ROI_ALARM
    assert set(preds.SEASON) == {"2012-13", "2013-14"}


def test_end_to_end_detects_leak():
    """Sanity check of the check: leaking the result into a feature is caught by ROI."""
    games = _synthetic_games()
    feats = bt.build_features(bt.run_elo(games))
    feats["LEAK"] = feats["HOME_WIN"]
    preds = bt.walk_forward_predict(feats, "2012-13", 1, features=bt.MODEL_FEATURES + ["LEAK"])
    roi = bt.roi_summary(bt.pick_bets(preds, "MODEL_PROB_HOME", 0.0))["ROI"]
    assert roi > bt.ROI_ALARM
