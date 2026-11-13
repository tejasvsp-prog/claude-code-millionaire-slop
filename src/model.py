"""The model under improvement. Experiments edit THIS file (not src/evaluate.py).

Contract used by src/evaluate.py and tests/locked/:

    predict_season(history, test_season) -> pd.Series

`history` holds every game up to and including `test_season` (results
included, market columns removed). Return P(home win) for every game of
`test_season`, indexed by GAME_ID. A prediction for a game may only depend
on games played before it; tests/locked/ checks this by flipping results.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.elo import run_elo
from src.features import ALL_FEATURES, build_features

WARMUP_SEASONS = 1  # first season(s) only warm up Elo/team stats; never trained on
FEATURES = ALL_FEATURES


def make_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


def predict_season(history: pd.DataFrame, test_season: str) -> pd.Series:
    feats = build_features(run_elo(history))
    seasons = sorted(feats["SEASON"].unique())
    train_seasons = seasons[WARMUP_SEASONS : seasons.index(test_season)]
    if not train_seasons:
        raise ValueError(f"no training seasons before {test_season}")
    train = feats[feats["SEASON"].isin(train_seasons)]
    test = feats[feats["SEASON"] == test_season]
    clf = make_classifier().fit(train[FEATURES], train["HOME_WIN"])
    return pd.Series(clf.predict_proba(test[FEATURES])[:, 1], index=test["GAME_ID"].values)
