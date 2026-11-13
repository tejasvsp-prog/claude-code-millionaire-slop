import numpy as np
import pandas as pd
import pytest

from src.elo import run_elo
from src.features import ALL_FEATURES, REST_CAP, SHRINK_GAMES, build_features
from src.geo import arena, haversine_km


def _games():
    rows = [
        # id, date, season, home, away, margin
        ("g1", "2020-01-01", "2019-20", "BOS", "NYK", 10),
        ("g2", "2020-01-02", "2019-20", "BOS", "LAL", -4),   # BOS back-to-back
        ("g3", "2020-01-05", "2019-20", "NYK", "BOS", 6),
        ("g4", "2020-01-05", "2019-20", "LAL", "DEN", 3),
        ("g5", "2020-10-20", "2020-21", "BOS", "NYK", 1),    # new season: reset
    ]
    df = pd.DataFrame(rows, columns=["GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM_ID",
                                     "AWAY_TEAM_ID", "MARGIN"])
    df["HOME_WIN"] = (df.MARGIN > 0).astype(int)
    return df


def _feats(games=None):
    return build_features(run_elo(_games() if games is None else games)).set_index("GAME_ID")


def test_first_game_defaults():
    f = _feats()
    g1 = f.loc["g1"]
    assert g1.HOME_REST_DAYS == REST_CAP and g1.HOME_GP == 0
    assert g1.HOME_WIN_PCT == pytest.approx(0.5) and g1.HOME_MARGIN_L10 == 0
    assert g1.ELO_DIFF == 0


def test_rest_and_b2b():
    f = _feats()
    assert f.loc["g2", "HOME_B2B"] == 1 and f.loc["g2", "HOME_REST_DAYS"] == 1
    assert f.loc["g2", "AWAY_B2B"] == 0
    assert f.loc["g3", "AWAY_REST_DAYS"] == 3  # A: Jan 2 -> Jan 5


def test_season_to_date_stats():
    f = _feats()
    # Before g3, A is 1-1 with margins +10, -4.
    assert f.loc["g3", "AWAY_GP"] == 2
    assert f.loc["g3", "AWAY_WIN_PCT"] == pytest.approx((1 + 0.5 * SHRINK_GAMES) / (2 + SHRINK_GAMES))
    assert f.loc["g3", "AWAY_MARGIN_AVG"] == pytest.approx(6 / (2 + SHRINK_GAMES))
    assert f.loc["g3", "AWAY_MARGIN_L10"] == pytest.approx(3.0)


def test_new_season_resets_team_stats():
    g5 = _feats().loc["g5"]
    assert g5.HOME_GP == 0 and g5.AWAY_GP == 0 and g5.HOME_REST_DAYS == REST_CAP
    assert g5.ELO_DIFF != 0  # Elo carries over (with reversion)


def test_no_leakage_from_own_or_later_results():
    """Changing game i's result must not change features of game i or earlier."""
    base = _feats()
    order = list(base.index)
    for i, gid in enumerate(order):
        g = _games()
        g.loc[g.GAME_ID == gid, "MARGIN"] *= -1
        g["HOME_WIN"] = (g.MARGIN > 0).astype(int)
        f = _feats(g)
        for earlier in order[: i + 1]:
            np.testing.assert_allclose(
                f.loc[earlier, ALL_FEATURES].astype(float),
                base.loc[earlier, ALL_FEATURES].astype(float),
                err_msg=f"flipping {gid} changed features of {earlier}",
            )


def test_same_day_double_game_rejected():
    g = _games()
    g.loc[3, ["HOME_TEAM_ID", "AWAY_TEAM_ID"]] = ["BOS", "DEN"]  # BOS plays twice on Jan 5
    with pytest.raises(ValueError):
        _feats(g)


def test_travel_and_time_zone():
    f = _feats()
    # g1 at BOS, g3 at NYK: NYK played g1 in Boston, then hosts g3 at home.
    bos, nyk = arena("BOS", 2019), arena("NYK", 2019)
    assert f.loc["g3", "HOME_TRAVEL_KM"] == pytest.approx(haversine_km(*bos[:2], *nyk[:2]))
    # LAL: g2 in Boston, then g4 at home in LA: 3 time zones.
    assert f.loc["g4", "HOME_TZ_CHANGE"] == 3
    assert f.loc["g4", "HOME_TRAVEL_KM"] > 4000
    assert f.loc["g1", "HOME_TRAVEL_KM"] == 0 and f.loc["g1", "AWAY_TZ_CHANGE"] == 0


def test_three_in_four():
    rows = [("x1", "2020-01-01", "BOS", "NYK"), ("x2", "2020-01-02", "LAL", "BOS"),
            ("x3", "2020-01-04", "BOS", "DEN"), ("x4", "2020-01-06", "BOS", "NYK")]
    g = pd.DataFrame(rows, columns=["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID"])
    g["SEASON"], g["MARGIN"] = "2019-20", 5
    g["HOME_WIN"] = 1
    f = _feats(g)
    assert f.loc["x3", "HOME_3IN4"] == 1   # Jan 1, 2, 4
    assert f.loc["x4", "HOME_3IN4"] == 0   # only Jan 4 within previous 3 days
    assert f.loc["x3", "AWAY_3IN4"] == 0


def test_altitude():
    f = _feats()
    assert f.loc["g4", "ALTITUDE"] == 0  # DEN is away
    g = _games()
    g.loc[3, ["HOME_TEAM_ID", "AWAY_TEAM_ID"]] = ["DEN", "LAL"]
    assert _feats(g).loc["g4", "ALTITUDE"] == 1


def test_geo_helpers():
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.1)
    assert arena("OKC", 2007)[2] == -8 and arena("OKC", 2008)[2] == -6  # Seattle -> OKC
    with pytest.raises(KeyError):
        arena("XXX", 2010)
