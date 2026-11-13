import numpy as np
import pandas as pd
import pytest

from src.elo import (
    EloParams,
    elo_update,
    expected_score,
    mov_multiplier,
    revert_to_mean,
    run_elo,
    season_metrics,
)


def test_expected_score():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1600, 1500) == pytest.approx(1 / (1 + 10 ** (-0.25)))
    assert expected_score(1700, 1500) + expected_score(1500, 1700) == pytest.approx(1.0)


def test_home_advantage_in_prob():
    p, _, _ = elo_update(1500, 1500, margin=5)
    assert p == pytest.approx(expected_score(1600, 1500))  # ~0.64
    p_no_hca, _, _ = elo_update(1500, 1500, 5, EloParams(home_adv=0))
    assert p_no_hca == pytest.approx(0.5)


def test_update_is_zero_sum_and_directional():
    p, h, a = elo_update(1500, 1500, margin=10)
    assert h > 1500 > a
    assert (h - 1500) == pytest.approx(1500 - a)
    _, h2, a2 = elo_update(1500, 1500, margin=-10)
    assert h2 < 1500 < a2


def test_update_without_mov_matches_classic_elo():
    params = EloParams(use_mov=False, home_adv=0)
    _, h, _ = elo_update(1500, 1500, margin=30, params=params)
    assert h == pytest.approx(1500 + 20 * 0.5)


def test_mov_multiplier():
    # Bigger wins move ratings more; favorites' wins are damped vs upsets.
    assert mov_multiplier(20, 0) > mov_multiplier(5, 0)
    assert mov_multiplier(10, 200) < mov_multiplier(10, -200)
    with pytest.raises(ValueError):
        mov_multiplier(0, 0)


def test_revert_to_mean():
    assert revert_to_mean(1700) == pytest.approx(1650)
    assert revert_to_mean(1300) == pytest.approx(1350)


def _games():
    # A=1 beats everyone; season 2 starts after 4 games.
    rows = [
        ("g1", "2022-10-20", "2022-23", 1, 2, 10),
        ("g2", "2022-10-21", "2022-23", 3, 1, -8),
        ("g3", "2022-10-22", "2022-23", 2, 3, 3),
        ("g4", "2022-10-23", "2022-23", 1, 3, 15),
        ("g5", "2023-10-20", "2023-24", 2, 1, -4),
        ("g6", "2023-10-21", "2023-24", 1, 2, 7),
    ]
    df = pd.DataFrame(
        rows, columns=["GAME_ID", "GAME_DATE", "SEASON", "HOME_TEAM_ID", "AWAY_TEAM_ID", "MARGIN"]
    )
    df["HOME_WIN"] = (df["MARGIN"] > 0).astype(int)
    return df


def test_run_elo_first_game_uses_initial_ratings():
    out = run_elo(_games())
    assert out.loc[0, "HOME_ELO_PRE"] == 1500 and out.loc[0, "AWAY_ELO_PRE"] == 1500
    assert out.loc[0, "ELO_PROB_HOME"] == pytest.approx(expected_score(1600, 1500))


def test_run_elo_no_leakage():
    """Changing a game's result must not change that game's (or earlier) predictions."""
    base = run_elo(_games())
    for i in range(len(base)):
        flipped = _games()
        flipped.loc[i, "MARGIN"] = -flipped.loc[i, "MARGIN"]
        out = run_elo(flipped)
        np.testing.assert_allclose(out["ELO_PROB_HOME"][: i + 1], base["ELO_PROB_HOME"][: i + 1])
        if i + 1 < len(base):
            assert not np.allclose(out["ELO_PROB_HOME"][i + 1 :], base["ELO_PROB_HOME"][i + 1 :])


def test_run_elo_sorts_by_date():
    shuffled = _games().sample(frac=1, random_state=0)
    pd.testing.assert_series_equal(
        run_elo(shuffled)["ELO_PROB_HOME"], run_elo(_games())["ELO_PROB_HOME"]
    )


def test_run_elo_season_reversion():
    out = run_elo(_games())
    # Team 1's rating after g4 (end of season 1), reverted, should be g5's AWAY_ELO_PRE.
    params = EloParams()
    last = out.iloc[3]
    _, team1_after, _ = elo_update(last.HOME_ELO_PRE, last.AWAY_ELO_PRE, last.MARGIN, params)
    assert out.loc[4, "AWAY_ELO_PRE"] == pytest.approx(revert_to_mean(team1_after, params))


def test_run_elo_validates_input():
    with pytest.raises(ValueError):
        run_elo(_games().drop(columns=["MARGIN"]))
    g = _games()
    g.loc[0, "MARGIN"] = 0
    with pytest.raises(ValueError):
        run_elo(g)


def test_season_metrics_shape():
    m = season_metrics(run_elo(_games()))
    assert list(m["SEASON"]) == ["2022-23", "2023-24", "ALL"]
    assert m.loc[m.SEASON == "ALL", "N"].item() == 6


def test_neutral_site_has_no_home_advantage():
    p, _, _ = elo_update(1500, 1500, 5, neutral=True)
    assert p == pytest.approx(0.5)
    g = _games()
    g["NEUTRAL"] = [1, 0, 0, 0, 0, 0]
    assert run_elo(g).loc[0, "ELO_PROB_HOME"] == pytest.approx(0.5)
