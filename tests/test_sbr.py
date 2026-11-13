import numpy as np
import pandas as pd
import pytest

from src import sbr
from src.odds import american_to_implied_prob

HEADER = "Date,Rot,VH,Team,1st,2nd,3rd,4th,Final,Open,Close,ML,2H\n"


def _write(tmp_path, body, name="nba_odds_2016-17.csv"):
    p = tmp_path / name
    p.write_text(HEADER + body)
    return p


def test_parse_date_rolls_year():
    assert sbr.parse_date("1025", 2016) == pd.Timestamp("2016-10-25")
    assert sbr.parse_date(105, 2016) == pd.Timestamp("2017-01-05")
    assert sbr.parse_date("612", 2016) == pd.Timestamp("2017-06-12")


def test_parse_season_file_basic(tmp_path):
    p = _write(
        tmp_path,
        "1025,501,V,NewYork,20,20,20,20,80,200,201,150,100\n"
        "1025,502,H,Cleveland,30,30,30,27,117,7,7.5,-170,3\n"
        "415,503,V,Seattle,25,25,25,25,100,200,201,-120,100\n"
        "415,504,H,NewJersey,20,20,20,20,80,1,1,100,1\n",
    )
    g = sbr.parse_season_file(p)
    assert list(g.HOME_TEAM_ID) == ["CLE", "BKN"]
    assert list(g.AWAY_TEAM_ID) == ["NYK", "OKC"]
    assert list(g.SEASON_TYPE) == ["Regular Season", "Playoffs"]  # 2017-04-15 > 04-12
    assert g.loc[0, "HOME_ML"] == -170 and g.loc[0, "AWAY_ML"] == 150
    assert g.loc[0, "GAME_ID"] == "20161025_NYK_CLE"


def test_score_fix_and_bad_final(tmp_path):
    p = _write(
        tmp_path,
        "1213,709,V,NewYork,25,21,27,28,10,3.5,2,-135,4\n"
        "1213,710,H,Phoenix,31,23,31,16,12,221,218.5,115,108.5\n",
    )
    g = sbr.parse_season_file(p)
    assert (g.AWAY_PTS.item(), g.HOME_PTS.item()) == (111, 113)

    p2 = _write(
        tmp_path,
        "1214,709,V,NewYork,25,21,27,28,10,3.5,2,-135,4\n"
        "1214,710,H,Phoenix,31,23,31,16,12,221,218.5,115,108.5\n",
    )
    with pytest.raises(ValueError, match="quarter sum"):
        sbr.parse_season_file(p2)


def test_bad_pairing_and_unknown_team(tmp_path):
    p = _write(
        tmp_path,
        "1025,501,H,NewYork,20,20,20,20,80,200,201,150,100\n"
        "1025,502,V,Cleveland,30,30,30,27,117,7,7.5,-170,3\n",
    )
    with pytest.raises(ValueError, match="pairing"):
        sbr.parse_season_file(p)
    p = _write(
        tmp_path,
        "1025,501,V,Gotham,20,20,20,20,80,200,201,150,100\n"
        "1025,502,H,Cleveland,30,30,30,27,117,7,7.5,-170,3\n",
    )
    with pytest.raises(ValueError, match="unknown teams"):
        sbr.parse_season_file(p)


def _market(home_ml, away_ml):
    return sbr.add_market_probs(pd.DataFrame({"HOME_ML": home_ml, "AWAY_ML": away_ml}))


def test_market_probs_no_vig():
    m = _market([-110.0, -200.0], [-110.0, 170.0])
    assert m.MARKET_PROB_HOME[0] == pytest.approx(0.5)
    ih, ia = american_to_implied_prob(-200), american_to_implied_prob(170)
    assert m.MARKET_PROB_HOME[1] == pytest.approx(ih / (ih + ia))
    assert m.OVERROUND[0] == pytest.approx(220 / 210 - 1)


def test_market_probs_reject_bad_lines():
    # Missing line; negative overround (both +150: arbitrage = data error);
    # absurd overround (both -300).
    m = _market([np.nan, 150.0, -300.0, -110.0], [-110.0, 150.0, -300.0, -110.0])
    assert m.MARKET_PROB_HOME[:3].isna().all()
    assert m.HOME_ML[:3].isna().all()
    assert m.MARKET_PROB_HOME[3] == pytest.approx(0.5)


def test_to_ml():
    assert sbr._to_ml("-150") == -150
    assert np.isnan(sbr._to_ml("NL"))
    assert np.isnan(sbr._to_ml("8"))  # |ML| < 100 is a typo


def test_load_sbr_drops_ties(tmp_path):
    _write(
        tmp_path,
        "1025,501,V,NewYork,20,20,20,20,80,200,201,150,100\n"
        "1025,502,H,Cleveland,30,30,30,27,117,7,7.5,-170,3\n"
        "1026,503,V,NewYork,25,25,25,20,95,200,201,150,100\n"
        "1026,504,H,Phoenix,25,25,25,20,95,7,7.5,-170,3\n",
    )
    g = sbr.load_sbr(tmp_path)
    assert len(g) == 1 and g.HOME_WIN.item() == 1 and g.MARGIN.item() == 37
