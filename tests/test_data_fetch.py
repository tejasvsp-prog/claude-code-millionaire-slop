import pandas as pd
import pytest

from src import data_fetch as dfm


def _logs():
    # Two games: BOS hosts NYK (BOS wins), LAL @ GSW (LAL wins on the road).
    return pd.DataFrame(
        {
            "GAME_ID": ["0022300001", "0022300001", "0022300002", "0022300002"],
            "GAME_DATE": ["2023-10-24", "2023-10-24", "2023-10-25", "2023-10-25"],
            "TEAM_ID": [1, 2, 3, 4],
            "TEAM_ABBREVIATION": ["BOS", "NYK", "GSW", "LAL"],
            "MATCHUP": ["BOS vs. NYK", "NYK @ BOS", "GSW vs. LAL", "LAL @ GSW"],
            "PTS": [110, 100, 99, 105],
            "SEASON": "2023-24",
            "SEASON_TYPE": dfm.REGULAR_SEASON,
        }
    )


def test_season_helpers():
    assert dfm.season_str(1999) == "1999-00"
    assert dfm.season_str(2023) == "2023-24"
    assert dfm.season_range("2021-22", "2023-24") == ["2021-22", "2022-23", "2023-24"]
    with pytest.raises(ValueError):
        dfm.season_range("2023-24", "2021-22")


def test_build_games_home_away():
    g = dfm.build_games(_logs())
    assert len(g) == 2
    first = g.iloc[0]
    assert (first.HOME_TEAM_ABBREVIATION, first.AWAY_TEAM_ABBREVIATION) == ("BOS", "NYK")
    assert first.HOME_WIN == 1 and first.MARGIN == 10
    second = g.iloc[1]
    assert (second.HOME_TEAM_ABBREVIATION, second.AWAY_TEAM_ABBREVIATION) == ("GSW", "LAL")
    assert second.HOME_WIN == 0 and second.MARGIN == -6


def test_build_games_drops_incomplete_game():
    logs = pd.concat(
        [_logs(), pd.DataFrame([{**_logs().iloc[0].to_dict(), "GAME_ID": "0022300003"}])]
    )
    assert len(dfm.build_games(logs)) == 2


def test_build_games_sorted_by_date():
    g = dfm.build_games(_logs().iloc[::-1])
    assert g["GAME_DATE"].is_monotonic_increasing


def test_build_games_rejects_unknown_matchup():
    logs = _logs()
    logs.loc[0, "MATCHUP"] = "BOS - NYK"
    with pytest.raises(ValueError):
        dfm.build_games(logs)


def test_fetch_caches_and_sleeps(tmp_path):
    calls, sleeps = [], []

    def fake_fetch(season, st):
        calls.append((season, st))
        return _logs().drop(columns=["SEASON", "SEASON_TYPE"])

    kw = dict(raw_dir=tmp_path, fetcher=fake_fetch, sleep=sleeps.append)
    a = dfm.fetch_team_gamelog("2023-24", **kw)
    b = dfm.fetch_team_gamelog("2023-24", **kw)

    assert calls == [("2023-24", dfm.REGULAR_SEASON)]  # second call hit the cache
    assert sleeps == [dfm.SLEEP_SECONDS]
    assert dfm.raw_path("2023-24", raw_dir=tmp_path).exists()
    pd.testing.assert_frame_equal(a, b)
    assert a["GAME_ID"].iloc[0] == "0022300001"  # leading zeros preserved


def test_fetch_retries_then_succeeds(tmp_path):
    attempts = []

    def flaky(season, st):
        attempts.append(1)
        if len(attempts) < 2:
            raise ConnectionError("boom")
        return _logs()

    dfm.fetch_team_gamelog("2023-24", raw_dir=tmp_path, fetcher=flaky, sleep=lambda s: None)
    assert len(attempts) == 2


def test_fetch_gives_up(tmp_path):
    def broken(season, st):
        raise ConnectionError("down")

    with pytest.raises(RuntimeError):
        dfm.fetch_team_gamelog("2023-24", raw_dir=tmp_path, fetcher=broken, sleep=lambda s: None)
    assert not dfm.raw_path("2023-24", raw_dir=tmp_path).exists()
