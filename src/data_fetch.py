"""Pull NBA team game logs from nba_api, cache raw CSVs, build a games table.

Raw pulls (one row per team per game) are cached to
    data/raw/team_gamelog_{season}_{season_type}.csv
and never re-fetched unless `force=True`. `build_games` turns them into one
row per game (home vs away) saved to data/processed/games.csv.

Usage:
    python -m src.data_fetch 2015-16 2024-25
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

SLEEP_SECONDS = 1.5  # pause after every live nba_api call
MAX_RETRIES = 3
REGULAR_SEASON = "Regular Season"
PLAYOFFS = "Playoffs"

log = logging.getLogger(__name__)


def season_str(start_year: int) -> str:
    """2023 -> '2023-24' (nba_api season format)."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_range(first: str, last: str) -> list[str]:
    """Inclusive list of seasons, e.g. ('2021-22', '2023-24')."""
    a, b = int(first[:4]), int(last[:4])
    if b < a:
        raise ValueError(f"last season {last} is before first {first}")
    return [season_str(y) for y in range(a, b + 1)]


def raw_path(season: str, season_type: str = REGULAR_SEASON, raw_dir: Path = RAW_DIR) -> Path:
    slug = season_type.lower().replace(" ", "_")
    return raw_dir / f"team_gamelog_{season}_{slug}.csv"


def _fetch_from_api(season: str, season_type: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    endpoint = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation="T",
        timeout=60,
    )
    return endpoint.get_data_frames()[0]


def fetch_team_gamelog(
    season: str,
    season_type: str = REGULAR_SEASON,
    *,
    force: bool = False,
    raw_dir: Path = RAW_DIR,
    fetcher: Callable[[str, str], pd.DataFrame] = _fetch_from_api,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Return the team game log for a season, reading the CSV cache if present.

    Every live call is followed by a sleep (rate limit). Failed calls are
    retried with backoff.
    """
    path = raw_path(season, season_type, raw_dir)
    if path.exists() and not force:
        log.info("cache hit %s", path.name)
        return pd.read_csv(path, dtype={"GAME_ID": str})

    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = fetcher(season, season_type)
            sleep(SLEEP_SECONDS)
            break
        except Exception as err:  # network / JSON errors from nba_api
            last_err = err
            log.warning("fetch %s %s failed (attempt %d): %s", season, season_type, attempt, err)
            sleep(SLEEP_SECONDS * 2**attempt)
    else:
        raise RuntimeError(f"Failed to fetch {season} {season_type}") from last_err

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("cached %s (%d rows)", path.name, len(df))
    # Re-read so the cached and fresh paths return identical dtypes.
    return pd.read_csv(path, dtype={"GAME_ID": str})


def fetch_seasons(
    seasons: Iterable[str],
    season_types: Iterable[str] = (REGULAR_SEASON, PLAYOFFS),
    **kwargs,
) -> pd.DataFrame:
    frames = []
    for season in seasons:
        for st in season_types:
            df = fetch_team_gamelog(season, st, **kwargs)
            frames.append(df.assign(SEASON=season, SEASON_TYPE=st))
    return pd.concat(frames, ignore_index=True)


_TEAM_COLS = ["TEAM_ID", "TEAM_ABBREVIATION", "PTS"]


def build_games(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Collapse team-level rows into one row per game with home/away columns.

    Home team is identified from MATCHUP: 'BOS vs. NYK' = BOS at home,
    'NYK @ BOS' = NYK away. Games without exactly one home and one away
    row (e.g. data glitches) are dropped with a warning.
    """
    df = team_logs.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
    df["IS_HOME"] = df["MATCHUP"].str.contains(" vs. ", regex=False)
    is_away = df["MATCHUP"].str.contains(" @ ", regex=False)
    bad = ~(df["IS_HOME"] ^ is_away)
    if bad.any():
        raise ValueError(f"Unrecognised MATCHUP values: {df.loc[bad, 'MATCHUP'].unique()[:5]}")

    counts = df.groupby("GAME_ID")["IS_HOME"].agg(["sum", "size"])
    ok = counts.index[(counts["sum"] == 1) & (counts["size"] == 2)]
    dropped = len(counts) - len(ok)
    if dropped:
        log.warning("dropping %d malformed games", dropped)
    df = df[df["GAME_ID"].isin(ok)]

    keep = ["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE"]
    keep = [c for c in keep if c in df.columns]
    home = df[df["IS_HOME"]][keep + _TEAM_COLS].rename(
        columns={c: f"HOME_{c}" for c in _TEAM_COLS}
    )
    away = df[~df["IS_HOME"]][["GAME_ID"] + _TEAM_COLS].rename(
        columns={c: f"AWAY_{c}" for c in _TEAM_COLS}
    )
    games = home.merge(away, on="GAME_ID", validate="one_to_one")
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games["HOME_WIN"] = (games["HOME_PTS"] > games["AWAY_PTS"]).astype(int)
    games["MARGIN"] = games["HOME_PTS"] - games["AWAY_PTS"]
    return games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("first_season", help="e.g. 2015-16")
    p.add_argument("last_season", help="e.g. 2024-25")
    p.add_argument("--no-playoffs", action="store_true")
    p.add_argument("--force", action="store_true", help="ignore cache and re-fetch")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    types = (REGULAR_SEASON,) if args.no_playoffs else (REGULAR_SEASON, PLAYOFFS)
    logs = fetch_seasons(
        season_range(args.first_season, args.last_season), types, force=args.force
    )
    games = build_games(logs)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "games.csv"
    games.to_csv(out, index=False)
    log.info("wrote %s (%d games)", out, len(games))


if __name__ == "__main__":
    main()
