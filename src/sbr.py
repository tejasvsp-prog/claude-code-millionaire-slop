"""Load Sports Book Reviews (SBR) NBA odds files into the games table.

Source files: data/raw/sbr/nba_odds_YYYY-YY.csv, the SBR season archives
for 2007-08 .. 2019-20 (2019-20 stops in late January 2020). Downloaded
(once, then cached) from the public repo
github.com/Joe-Ferrara/predicting-nba-games (odds_data/), pinned to a commit.

Each game is two consecutive rows (visitor then home, or two 'N' rows for
a neutral-site game) with columns:
    Date (MMDD), Rot, VH, Team, 1st..4th, Final, Open, Close, ML, 2H
`ML` is the closing moneyline (American odds). We only use Final and ML.

Output columns match src.data_fetch.build_games, plus market columns:
    HOME_ML, AWAY_ML, OVERROUND, MARKET_PROB_HOME (no-vig), NEUTRAL

Usage:
    python -m src.sbr
"""

from __future__ import annotations

import argparse
import logging
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.odds import american_to_implied_prob

ROOT = Path(__file__).resolve().parents[1]
SBR_DIR = ROOT / "data" / "raw" / "sbr"
PROCESSED_DIR = ROOT / "data" / "processed"

log = logging.getLogger(__name__)

# Franchise-level IDs, so relocations keep their Elo history.
TEAM_MAP = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "NewJersey": "BKN",
    "Charlotte": "CHA", "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL",
    "Denver": "DEN", "Detroit": "DET", "GoldenState": "GSW", "Houston": "HOU",
    "Indiana": "IND", "LAClippers": "LAC", "LA Clippers": "LAC", "LALakers": "LAL",
    "Memphis": "MEM", "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN",
    "NewOrleans": "NOP", "NewYork": "NYK", "OklahomaCity": "OKC",
    "Oklahoma City": "OKC", "Seattle": "OKC", "Orlando": "ORL",
    "Philadelphia": "PHI", "Phoenix": "PHX", "Portland": "POR",
    "Sacramento": "SAC", "SanAntonio": "SAS", "Toronto": "TOR", "Utah": "UTA",
    "Washington": "WAS",
}

# Last day of the regular season, keyed by season start year (from the
# NBA schedule; the same table ships with the source repo).
REGULAR_SEASON_END = {
    2007: "2008-04-16", 2008: "2009-04-15", 2009: "2010-04-14", 2010: "2011-04-13",
    2011: "2012-04-26", 2012: "2013-04-17", 2013: "2014-04-16", 2014: "2015-04-15",
    2015: "2016-04-13", 2016: "2017-04-12", 2017: "2018-04-11", 2018: "2019-04-10",
    2019: "2020-03-11",
}

# Known scoring errors in the raw files: (date, away, home) -> (away pts, home pts).
SCORE_FIXES = {
    # Raw Final reads 10-12; quarters sum to 101-101 (OT). Suns won 113-111.
    ("2016-12-13", "NYK", "PHX"): (111, 113),
}

# Pinned to a fixed commit so the raw data can't change underneath us.
SOURCE_URL = (
    "https://raw.githubusercontent.com/Joe-Ferrara/predicting-nba-games/"
    "6214a2171ac5644b74fac5f2fdb0ec68a6b47119/odds_data/nba_odds_{season}.csv"
)
SEASONS = [f"{y}-{(y + 1) % 100:02d}" for y in range(2007, 2020)]


def download_sbr(sbr_dir: Path = SBR_DIR, sleep_seconds: float = 1.0, force: bool = False) -> None:
    """Download any missing season files into sbr_dir (cached; sleeps between requests)."""
    sbr_dir.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        path = sbr_dir / f"nba_odds_{season}.csv"
        if path.exists() and not force:
            continue
        url = SOURCE_URL.format(season=season)
        log.info("downloading %s", url)
        with urllib.request.urlopen(url, timeout=60) as resp:
            path.write_bytes(resp.read())
        time.sleep(sleep_seconds)


# A two-way closing line with an overround outside this range is a data error.
MIN_OVERROUND, MAX_OVERROUND = 0.0, 0.10


def parse_date(mmdd: int | str, start_year: int) -> pd.Timestamp:
    """SBR dates are MMDD with no year: Aug-Dec belong to the start year."""
    s = str(int(mmdd)).zfill(4)
    month, day = int(s[:2]), int(s[2:])
    year = start_year if month >= 8 else start_year + 1
    return pd.Timestamp(year=year, month=month, day=day)


def _to_ml(x) -> float:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v) or -100 < v < 100:
        return np.nan
    return float(v)


def parse_season_file(path: Path) -> pd.DataFrame:
    m = re.search(r"(\d{4})-(\d{2})", path.name)
    if not m:
        raise ValueError(f"cannot read season from {path.name}")
    start = int(m.group(1))
    season = f"{start}-{m.group(2)}"
    raw = pd.read_csv(path, dtype=str)
    if len(raw) % 2:
        raise ValueError(f"{path.name}: odd number of rows")

    a, b = raw.iloc[0::2].reset_index(drop=True), raw.iloc[1::2].reset_index(drop=True)
    pair_ok = ((a.VH == "V") & (b.VH == "H")) | ((a.VH == "N") & (b.VH == "N"))
    if not pair_ok.all():
        raise ValueError(f"{path.name}: bad V/H pairing at games {list(np.where(~pair_ok)[0][:5])}")
    # Compare mod 100: the raw files contain at least one truncated Rot ("33" for 533).
    rot_ok = b.Rot.astype(int) % 100 == (a.Rot.astype(int) + 1) % 100
    if not rot_ok.all():
        raise ValueError(f"{path.name}: non-consecutive rotation numbers at {list(np.where(~rot_ok)[0][:5])}")
    date_diff = a.Date != b.Date
    if date_diff.any():
        # Seen for late games straddling midnight; the visitor row's date is used.
        log.warning("%s: %d games with mismatched row dates (using visitor date)",
                    path.name, int(date_diff.sum()))
    unknown = (set(a.Team) | set(b.Team)) - TEAM_MAP.keys()
    if unknown:
        raise ValueError(f"{path.name}: unknown teams {sorted(unknown)}")

    end = pd.Timestamp(REGULAR_SEASON_END[start])
    dates = [parse_date(d, start) for d in a.Date]
    away_ids, home_ids = a.Team.map(TEAM_MAP), b.Team.map(TEAM_MAP)
    games = pd.DataFrame(
        {
            "GAME_ID": [f"{d:%Y%m%d}_{x}_{y}" for d, x, y in zip(dates, away_ids, home_ids)],
            "GAME_DATE": dates,
            "SEASON": season,
            "SEASON_TYPE": ["Regular Season" if d <= end else "Playoffs" for d in dates],
            "HOME_TEAM_ID": home_ids,
            "AWAY_TEAM_ID": away_ids,
            "HOME_TEAM_ABBREVIATION": home_ids,
            "AWAY_TEAM_ABBREVIATION": away_ids,
            "HOME_PTS": b.Final.astype(int),
            "AWAY_PTS": a.Final.astype(int),
            "NEUTRAL": (a.VH == "N").astype(int),
            "HOME_ML": b.ML.map(_to_ml),
            "AWAY_ML": a.ML.map(_to_ml),
        }
    )
    qa = a[["1st", "2nd", "3rd", "4th"]].astype(int).sum(axis=1)
    qb = b[["1st", "2nd", "3rd", "4th"]].astype(int).sum(axis=1)
    for i, row in games.iterrows():
        key = (f"{row.GAME_DATE:%Y-%m-%d}", row.AWAY_TEAM_ID, row.HOME_TEAM_ID)
        if key in SCORE_FIXES:
            games.loc[i, ["AWAY_PTS", "HOME_PTS"]] = SCORE_FIXES[key]
        elif row.AWAY_PTS < qa[i] or row.HOME_PTS < qb[i]:
            raise ValueError(f"{path.name}: final below quarter sum for {key}")
    return games


def add_market_probs(games: pd.DataFrame) -> pd.DataFrame:
    """Add OVERROUND and MARKET_PROB_HOME (proportional no-vig) from closing MLs.

    Lines that are missing or imply an overround outside
    [MIN_OVERROUND, MAX_OVERROUND] get NaN market probability.
    """
    df = games.copy()
    ih = df["HOME_ML"].map(lambda x: american_to_implied_prob(x) if pd.notna(x) else np.nan)
    ia = df["AWAY_ML"].map(lambda x: american_to_implied_prob(x) if pd.notna(x) else np.nan)
    df["OVERROUND"] = ih + ia - 1.0
    bad = df["OVERROUND"].isna() | (df["OVERROUND"] < MIN_OVERROUND) | (df["OVERROUND"] > MAX_OVERROUND)
    df["MARKET_PROB_HOME"] = (ih / (ih + ia)).where(~bad)
    for col in ("HOME_ML", "AWAY_ML", "OVERROUND"):
        df[col] = df[col].where(~bad)
    if bad.any():
        log.warning("%d games have no usable closing line", int(bad.sum()))
    return df


def load_sbr(sbr_dir: Path = SBR_DIR) -> pd.DataFrame:
    files = sorted(sbr_dir.glob("nba_odds_*.csv"))
    if not files:
        raise FileNotFoundError(f"no nba_odds_*.csv files in {sbr_dir}")
    games = pd.concat([parse_season_file(f) for f in files], ignore_index=True)
    if games["GAME_ID"].duplicated().any():
        raise ValueError(f"duplicate games: {games.GAME_ID[games.GAME_ID.duplicated()].tolist()[:5]}")
    games["HOME_WIN"] = (games["HOME_PTS"] > games["AWAY_PTS"]).astype(int)
    games["MARGIN"] = games["HOME_PTS"] - games["AWAY_PTS"]
    # A tie means the file stored the end-of-regulation score of an OT game,
    # so the winner is unknown. Drop rather than guess.
    ties = games["MARGIN"] == 0
    if ties.any():
        log.warning("dropping %d tied (OT score missing) games: %s",
                    int(ties.sum()), games.loc[ties, "GAME_ID"].tolist())
        games = games[~ties]
    games = add_market_probs(games)
    return games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build games.csv from SBR odds files")
    p.add_argument("--sbr-dir", type=Path, default=SBR_DIR)
    p.add_argument("--out", type=Path, default=PROCESSED_DIR / "games.csv")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_sbr(args.sbr_dir)
    games = load_sbr(args.sbr_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(args.out, index=False)
    log.info("wrote %s (%d games, %d with a closing line)",
             args.out, len(games), int(games.MARKET_PROB_HOME.notna().sum()))


if __name__ == "__main__":
    main()
