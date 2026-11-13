"""Walk-forward backtest: model vs Elo vs no-vig closing market.

For each test season S (from --first-test on), a logistic regression is
fit on every season strictly before S (skipping the warm-up season, where
Elo and team stats are still settling), then used to predict S. No random
splits, no peeking at S.

Reports, on games that have a closing line:
  * log loss / Brier score per season for MODEL, ELO and MARKET
  * a flat-stake betting simulation at closing moneylines: bet the side
    with the higher expected value when EV > threshold

Caveat: the closing line is the comparator AND the price we "bet" at.
In practice you can't reliably get the close, so this is an upper bound
on what the model could earn. If ROI comes out above ~10%, treat it as a
bug or leakage until proven otherwise (a warning is printed).

Usage:
    python -m src.sbr && python -m src.backtest
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.elo import run_elo
from src.features import ALL_FEATURES, build_features
from src.metrics import brier_score, log_loss
from src.odds import american_to_decimal, bet_profit

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

ROI_ALARM = 0.10
PROB_COLS = {
    "MODEL": "MODEL_PROB_HOME",          # features only, no market info
    "BLEND": "BLEND_PROB_HOME",          # market logit + features
    "MKT_RECAL": "MKT_RECAL_PROB_HOME",  # market logit only, refit (bias correction)
    "ELO": "ELO_PROB_HOME",
    "MARKET": "MARKET_PROB_HOME",
}
# name -> feature list for the walk-forward models
MODEL_SPECS = {
    "MODEL": ALL_FEATURES,
    "BLEND": ["MARKET_LOGIT"] + ALL_FEATURES,
    "MKT_RECAL": ["MARKET_LOGIT"],
}

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- splits / model


def walk_forward_splits(
    seasons: list[str], first_test: str, warmup: int = 1
) -> list[tuple[list[str], str]]:
    """[(train_seasons, test_season), ...] with train strictly before test."""
    seasons = sorted(seasons)
    if first_test not in seasons:
        raise ValueError(f"{first_test} not in {seasons}")
    start = seasons.index(first_test)
    if start <= warmup:
        raise ValueError("need at least one training season after the warm-up")
    return [(seasons[warmup:i], seasons[i]) for i in range(start, len(seasons))]


def make_model(c: float = 1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))


def walk_forward_predict(
    df: pd.DataFrame,
    first_test: str,
    warmup: int = 1,
    features: list[str] = ALL_FEATURES,
    out_col: str = "MODEL_PROB_HOME",
) -> pd.DataFrame:
    """Return test-season rows with `out_col` filled in.

    Rows with NaN in any feature (e.g. no market line) are left out of
    training, and get NaN predictions.
    """
    out = []
    for train_seasons, test_season in walk_forward_splits(
        df["SEASON"].unique().tolist(), first_test, warmup
    ):
        train = df[df["SEASON"].isin(train_seasons)].dropna(subset=features)
        test = df[df["SEASON"] == test_season].copy()
        assert train["GAME_DATE"].max() < test["GAME_DATE"].min(), "train overlaps test"
        model = make_model().fit(train[features], train["HOME_WIN"])
        ok = test[features].notna().all(axis=1)
        test[out_col] = np.nan
        test.loc[ok, out_col] = model.predict_proba(test.loc[ok, features])[:, 1]
        log.info("%s test %s: trained on %d games (%s..%s)",
                 out_col, test_season, len(train), train_seasons[0], train_seasons[-1])
        out.append(test)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- scoring


def compare_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-season and overall log loss / Brier for MODEL, ELO, MARKET.

    Only games with a market line are scored, so all three use the same games.
    """
    cols = {k: v for k, v in PROB_COLS.items() if v in df.columns}
    d = df.dropna(subset=list(cols.values()))
    rows = []
    for season, g in list(d.groupby("SEASON", sort=True)) + [("ALL", d)]:
        row = {"SEASON": season, "N": len(g)}
        for name, col in cols.items():
            row[f"{name}_LOGLOSS"] = log_loss(g["HOME_WIN"], g[col])
            row[f"{name}_BRIER"] = brier_score(g["HOME_WIN"], g[col])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- betting


def pick_bets(df: pd.DataFrame, prob_col: str, threshold: float = 0.0) -> pd.DataFrame:
    """Choose at most one side per game: the one with higher EV, if EV > threshold.

    EV per unit staked = p * decimal_odds - 1, at the closing moneyline.
    Returns one row per bet with BET_SIDE, BET_ODDS, BET_PROB, EV, WON, PROFIT.
    """
    d = df[df["HOME_ML"].notna() & df["AWAY_ML"].notna() & df[prob_col].notna()].copy()
    p = d[prob_col]
    dec_h = d["HOME_ML"].map(american_to_decimal)
    dec_a = d["AWAY_ML"].map(american_to_decimal)
    ev_h = p * dec_h - 1.0
    ev_a = (1.0 - p) * dec_a - 1.0
    home_side = ev_h >= ev_a
    d["EV"] = np.where(home_side, ev_h, ev_a)
    d = d[d["EV"] > threshold].copy()
    home_side = home_side.loc[d.index]
    d["BET_SIDE"] = np.where(home_side, "HOME", "AWAY")
    d["BET_ODDS"] = np.where(home_side, d["HOME_ML"], d["AWAY_ML"])
    d["BET_PROB"] = np.where(home_side, d[prob_col], 1.0 - d[prob_col])
    d["WON"] = np.where(home_side, d["HOME_WIN"] == 1, d["HOME_WIN"] == 0)
    d["PROFIT"] = [bet_profit(bool(w), o) for w, o in zip(d["WON"], d["BET_ODDS"])]
    return d


def roi_summary(bets: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Flat 1-unit stakes: bets, win rate, profit, ROI and a bootstrap 95% CI."""
    n = len(bets)
    if n == 0:
        return {"BETS": 0, "WIN_RATE": np.nan, "PROFIT": 0.0, "ROI": np.nan,
                "ROI_LO": np.nan, "ROI_HI": np.nan}
    profit = bets["PROFIT"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = profit[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return {
        "BETS": n,
        "WIN_RATE": float(bets["WON"].mean()),
        "PROFIT": float(profit.sum()),
        "ROI": float(profit.mean()),
        "ROI_LO": float(np.percentile(boots, 2.5)),
        "ROI_HI": float(np.percentile(boots, 97.5)),
    }


def betting_table(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for name in ("MODEL", "BLEND", "MKT_RECAL", "ELO"):
        if PROB_COLS[name] not in df.columns:
            continue
        for t in thresholds:
            s = roi_summary(pick_bets(df, PROB_COLS[name], t))
            rows.append({"PROBS": name, "EV_THRESHOLD": t, **s})
    # Reference: blindly bet every home team / every favorite at the close.
    d = df[df["HOME_ML"].notna()].copy()
    d["ALWAYS_HOME"] = 1.0
    d["FAV"] = (d["MARKET_PROB_HOME"] >= 0.5).astype(float)
    for name, col in (("ALWAYS_HOME", "ALWAYS_HOME"), ("ALWAYS_FAVORITE", "FAV")):
        rows.append({"PROBS": name, "EV_THRESHOLD": np.nan, **roi_summary(pick_bets(d, col, -1.0))})
    table = pd.DataFrame(rows)
    alarm = table[(table["ROI"] > ROI_ALARM) & (table["BETS"] >= 50)]
    for _, r in alarm.iterrows():
        log.warning("ROI %.1f%% for %s at threshold %s: assume a bug or leakage first!",
                    100 * r.ROI, r.PROBS, r.EV_THRESHOLD)
    return table


# --------------------------------------------------------------------------- plots


def plot_reports(preds: pd.DataFrame, metrics: pd.DataFrame, bets: pd.DataFrame, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"MODEL": "#2a6fdb", "BLEND": "#1b9e77", "ELO": "#e07b39", "MARKET": "#3a3a3a"}
    colors = {k: v for k, v in colors.items() if f"{k}_LOGLOSS" in metrics.columns}

    # Log loss by season
    m = metrics[metrics["SEASON"] != "ALL"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, c in colors.items():
        ax.plot(m["SEASON"], m[f"{name}_LOGLOSS"], marker="o", color=c, label=name.title())
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_title("Walk-forward log loss by test season")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "logloss_by_season.png", dpi=150)
    plt.close(fig)

    # Calibration
    d = preds[preds["MARKET_PROB_HOME"].notna()]
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
    for name, c in colors.items():
        col = PROB_COLS[name]
        b = pd.cut(d[col], bins)
        cal = d.groupby(b, observed=True).agg(p=(col, "mean"), y=("HOME_WIN", "mean"), n=(col, "size"))
        cal = cal[cal["n"] >= 30]
        ax.plot(cal["p"], cal["y"], marker="o", color=c, label=name.title())
    ax.set_xlabel("Predicted home win probability")
    ax.set_ylabel("Observed home win rate")
    ax.set_title("Calibration (test seasons)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=150)
    plt.close(fig)

    # Cumulative profit of the model's bets
    if len(bets):
        b = bets.sort_values("GAME_DATE")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(pd.to_datetime(b["GAME_DATE"]), b["PROFIT"].cumsum(), color=colors["MODEL"])
        ax.axhline(0, color="#999999", lw=1)
        ax.set_ylabel("Cumulative profit (units)")
        ax.set_title("Model bets at closing lines (flat 1 unit, EV > 0)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "cumulative_profit.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- main


def add_market_logit(df: pd.DataFrame) -> pd.DataFrame:
    p = df["MARKET_PROB_HOME"].clip(1e-6, 1 - 1e-6)
    return df.assign(MARKET_LOGIT=np.log(p / (1 - p)))


def run(games: pd.DataFrame, first_test: str, warmup: int, thresholds: list[float]):
    feats = add_market_logit(build_features(run_elo(games)))
    preds = None
    for name, features in MODEL_SPECS.items():
        col = PROB_COLS[name]
        out = walk_forward_predict(feats, first_test, warmup, features, col)
        if preds is None:
            preds = out
        else:
            preds = preds.merge(out[["GAME_ID", col]], on="GAME_ID", validate="one_to_one")
    return preds, compare_metrics(preds), betting_table(preds, thresholds)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Walk-forward backtest vs Elo and market")
    p.add_argument("--games", type=Path, default=PROCESSED_DIR / "games.csv")
    p.add_argument("--first-test", default="2010-11")
    p.add_argument("--warmup", type=int, default=1, help="seasons excluded from training")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10])
    p.add_argument("--out", type=Path, default=REPORTS_DIR)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    games = pd.read_csv(args.games, dtype={"GAME_ID": str})
    preds, metrics, betting = run(games, args.first_test, args.warmup, args.thresholds)

    args.out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PROCESSED_DIR / "backtest_predictions.csv", index=False)
    metrics.to_csv(args.out / "metrics_by_season.csv", index=False)
    betting.to_csv(args.out / "betting_summary.csv", index=False)
    plot_reports(preds, metrics, pick_bets(preds, "MODEL_PROB_HOME", 0.0), args.out)

    pd.set_option("display.width", 200)
    print("\nProbability quality (games with a closing line):")
    print(metrics.to_string(index=False, float_format="%.4f"))
    print("\nFlat-stake betting at closing lines:")
    print(betting.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
