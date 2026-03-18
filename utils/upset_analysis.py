"""
Upset analysis for March Madness tournaments.

Quantifies where and why models miss upsets by analyzing:
- Upset rates by round and seed matchup
- Regular season features correlated with upset outcomes
- Logistic regression on feature diffs to identify most predictive signals

Usage:
    poetry run python utils/upset_analysis.py --data-dir data/2026
    poetry run python utils/upset_analysis.py --data-dir data/2026 --output output/upset_analysis.csv
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.evaluation.scorer import ROUND_DAY_MAP


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_csv(data_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def _seed_number(seed_val: str) -> int:
    return int(str(seed_val)[1:3])


def _day_to_round(day_num: int) -> str:
    day_map = ROUND_DAY_MAP["M"]
    for round_name, (low, high) in day_map.items():
        if low <= day_num <= high:
            return round_name
    return "Unknown"


# ── Feature computation ───────────────────────────────────────────────────────

def _compute_four_factors(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Four Factors + efficiency stats per (Season, TeamID) from detailed results."""
    records = {}

    def _add(season, team_id, fgm, fga, fgm3, fga3, ftm, fta, orb, drb, ast, tov, stl, pts, opp_pts, possessions):
        key = (season, team_id)
        if key not in records:
            records[key] = {
                "Season": season, "TeamID": team_id,
                "FGM": 0, "FGA": 0, "FGM3": 0, "FGA3": 0,
                "FTM": 0, "FTA": 0, "ORB": 0, "DRB": 0,
                "AST": 0, "TOV": 0, "STL": 0,
                "Pts": 0, "OppPts": 0, "Poss": 0, "Games": 0,
            }
        r = records[key]
        r["FGM"] += fgm; r["FGA"] += fga
        r["FGM3"] += fgm3; r["FGA3"] += fga3
        r["FTM"] += ftm; r["FTA"] += fta
        r["ORB"] += orb; r["DRB"] += drb
        r["AST"] += ast; r["TOV"] += tov; r["STL"] += stl
        r["Pts"] += pts; r["OppPts"] += opp_pts
        r["Poss"] += possessions
        r["Games"] += 1

    for _, row in detailed_df.iterrows():
        season = int(row["Season"])
        w_poss = row["WFGA"] - row["WOR"] + row["WTO"] + 0.44 * row["WFTA"]
        l_poss = row["LFGA"] - row["LOR"] + row["LTO"] + 0.44 * row["LFTA"]
        poss = (w_poss + l_poss) / 2.0

        _add(season, int(row["WTeamID"]),
             row["WFGM"], row["WFGA"], row["WFGM3"], row["WFGA3"],
             row["WFTM"], row["WFTA"], row["WOR"], row["WDR"],
             row["WAst"], row["WTO"], row["WStl"],
             row["WScore"], row["LScore"], poss)

        _add(season, int(row["LTeamID"]),
             row["LFGM"], row["LFGA"], row["LFGM3"], row["LFGA3"],
             row["LFTM"], row["LFTA"], row["LOR"], row["LDR"],
             row["LAst"], row["LTO"], row["LStl"],
             row["LScore"], row["WScore"], poss)

    df = pd.DataFrame(list(records.values()))
    df["eFG_pct"] = (df["FGM"] + 0.5 * df["FGM3"]) / df["FGA"].clip(lower=1)
    df["TOV_pct"] = df["TOV"] / (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]).clip(lower=1)
    df["ORB_pct"] = df["ORB"] / (df["ORB"] + df["DRB"]).clip(lower=1)
    df["FTRate"] = df["FTA"] / df["FGA"].clip(lower=1)
    df["ThreePtReliance"] = df["FGA3"] / df["FGA"].clip(lower=1)
    df["OffEff"] = df["Pts"] / df["Poss"].clip(lower=1) * 100
    df["DefEff"] = df["OppPts"] / df["Poss"].clip(lower=1) * 100
    df["NetEff"] = df["OffEff"] - df["DefEff"]
    df["DefFirstRatio"] = df["DefEff"] / df["OffEff"].clip(lower=1)
    df["Pace"] = df["Poss"] / df["Games"].clip(lower=1)

    return df[["Season", "TeamID", "eFG_pct", "TOV_pct", "ORB_pct", "FTRate",
               "ThreePtReliance", "OffEff", "DefEff", "NetEff", "DefFirstRatio", "Pace"]]


def _compute_momentum(compact_df: pd.DataFrame, cutoff_day: int = 132) -> pd.DataFrame:
    """last_10_win_pct - full_season_win_pct per (Season, TeamID)."""
    records = {}

    for _, row in compact_df.iterrows():
        season, day = int(row["Season"]), int(row["DayNum"])
        for team_id, won in [(int(row["WTeamID"]), True), (int(row["LTeamID"]), False)]:
            key = (season, team_id)
            if key not in records:
                records[key] = {"Season": season, "TeamID": team_id, "games": []}
            records[key]["games"].append((day, won))

    rows = []
    for key, data in records.items():
        games = sorted(data["games"], key=lambda x: x[0])
        total = len(games)
        full_win_pct = sum(1 for _, w in games if w) / max(total, 1)
        last10 = games[-10:]
        last10_win_pct = sum(1 for _, w in last10 if w) / max(len(last10), 1)
        rows.append({
            "Season": data["Season"],
            "TeamID": data["TeamID"],
            "RecentMomentum": last10_win_pct - full_win_pct,
            "FullSeasonWinPct": full_win_pct,
        })

    return pd.DataFrame(rows)


def _compute_massey_seed_gap(massey_df: pd.DataFrame, seeds_df: pd.DataFrame) -> pd.DataFrame:
    """MasseyImpliedSeed vs ActualSeed — positive means underseeded (upset threat)."""
    late = massey_df[massey_df["RankingDayNum"] <= 128].copy()
    avg_rank = (
        late.groupby(["Season", "TeamID"])["OrdinalRank"]
        .mean()
        .reset_index(name="MasseyRankAvg")
    )
    avg_rank["MasseyRankNorm"] = avg_rank.groupby("Season")["MasseyRankAvg"].transform(
        lambda x: 1 - (x - x.min()) / (x.max() - x.min() + 1e-9)
    )
    avg_rank["MasseyImpliedSeed"] = (1 - avg_rank["MasseyRankNorm"]) * 15 + 1

    result = avg_rank.merge(seeds_df[["Season", "TeamID", "Seed"]], on=["Season", "TeamID"], how="inner")
    result["ActualSeed"] = result["Seed"].apply(_seed_number)
    result["MasseyVsSeedGap"] = result["ActualSeed"] - result["MasseyImpliedSeed"]
    return result[["Season", "TeamID", "ActualSeed", "MasseyImpliedSeed", "MasseyVsSeedGap"]]


# ── Upset dataset builder ────────────────────────────────────────────────────

def build_upset_dataset(
    tourney_results: pd.DataFrame,
    seeds_df: pd.DataFrame,
    four_factors: pd.DataFrame,
    momentum: pd.DataFrame,
    massey_gap: pd.DataFrame | None,
    min_season: int = 2010,
) -> pd.DataFrame:
    """Join tournament results with team features to create upset analysis dataset."""
    # Filter to main bracket (exclude play-in)
    main = tourney_results[
        (tourney_results["Season"] >= min_season) &
        (tourney_results["DayNum"] > 135)
    ].copy()

    main["Round"] = main["DayNum"].apply(_day_to_round)

    # Merge seeds
    seed_map = seeds_df.copy()
    seed_map["SeedNum"] = seed_map["Seed"].apply(_seed_number)

    rows = []
    for _, game in main.iterrows():
        season = int(game["Season"])
        winner = int(game["WTeamID"])
        loser = int(game["LTeamID"])

        w_seed_row = seed_map[(seed_map["Season"] == season) & (seed_map["TeamID"] == winner)]
        l_seed_row = seed_map[(seed_map["Season"] == season) & (seed_map["TeamID"] == loser)]
        if len(w_seed_row) == 0 or len(l_seed_row) == 0:
            continue

        w_seed = int(w_seed_row.iloc[0]["SeedNum"])
        l_seed = int(l_seed_row.iloc[0]["SeedNum"])
        is_upset = int(w_seed > l_seed)
        seed_diff_abs = abs(w_seed - l_seed)

        # Higher seed = lower seed number (worse team). For diff, use loser seed - winner seed
        # Positive = upset (winner was a higher seed number = weaker team)
        row = {
            "Season": season,
            "Round": game["Round"],
            "DayNum": int(game["DayNum"]),
            "WinnerID": winner,
            "LoserID": loser,
            "WinnerSeed": w_seed,
            "LoserSeed": l_seed,
            "SeedDiffAbs": seed_diff_abs,
            "is_upset": is_upset,
            "SeedMatchup": f"{min(w_seed, l_seed)}v{max(w_seed, l_seed)}",
        }

        # Feature diffs: (winner stat - loser stat) — positive = winner was better
        for feat_df_source in [four_factors, momentum]:
            for team_id, prefix in [(winner, "W"), (loser, "L")]:
                feat_row = feat_df_source[
                    (feat_df_source["Season"] == season) & (feat_df_source["TeamID"] == team_id)
                ]
                if len(feat_row) == 0:
                    continue
                feat_row = feat_row.iloc[0]
                for col in [c for c in feat_df_source.columns if c not in ["Season", "TeamID"]]:
                    row[f"{prefix}_{col}"] = feat_row.get(col, np.nan)

        if massey_gap is not None:
            for team_id, prefix in [(winner, "W"), (loser, "L")]:
                mgap_row = massey_gap[
                    (massey_gap["Season"] == season) & (massey_gap["TeamID"] == team_id)
                ]
                if len(mgap_row) > 0:
                    row[f"{prefix}_MasseyVsSeedGap"] = mgap_row.iloc[0]["MasseyVsSeedGap"]
                    row[f"{prefix}_MasseyImpliedSeed"] = mgap_row.iloc[0]["MasseyImpliedSeed"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute diffs (winner - loser) for numeric feature columns
    feat_cols = [c for c in df.columns if c.startswith("W_") or c.startswith("L_")]
    w_cols = [c for c in feat_cols if c.startswith("W_")]
    for wc in w_cols:
        base = wc[2:]
        lc = f"L_{base}"
        if lc in df.columns:
            df[f"Diff_{base}"] = df[wc] - df[lc]

    return df


# ── Analysis functions ────────────────────────────────────────────────────────

def analyze_upset_rates_by_round(df: pd.DataFrame) -> pd.DataFrame:
    """Upset rate and count by tournament round."""
    summary = (
        df.groupby("Round")
        .agg(
            n_games=("is_upset", "count"),
            n_upsets=("is_upset", "sum"),
            upset_rate=("is_upset", "mean"),
        )
        .reset_index()
    )
    round_order = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    summary["Round"] = pd.Categorical(summary["Round"], categories=round_order, ordered=True)
    return summary.sort_values("Round").reset_index(drop=True)


def analyze_upset_rates_by_matchup(df: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """Upset rate by (lower_seed, higher_seed) pair."""
    summary = (
        df.groupby("SeedMatchup")
        .agg(
            n_games=("is_upset", "count"),
            n_upsets=("is_upset", "sum"),
            upset_rate=("is_upset", "mean"),
        )
        .reset_index()
    )
    return summary[summary["n_games"] >= min_n].sort_values("upset_rate", ascending=False)


def analyze_feature_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Point-biserial correlations between feature diffs and is_upset."""
    diff_cols = [c for c in df.columns if c.startswith("Diff_")]
    rows = []
    for col in diff_cols:
        valid = df[[col, "is_upset"]].dropna()
        if len(valid) < 20:
            continue
        corr, pval = pointbiserialr(valid["is_upset"], valid[col])
        rows.append({
            "feature": col,
            "correlation": round(corr, 4),
            "p_value": round(pval, 4),
            "abs_corr": abs(corr),
        })
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)


def run_logistic_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Logistic regression on feature diffs to predict is_upset."""
    diff_cols = [c for c in df.columns if c.startswith("Diff_")]
    valid_cols = []
    for col in diff_cols:
        if df[col].notna().sum() >= 50:
            valid_cols.append(col)

    if not valid_cols:
        print("  [warn] No feature diff columns with sufficient data for LR.")
        return pd.DataFrame()

    feat_df = df[valid_cols + ["is_upset"]].dropna()
    if len(feat_df) < 50:
        print(f"  [warn] Too few complete rows ({len(feat_df)}) for LR.")
        return pd.DataFrame()

    X = feat_df[valid_cols].values
    y = feat_df["is_upset"].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X, y)

    coefs = model.named_steps["clf"].coef_[0]
    odds_ratios = np.exp(coefs)

    result = pd.DataFrame({
        "feature": valid_cols,
        "coefficient": coefs,
        "odds_ratio": odds_ratios,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Upset analysis for March Madness tournaments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/2026", help="Data directory (cumulative).")
    parser.add_argument("--output", default="output/upset_analysis.csv", help="Output CSV path.")
    parser.add_argument("--min-season", type=int, default=2010, help="First season to analyze.")
    args = parser.parse_args()

    data_dir = args.data_dir
    print(f"Loading data from {data_dir}...")

    # Load required files
    tourney = _load_csv(data_dir, "MNCAATourneyCompactResults.csv")
    seeds = _load_csv(data_dir, "MNCAATourneySeeds.csv")

    try:
        detailed = _load_csv(data_dir, "MRegularSeasonDetailedResults.csv")
        print(f"  Loaded detailed results: {len(detailed)} rows")
    except FileNotFoundError:
        detailed = None
        print("  [warn] Detailed results not found — skipping efficiency stats.")

    compact = _load_csv(data_dir, "MRegularSeasonCompactResults.csv")

    try:
        massey = _load_csv(data_dir, "MMasseyOrdinals.csv")
        print(f"  Loaded Massey ordinals: {len(massey)} rows")
    except FileNotFoundError:
        massey = None
        print("  [warn] Massey ordinals not found — skipping Massey gap analysis.")

    print("\nComputing team features...")

    four_factors = _compute_four_factors(detailed) if detailed is not None else pd.DataFrame()
    print(f"  Four factors: {len(four_factors)} team-season rows")

    momentum = _compute_momentum(compact)
    print(f"  Momentum: {len(momentum)} team-season rows")

    massey_gap = _compute_massey_seed_gap(massey, seeds) if massey is not None else None
    if massey_gap is not None:
        print(f"  Massey-seed gap: {len(massey_gap)} team-season rows")

    print("\nBuilding upset dataset...")
    upset_df = build_upset_dataset(
        tourney, seeds,
        four_factors if not four_factors.empty else pd.DataFrame(columns=["Season", "TeamID"]),
        momentum,
        massey_gap,
        min_season=args.min_season,
    )
    print(f"  Dataset: {len(upset_df)} games, {upset_df['is_upset'].sum()} upsets "
          f"({upset_df['is_upset'].mean():.1%} upset rate)")

    # ── Analysis 1: Upset rates by round ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("UPSET RATES BY ROUND")
    print("=" * 70)
    round_table = analyze_upset_rates_by_round(upset_df)
    print(round_table.to_string(index=False))

    # ── Analysis 2: Upset rates by seed matchup ───────────────────────────────
    print("\n" + "=" * 70)
    print("UPSET RATES BY SEED MATCHUP (min 10 games)")
    print("=" * 70)
    matchup_table = analyze_upset_rates_by_matchup(upset_df)
    print(matchup_table.head(20).to_string(index=False))

    # ── Analysis 3: Feature correlations ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("FEATURE CORRELATIONS WITH UPSET (point-biserial)")
    print("=" * 70)
    corr_table = analyze_feature_correlations(upset_df)
    if not corr_table.empty:
        print(corr_table.head(20).to_string(index=False))
    else:
        print("  No sufficient feature data for correlation analysis.")

    # ── Analysis 4: Logistic regression ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION ON UPSET PREDICTORS (class_weight='balanced')")
    print("=" * 70)
    lr_table = run_logistic_regression(upset_df)
    if not lr_table.empty:
        print(lr_table.head(20).to_string(index=False))
        print("\nInterpretation: positive coefficient = feature diff (winner - loser) predicts upset.")
        print("  High abs odds_ratio features are strongest upset predictors.")
    else:
        print("  Could not run logistic regression.")

    # ── Save output ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    upset_df.to_csv(args.output, index=False)
    print(f"\nSaved upset dataset → {args.output} ({len(upset_df)} rows)")

    # Also save analysis tables
    out_dir = os.path.dirname(args.output) or "."
    round_table.to_csv(os.path.join(out_dir, "upset_by_round.csv"), index=False)
    matchup_table.to_csv(os.path.join(out_dir, "upset_by_matchup.csv"), index=False)
    if not corr_table.empty:
        corr_table.to_csv(os.path.join(out_dir, "upset_correlations.csv"), index=False)
    if not lr_table.empty:
        lr_table.to_csv(os.path.join(out_dir, "upset_lr_coefficients.csv"), index=False)

    print("Done.")


if __name__ == "__main__":
    main()
