"""
Combine the best N methods into a Kaggle-ready CSV submission.

Reads output/eval_results.csv to pick the top-performing methods by average
Brier score, loads per-method per-gender CSVs, blends them, and writes a
single combined M+W submission file.

Usage (from project root):
    poetry run python utils/kaggle_submission.py --season 2026
    poetry run python utils/kaggle_submission.py --season 2026 --strategy weighted --top-n 3
    poetry run python utils/kaggle_submission.py --season 2026 --years-for-ranking 2024 2025
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


DEFAULT_EVAL_RESULTS = os.path.join(PROJECT_ROOT, "output", "eval_results.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEFAULT_YEARS_FOR_RANKING = [2024, 2025]


def _pick_top_methods(eval_df: pd.DataFrame, gender: str, years: list, top_n: int) -> list:
    """Return top_n method names sorted ascending by average Brier score for given gender+years."""
    subset = eval_df[(eval_df["Gender"] == gender) & (eval_df["Year"].isin(years))]
    if subset.empty:
        return []
    avg = subset.groupby("Method")["BrierScore"].mean().sort_values()
    return avg.head(top_n).index.tolist()


def _load_submission(output_dir: str, season: int, method: str, gender: str) -> pd.DataFrame:
    """Load output/{season}/{method}/submissions/{method}_{gender}.csv, return rows for this season."""
    path = os.path.join(output_dir, str(season), method, "submissions", f"{method}_{gender}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Submission not found: {path}")
    df = pd.read_csv(path)
    season_prefix = f"{season}_"
    df = df[df["ID"].str.startswith(season_prefix)].copy()
    if df.empty:
        raise ValueError(f"No rows for season {season} in {path}")
    return df[["ID", "Pred"]].copy()


def _blend(sub_dfs: list, weights: list) -> pd.DataFrame:
    """Weighted average of submission DataFrames. Missing IDs filled with 0.5."""
    merged = sub_dfs[0].rename(columns={"Pred": "Pred_0"})
    for i, df in enumerate(sub_dfs[1:], start=1):
        merged = merged.merge(df.rename(columns={"Pred": f"Pred_{i}"}), on="ID", how="outer")

    pred_cols = [f"Pred_{i}" for i in range(len(sub_dfs))]
    merged[pred_cols] = merged[pred_cols].fillna(0.5)

    weights_arr = np.array(weights, dtype=float)
    weights_arr /= weights_arr.sum()

    merged["Pred"] = sum(merged[f"Pred_{i}"] * w for i, w in enumerate(weights_arr))
    merged["Pred"] = merged["Pred"].clip(0.025, 0.975)
    return merged[["ID", "Pred"]]


def main():
    parser = argparse.ArgumentParser(
        description="Combine top methods into a Kaggle-ready submission CSV."
    )
    parser.add_argument("--season", type=int, required=True, help="Target season (e.g. 2026)")
    parser.add_argument(
        "--top-n", type=int, default=2,
        help="Number of top methods to blend (default: 2)",
    )
    parser.add_argument(
        "--strategy", choices=["equal", "weighted"], default="equal",
        help="Blend strategy: 'equal' (50/50) or 'weighted' (inverse-Brier, default: equal)",
    )
    parser.add_argument(
        "--years-for-ranking", nargs="+", type=int, default=DEFAULT_YEARS_FOR_RANKING,
        help=f"Years to average Brier score over (default: {DEFAULT_YEARS_FOR_RANKING})",
    )
    parser.add_argument(
        "--eval-results", default=DEFAULT_EVAL_RESULTS,
        help=f"Path to eval_results.csv (default: {DEFAULT_EVAL_RESULTS})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.eval_results):
        print(f"[error] eval_results.csv not found at {args.eval_results}")
        print("  Run: poetry run python utils/eval_framework.py  to generate it first.")
        sys.exit(1)

    eval_df = pd.read_csv(args.eval_results)
    required_cols = {"Method", "Gender", "Year", "BrierScore"}
    if not required_cols.issubset(eval_df.columns):
        print(f"[error] eval_results.csv is missing columns. Expected: {required_cols}")
        print(f"  Found: {set(eval_df.columns)}")
        sys.exit(1)

    all_parts = []

    for gender in ("M", "W"):
        top_methods = _pick_top_methods(eval_df, gender, args.years_for_ranking, args.top_n)
        if not top_methods:
            print(f"[warn] No eval results found for gender={gender} years={args.years_for_ranking}; skipping.")
            continue

        print(f"\nGender {gender} — top {len(top_methods)} methods: {top_methods}")

        sub_dfs = []
        loaded_methods = []
        for method in top_methods:
            try:
                df = _load_submission(args.output_dir, args.season, method, gender)
                sub_dfs.append(df)
                loaded_methods.append(method)
                print(f"  [ok] loaded {method}_{gender}.csv ({len(df)} rows)")
            except Exception as e:
                print(f"  [warn] could not load {method}_{gender}: {e}")

        if not sub_dfs:
            print(f"  [error] No submissions loaded for {gender}; skipping.")
            continue

        if len(sub_dfs) == 1:
            blended = sub_dfs[0]
        else:
            if args.strategy == "weighted":
                # Weights = inverse of average Brier score (lower Brier = higher weight)
                scores = []
                for method in loaded_methods:
                    subset = eval_df[
                        (eval_df["Method"] == method)
                        & (eval_df["Gender"] == gender)
                        & (eval_df["Year"].isin(args.years_for_ranking))
                    ]
                    scores.append(subset["BrierScore"].mean() if not subset.empty else 0.25)
                # Avoid division by zero; add small epsilon
                weights = [1.0 / (s + 1e-9) for s in scores]
            else:
                weights = [1.0] * len(sub_dfs)

            blended = _blend(sub_dfs, weights)

        all_parts.append(blended)

    if not all_parts:
        print("\n[error] No submissions could be built.")
        sys.exit(1)

    combined = pd.concat(all_parts, ignore_index=True)

    out_dir = os.path.join(args.output_dir, str(args.season))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"kaggle_submission_{args.season}.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n[ok] Saved Kaggle submission ({len(combined)} rows) → {out_path}")


if __name__ == "__main__":
    main()
