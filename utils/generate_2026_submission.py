"""
Generate 2026 March Madness submission using top-performing methods from historical eval.

Reads eval_results.csv to identify best methods, generates predictions for 2026,
and produces a combined M+W submission CSV.

Usage (from project root):
    poetry run python utils/generate_2026_submission.py
    poetry run python utils/generate_2026_submission.py --strategy average_top3
    poetry run python utils/generate_2026_submission.py --top-n 3 --years-for-ranking 2024 2025
    poetry run python utils/generate_2026_submission.py --eval-results output/eval_results.csv
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

from src.approaches import generate_notebook_submission


# ── helpers ───────────────────────────────────────────────────────────────────

def load_top_methods(
    eval_results_path: str,
    gender: str,
    top_n: int = 3,
    years: list[int] | None = None,
) -> list[str]:
    """Read eval_results.csv and return top_n method names sorted best→worst for given gender.

    Parameters
    ----------
    eval_results_path : str
        Path to eval_results.csv (output of utils/eval_framework.py).
    gender : str
        "M" or "W".
    top_n : int
        Number of top methods to return.
    years : list[int], optional
        Filter to these years only. Default: use all available years.

    Returns
    -------
    list[str] : Method names sorted from best (lowest avg Brier) to worst.
    """
    if not os.path.exists(eval_results_path):
        raise FileNotFoundError(f"eval_results.csv not found: {eval_results_path}")

    df = pd.read_csv(eval_results_path)
    # Support both 'method' and 'model' column names
    method_col = "method" if "method" in df.columns else "model"
    if method_col not in df.columns:
        raise ValueError(f"eval_results.csv has no 'method' or 'model' column. Columns: {df.columns.tolist()}")

    gdf = df[df["gender"] == gender].copy()
    if gdf.empty:
        raise ValueError(f"No results for gender='{gender}' in {eval_results_path}")

    if years is not None:
        gdf = gdf[gdf["year"].isin(years)]
        if gdf.empty:
            raise ValueError(f"No results for gender='{gender}', years={years}")

    avg_brier = gdf.groupby(method_col)["brier_score"].mean().sort_values()
    top_methods = avg_brier.head(top_n).index.tolist()
    print(f"Top {top_n} methods for {gender} (avg Brier over {sorted(gdf['year'].unique().tolist())} seasons):")
    for i, m in enumerate(top_methods, 1):
        print(f"  {i}. {m:35s} avg_brier={avg_brier[m]:.5f}")
    return top_methods


def generate_2026_predictions(
    methods: list[str],
    gender: str,
    data_dir: str,
    output_dir: str,
    skip_existing: bool = True,
    season: int = 2026,
) -> dict[str, pd.DataFrame]:
    """Generate 2026 predictions for each method.

    Parameters
    ----------
    methods : list[str]
        Method names to generate (must be supported by generate_notebook_submission).
    gender : str
        "M" or "W".
    data_dir : str
        Data directory for 2026 (e.g. "data/2026").
    output_dir : str
        Root output directory. CSVs saved to output_dir/{season}/submissions/.
    skip_existing : bool
        If True, load existing CSVs instead of regenerating.
    season : int
        Target season year (default: 2026).

    Returns
    -------
    dict[str, pd.DataFrame] : method → submission DataFrame.
    """
    sub_dir = os.path.join(output_dir, str(season), "submissions")
    os.makedirs(sub_dir, exist_ok=True)

    results = {}
    for method in methods:
        sub_path = os.path.join(sub_dir, f"{method}_{gender}.csv")
        if skip_existing and os.path.exists(sub_path):
            print(f"  [skip] {method}/{gender}/{season} — loading from {sub_path}")
            try:
                results[method] = pd.read_csv(sub_path)
                continue
            except Exception as e:
                print(f"  [warn] Could not load {sub_path}: {e}")

        print(f"  [gen]  {method}/{gender}/{season} ...")
        try:
            sub_df = generate_notebook_submission(method, data_dir, season, gender)
            sub_df.to_csv(sub_path, index=False)
            print(f"  [OK]   saved {len(sub_df)} rows → {sub_path}")
            results[method] = sub_df
        except Exception as e:
            print(f"  [error] {method}/{gender}/{season}: {e}")
            import traceback
            traceback.print_exc()

    return results


def combine_submissions(
    method_predictions: dict[str, pd.DataFrame],
    strategy: str = "top1",
) -> pd.DataFrame:
    """Combine method predictions into a single submission.

    Parameters
    ----------
    method_predictions : dict[str, pd.DataFrame]
        method → DataFrame with [ID, Pred] columns.
        Methods should be ordered best→worst (first is best).
    strategy : str
        "top1" — use predictions from the best method only.
        "average_top3" — mean of up to top-3 method predictions.

    Returns
    -------
    pd.DataFrame with [ID, Pred] columns.
    """
    if not method_predictions:
        raise ValueError("No method predictions provided.")

    ordered_methods = list(method_predictions.keys())

    if strategy == "top1":
        best_method = ordered_methods[0]
        print(f"  Strategy=top1: using {best_method}")
        return method_predictions[best_method][["ID", "Pred"]].copy()

    elif strategy == "average_top3":
        top_methods = ordered_methods[:3]
        print(f"  Strategy=average_top3: averaging {top_methods}")
        frames = [method_predictions[m][["ID", "Pred"]].rename(columns={"Pred": f"Pred_{m}"}) for m in top_methods]
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="ID", how="outer")
        pred_cols = [f"Pred_{m}" for m in top_methods if f"Pred_{m}" in merged.columns]
        merged["Pred"] = merged[pred_cols].mean(axis=1)
        merged["Pred"] = merged["Pred"].clip(0.025, 0.975)
        return merged[["ID", "Pred"]]

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'top1' or 'average_top3'.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate 2026 March Madness submission using top historical methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval-results",
        default=os.path.join(PROJECT_ROOT, "output", "eval_results.csv"),
        help="Path to eval_results.csv (output of utils/eval_framework.py).",
    )
    parser.add_argument(
        "--top-n", type=int, default=3,
        help="Number of top methods to consider per gender.",
    )
    parser.add_argument(
        "--strategy", choices=["top1", "average_top3"], default="top1",
        help="Combination strategy.",
    )
    parser.add_argument(
        "--years-for-ranking", nargs="+", type=int, default=None,
        help="Which years to use for ranking methods. Default: all available.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "data", "2026"),
        help="Data directory for season 2026.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "output"),
        help="Root output directory.",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-generate submissions even if CSV already exists.",
    )
    parser.add_argument(
        "--season", type=int, default=2026,
        help="Target season (default: 2026).",
    )
    args = parser.parse_args()

    season = args.season
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"[error] Data directory not found: {data_dir}")
        sys.exit(1)

    all_submissions = []
    for gender in ("M", "W"):
        print(f"\n{'=' * 60}")
        print(f"Gender: {gender}")
        print(f"{'=' * 60}")

        # Identify top methods
        try:
            top_methods = load_top_methods(
                args.eval_results,
                gender=gender,
                top_n=args.top_n,
                years=args.years_for_ranking,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"[warn] Could not load top methods for {gender}: {e}")
            print("  Falling back to: xgb_ensemble_v2, baseline_massey, massey_direct")
            top_methods = ["xgb_ensemble_v2", "baseline_massey", "massey_direct"]

        # Generate predictions
        print(f"\nGenerating {gender} predictions for season {season}...")
        preds = generate_2026_predictions(
            methods=top_methods,
            gender=gender,
            data_dir=data_dir,
            output_dir=args.output_dir,
            skip_existing=not args.no_skip_existing,
            season=season,
        )

        if not preds:
            print(f"[error] No predictions generated for {gender}. Skipping.")
            continue

        # Combine
        print(f"\nCombining {gender} predictions (strategy={args.strategy})...")
        combined = combine_submissions(preds, strategy=args.strategy)
        all_submissions.append(combined)
        print(f"  {len(combined)} matchup rows for {gender}")

    if not all_submissions:
        print("[error] No predictions generated for any gender.")
        sys.exit(1)

    # Merge M + W into one file
    final_sub = pd.concat(all_submissions, ignore_index=True)
    final_sub = final_sub.drop_duplicates(subset=["ID"])
    final_sub["Pred"] = final_sub["Pred"].clip(0.025, 0.975)
    final_sub = final_sub[["ID", "Pred"]].sort_values("ID").reset_index(drop=True)

    out_path = os.path.join(args.output_dir, str(season), f"submission_{season}_{args.strategy}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_sub.to_csv(out_path, index=False)
    print(f"\n[OK] Combined M+W submission ({len(final_sub)} rows) → {out_path}")
    print(f"     Sample:")
    print(final_sub.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
