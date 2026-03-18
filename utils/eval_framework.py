"""
Evaluation framework CLI for March Madness prediction methods.

Generates submissions for all methods × years × genders, scores them, and
outputs a leaderboard. Skips existing submission CSVs by default.

Data isolation:
- Prediction data: data/{year}/ (strict data fence — no future leakage)
- Scoring data: data/{year+1}/ (next year has tournament results)

Usage (from project root):
    poetry run python utils/eval_framework.py
    poetry run python utils/eval_framework.py --methods seed_spline_only baseline_massey
    poetry run python utils/eval_framework.py --methods all --include-slow
    poetry run python utils/eval_framework.py --report-only
    poetry run python utils/eval_framework.py --years 2024 2025 --top-n 10
"""

import argparse
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.approaches import generate_notebook_submission
from src.evaluation.scorer import (
    generate_leaderboard,
    load_tournament_results,
    score_submission,
)


# ── Method registry ───────────────────────────────────────────────────────────

_METHOD_REGISTRY = {
    # notebook-backed methods (no Predictor needed)
    "seed_spline_only": "notebook",
    "baseline_massey": "notebook",
    "massey_direct": "notebook",
    "massey_blend": "notebook",
    "xgb_ensemble": "notebook",
    "xgb_ensemble_v2": "notebook",
    "recency_xgb": "notebook",
    "poisson_margin": "notebook",
    "seed_matchup_calibration": "notebook",
    "modeh7": "notebook",         # LOSO XGBoost regressor (2025 Kaggle 1st-place)
    "meta_ensemble": "notebook",  # slow — OOF loop
    "upset_aware_ensemble": "notebook",  # xgb_ensemble_v2 + upset features + 50/50 blend (M)
    # predictor-backed methods (require ELO + ML initialization)
    "elo": "predictor",
    "elo_enhanced": "predictor",
}

_DEFAULT_METHODS = [
    "seed_spline_only",
    "baseline_massey",
    "massey_direct",
    "massey_blend",
    "xgb_ensemble",
    "xgb_ensemble_v2",
    "recency_xgb",
    "poisson_margin",
    "seed_matchup_calibration",
    "modeh7",
]

_SLOW_METHODS = {"meta_ensemble", "elo", "elo_enhanced"}

_SCORABLE_YEARS = [2022, 2023, 2024, 2025]


# ── Predictor cache ───────────────────────────────────────────────────────────

class _PredictorCache:
    """Lazily builds and caches MarchMadnessPredictor instances per (data_dir, season, gender)."""

    def __init__(self):
        self._cache: dict = {}

    def get(self, data_dir: str, season: int, gender: str):
        key = (data_dir, season, gender)
        if key not in self._cache:
            self._cache[key] = self._build(data_dir, season, gender)
        return self._cache[key]

    @staticmethod
    def _build(data_dir: str, season: int, gender: str):
        from src.data_classes.processing.DataManager import MarchMadnessDataManager
        from src.data_classes.processing.EloRatingSystem import EloRatingSystem
        from src.data_classes.processing.TeamStatsCalculator import TeamStatsCalculator
        from src.data_classes.processing.MLModel import MarchMadnessMLModel
        from src.data_classes.processing.Predictor import MarchMadnessPredictor

        dm = MarchMadnessDataManager(data_dir, gender=gender, current_season=season)
        dm.load_data()
        elo = EloRatingSystem(dm)
        stats = TeamStatsCalculator(dm)
        ml = MarchMadnessMLModel(dm, elo, stats)
        predictor = MarchMadnessPredictor(
            data_manager=dm,
            elo_system=elo,
            stats_calculator=stats,
            ml_model=ml,
            current_season=season,
        )
        predictor.initialize_models(
            calculate_elo=True,
            calculate_stats=True,
            train_ml=True,
            train_years_range=(max(2003, season - 8), season - 1),
            calibrate_ml=False,
        )
        return predictor


# ── Core generation ───────────────────────────────────────────────────────────

def _submission_path(output_dir: str, season: int, method: str, gender: str) -> str:
    sub_dir = os.path.join(output_dir, str(season), method)
    os.makedirs(sub_dir, exist_ok=True)
    return os.path.join(sub_dir, f"{method}_{gender}.csv")


def _generate_submission_for_method(
    method: str,
    method_type: str,
    data_dir: str,
    season: int,
    gender: str,
    predictor_cache: _PredictorCache,
) -> "pd.DataFrame | None":
    """Generate a submission DataFrame for one method/season/gender.

    Returns None on failure (logs warning).
    """
    try:
        if method_type == "notebook":
            return generate_notebook_submission(method, data_dir, season, gender)
        elif method_type == "predictor":
            predictor = predictor_cache.get(data_dir, season, gender)
            sub = predictor.generate_predictions(method=method)
            if "ID" not in sub.columns or "Pred" not in sub.columns:
                print(f"  [warn] predictor returned unexpected columns for {method}: {sub.columns.tolist()}")
                return None
            return sub
        else:
            print(f"  [warn] Unknown method type '{method_type}' for {method}")
            return None
    except Exception as e:
        print(f"  [error] {method}/{gender}/{season}: {e}")
        traceback.print_exc()
        return None


def _score_one(sub_df: pd.DataFrame, scoring_data_dir: str, season: int, gender: str) -> dict:
    """Score a submission for one season/gender. Returns {} on failure."""
    try:
        results_df = load_tournament_results(scoring_data_dir, season, gender)
        if len(results_df) == 0:
            return {}
        stats = score_submission(sub_df, results_df, season, gender)
        return stats
    except FileNotFoundError:
        return {}


# ── Main eval loop ────────────────────────────────────────────────────────────

def _add_skill_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add skill_score column: fractional improvement over seed_spline_only baseline.

    skill_score = (spline_brier - model_brier) / spline_brier

    Higher is better. NaN when seed_spline_only is absent for a given year/gender.
    """
    if df.empty or "brier_score" not in df.columns:
        return df

    df = df.copy()
    method_col = "method" if "method" in df.columns else "model"
    spline_rows = df[df[method_col] == "seed_spline_only"][["year", "gender", "brier_score"]]
    if spline_rows.empty:
        df["skill_score"] = float("nan")
        return df

    spline_map = spline_rows.rename(columns={"brier_score": "_spline_brier"})
    df = df.merge(spline_map, on=["year", "gender"], how="left")
    df["skill_score"] = (df["_spline_brier"] - df["brier_score"]) / df["_spline_brier"]
    df = df.drop(columns=["_spline_brier"])
    return df


def run_eval(
    methods: list[str],
    years: list[int],
    genders: list[str],
    base_data_dir: str,
    output_dir: str,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Run evaluation: generate submissions and score them.

    Parameters
    ----------
    methods : list[str]
        Method names to evaluate (must be keys in _METHOD_REGISTRY).
    years : list[int]
        Tournament seasons to score.
    genders : list[str]
        List of "M" and/or "W".
    base_data_dir : str
        Root data directory. Prediction data comes from base_data_dir/{year}/.
        Scoring data comes from base_data_dir/{year+1}/.
    output_dir : str
        Root output directory. Submissions saved to output_dir/{year}/{method}/submissions/.
    skip_existing : bool
        If True, reuse existing submission CSVs instead of regenerating.

    Returns
    -------
    pd.DataFrame with columns: year, method, gender, brier_score, accuracy, n_games, skill_score
    """
    predictor_cache = _PredictorCache()
    rows = []

    for gender in genders:
        print(f"\n{'=' * 60}")
        print(f"Gender: {gender}")
        print(f"{'=' * 60}")

        for season in sorted(years):
            prediction_data_dir = os.path.join(base_data_dir, str(season))
            if not os.path.isdir(prediction_data_dir):
                print(f"  [skip] Missing prediction data dir: {prediction_data_dir}")
                continue

            scoring_data_dir = os.path.join(base_data_dir, str(season + 1))
            if not os.path.isdir(scoring_data_dir):
                print(f"  [warn] No scoring data dir for season {season}: {scoring_data_dir}")
                scoring_data_dir = None

            print(f"\n  Season: {season}")

            for method in methods:
                method_type = _METHOD_REGISTRY.get(method)
                if method_type is None:
                    print(f"  [warn] Unknown method '{method}', skipping")
                    continue

                sub_path = _submission_path(output_dir, season, method, gender)

                # Load or generate submission
                if skip_existing and os.path.exists(sub_path):
                    print(f"    [skip] {method}/{gender}/{season} — loading from {sub_path}")
                    try:
                        sub_df = pd.read_csv(sub_path)
                    except Exception as e:
                        print(f"    [error] Could not load {sub_path}: {e}")
                        continue
                else:
                    print(f"    [gen]  {method}/{gender}/{season} ...")
                    sub_df = _generate_submission_for_method(
                        method, method_type, prediction_data_dir, season, gender, predictor_cache
                    )
                    if sub_df is None:
                        continue
                    try:
                        sub_df.to_csv(sub_path, index=False)
                        print(f"    [OK]   saved {len(sub_df)} rows → {sub_path}")
                    except Exception as e:
                        print(f"    [warn] Could not save {sub_path}: {e}")

                # Score
                if scoring_data_dir is None:
                    print(f"    [skip] no scoring data for {season}")
                    continue

                stats = _score_one(sub_df, scoring_data_dir, season, gender)
                if stats.get("n_games", 0) == 0:
                    print(f"    [warn] {method}/{gender}/{season}: 0 scorable games")
                    continue

                rows.append({
                    "year": season,
                    "method": method,
                    "gender": gender,
                    "brier_score": round(stats["brier_score"], 5),
                    "accuracy": round(stats["accuracy"], 5),
                    "n_games": stats["n_games"],
                    "missing_games": stats.get("missing_games", 0),
                })
                print(
                    f"    [score] brier={stats['brier_score']:.4f} "
                    f"acc={stats['accuracy']:.3f} games={stats['n_games']}"
                )

    result = pd.DataFrame(rows)
    return _add_skill_scores(result)


# ── Leaderboard helpers ───────────────────────────────────────────────────────

def build_leaderboard(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build a leaderboard from eval results.

    Wraps generate_leaderboard() using 'method' as the model column.
    """
    if results_df.empty:
        return pd.DataFrame()
    # generate_leaderboard expects a 'model' column
    df = results_df.rename(columns={"method": "model"})
    return generate_leaderboard(df, sort_by="avg_brier")


def print_leaderboard(results_df: pd.DataFrame, top_n: int = 15) -> None:
    """Print a formatted leaderboard table to stdout."""
    if results_df.empty:
        print("No results to display.")
        return

    for gender in ("M", "W"):
        gdf = results_df[results_df["gender"] == gender]
        if gdf.empty:
            continue
        gender_label = "Men's" if gender == "M" else "Women's"
        lb = build_leaderboard(gdf)
        if lb.empty:
            continue

        # Compute avg_skill_score per method if available
        has_skill = "skill_score" in gdf.columns and gdf["skill_score"].notna().any()
        if has_skill:
            method_col = "method" if "method" in gdf.columns else "model"
            skill_avg = (
                gdf.groupby(method_col)["skill_score"]
                .mean()
                .rename("avg_skill")
                .reset_index()
                .rename(columns={method_col: "model"})
            )
            lb = lb.merge(skill_avg, on="model", how="left")

        print(f"\n{'=' * 80}")
        print(f"LEADERBOARD — {gender_label} (top {top_n})")
        print(f"Lower brier = better. skill_score = improvement over seed baseline (higher = better).")
        print(f"{'=' * 80}")
        year_cols = sorted([c for c in lb.columns if c.isdigit()])
        skill_hdr = f"{'AvgSkill':>10}" if has_skill else ""
        header = f"{'Method':<35}{'AvgBrier':>10}{skill_hdr}{'Rank':>8}{'N':>5}" + "".join(f"{y:>8}" for y in year_cols)
        print(header)
        print("-" * 80)
        for _, row in lb.head(top_n).iterrows():
            skill_col = ""
            if has_skill:
                sv = row.get("avg_skill", float("nan"))
                skill_col = f"{sv:>10.4f}" if isinstance(sv, float) and not np.isnan(sv) else f"{'—':>10}"
            line = f"{str(row['model']):<35}{row['avg_brier']:>10.4f}{skill_col}{row.get('avg_rank', 0):>8.1f}{int(row['n_years']):>5}"
            for y in year_cols:
                val = row.get(y, float("nan"))
                if isinstance(val, float) and not np.isnan(val):
                    line += f"{val:>8.4f}"
                else:
                    line += f"{'—':>8}"
            print(line)
        print("=" * 80)


def print_per_year_breakdown(results_df: pd.DataFrame) -> None:
    """Print per-year Brier scores grouped by gender."""
    if results_df.empty:
        print("No results to display.")
        return

    for gender in ("M", "W"):
        gdf = results_df[results_df["gender"] == gender]
        if gdf.empty:
            continue
        gender_label = "Men's" if gender == "M" else "Women's"
        print(f"\n{'=' * 72}")
        print(f"PER-YEAR BREAKDOWN — {gender_label}")
        print(f"{'=' * 72}")
        for year, year_grp in gdf.groupby("year"):
            print(f"\n  Season {year}:")
            year_grp = year_grp.sort_values("brier_score")
            for _, row in year_grp.iterrows():
                method = row.get("method", row.get("model", "?"))
                print(
                    f"    {method:<35} brier={row['brier_score']:.4f} "
                    f"acc={row['accuracy']:.3f} games={row['n_games']}"
                )


# ── Output persistence ────────────────────────────────────────────────────────

_EVAL_RESULTS_FILE = "eval_results.csv"


def _load_existing_results(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, _EVAL_RESULTS_FILE)
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[warn] Could not load existing eval_results.csv: {e}")
    return pd.DataFrame()


def _save_results(results_df: pd.DataFrame, output_dir: str, append: bool = True) -> str:
    """Save results to output_dir/eval_results.csv. Appends by default (deduplicates)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, _EVAL_RESULTS_FILE)
    if append and os.path.exists(path):
        existing = _load_existing_results(output_dir)
        combined = pd.concat([existing, results_df], ignore_index=True)
        key_cols = [c for c in ["year", "method", "gender"] if c in combined.columns]
        if key_cols:
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
        combined.to_csv(path, index=False)
        return path
    else:
        results_df.to_csv(path, index=False)
        return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate March Madness prediction methods with Brier score leaderboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help=(
            "Methods to evaluate. Use 'all' for all methods. "
            f"Default: {_DEFAULT_METHODS}"
        ),
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=_SCORABLE_YEARS,
        help="Tournament seasons to score.",
    )
    parser.add_argument(
        "--genders", nargs="+", choices=["M", "W"], default=["M", "W"],
        help="Genders to evaluate.",
    )
    parser.add_argument(
        "--base-data-dir",
        default=os.path.join(PROJECT_ROOT, "data"),
        help="Root data directory. Expects data/{year}/ subdirs.",
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
        "--include-slow", action="store_true",
        help="Also run slow methods: meta_ensemble, elo, elo_enhanced.",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Load existing eval_results.csv and display report without regenerating.",
    )
    parser.add_argument(
        "--top-n", type=int, default=15,
        help="Number of rows to show in leaderboard.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    output_dir = args.output_dir

    # Report-only mode
    if args.report_only:
        results_df = _load_existing_results(output_dir)
        if results_df.empty:
            print(f"No eval_results.csv found in {output_dir}. Run without --report-only first.")
            sys.exit(1)
        if "skill_score" not in results_df.columns:
            results_df = _add_skill_scores(results_df)
        print_leaderboard(results_df, top_n=args.top_n)
        print_per_year_breakdown(results_df)
        return

    # Resolve methods
    if args.methods is None:
        methods = list(_DEFAULT_METHODS)
    elif len(args.methods) == 1 and args.methods[0] == "all":
        methods = list(_METHOD_REGISTRY.keys())
    else:
        methods = args.methods
        unknown = [m for m in methods if m not in _METHOD_REGISTRY]
        if unknown:
            print(f"[error] Unknown methods: {unknown}")
            print(f"Known methods: {sorted(_METHOD_REGISTRY.keys())}")
            sys.exit(1)

    if not args.include_slow:
        slow_found = [m for m in methods if m in _SLOW_METHODS]
        if slow_found:
            print(f"[info] Skipping slow methods (use --include-slow to include): {slow_found}")
        methods = [m for m in methods if m not in _SLOW_METHODS]

    print(f"Methods: {methods}")
    print(f"Years: {args.years}")
    print(f"Genders: {args.genders}")
    print(f"Base data dir: {args.base_data_dir}")
    print(f"Output dir: {output_dir}")

    results_df = run_eval(
        methods=methods,
        years=args.years,
        genders=args.genders,
        base_data_dir=args.base_data_dir,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing,
    )

    if results_df.empty:
        print("\nNo scoring results generated.")
        return

    # Save results
    out_path = _save_results(results_df, output_dir)
    print(f"\nSaved eval results → {out_path} ({len(results_df)} rows)")

    # Print leaderboard
    print_leaderboard(results_df, top_n=args.top_n)
    print_per_year_breakdown(results_df)


if __name__ == "__main__":
    main()
