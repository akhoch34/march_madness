"""
Generate historical March Madness artifacts: submission CSVs, bracket PNGs, and a
consolidated scoring results table.

Strict data-leakage prevention:
- Predictions for season Y are built strictly from data/Y.
- Scoring for season Y is loaded from data/(Y+1) when available.
- ML training uses only completed tournaments prior to season Y.

Usage (from project root):
    poetry run python utils/generate_historical_artifacts.py
    poetry run python utils/generate_historical_artifacts.py --years 2024 2025
    poetry run python utils/generate_historical_artifacts.py \\
        --years 2022 2023 2024 2025 \\
        --genders M W \\
        --methods elo elo_enhanced \\
        --base-data-dir data

Outputs:
    output/{year}/{method}/{method}_{gender}.csv      - skipped if already exists
    output/{year}/{method}/bracket_{gender}.png
    output/{year}/{method}/bracket_{gender}.html
    output/{year}/features/{gender}/feature_dataset.csv
    output/scoring_results.csv                        - aggregate Brier scores
"""

import argparse
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
sys.path.insert(0, PROJECT_ROOT)

from src.data_classes.processing.DataManager import MarchMadnessDataManager
from src.data_classes.processing.EloRatingSystem import EloRatingSystem
from src.data_classes.processing.TeamStatsCalculator import TeamStatsCalculator
from src.data_classes.processing.MLModel import MarchMadnessMLModel
from src.data_classes.processing.Predictor import MarchMadnessPredictor
from src.data_classes.simple_predictor import SimplePredictor
from src.approaches import generate_notebook_submission
from src.evaluation.scorer import (
    load_tournament_results,
    score_all_submissions,
    score_submission,
)

try:
    from src.visualization.bracket_viz import visualize_bracket, visualize_bracket_html
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False
    print("Warning: src.visualization.bracket_viz not available - bracket PNGs will be skipped.")


# ── helpers ──────────────────────────────────────────────────────────────────

def _resolve_year_data_dir(base_data_dir: str, year: int) -> str:
    year_dir = os.path.join(base_data_dir, str(year))
    if not os.path.isdir(year_dir):
        raise FileNotFoundError(f"Missing data directory for season {year}: {year_dir}")
    return year_dir


def _resolve_scoring_data_dir(base_data_dir: str, year: int) -> str | None:
    next_year_dir = os.path.join(base_data_dir, str(year + 1))
    if os.path.isdir(next_year_dir):
        return next_year_dir
    return None


def _build_predictor(
    data_dir: str,
    gender: str,
    season: int,
    methods: list[str],
    lookback_years: int,
    calibrate_ml: bool,
) -> MarchMadnessPredictor:
    """Build a fully initialized predictor time-fenced to season Y."""
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
    needs_ml = any(method in {"elo_enhanced", "ensemble"} for method in methods)
    predictor.initialize_models(
        calculate_elo=True,
        calculate_stats=needs_ml,
        train_ml=needs_ml,
        train_years_range=(max(2003, season - lookback_years), season - 1),
        calibrate_ml=calibrate_ml,
    )
    return predictor


def _submission_path(year: int, method: str, gender: str, artifact_suffix: str = "") -> str:
    method_dir = f"{method}_{artifact_suffix}" if artifact_suffix else method
    path = os.path.join(OUTPUT_ROOT, str(year), method_dir)
    os.makedirs(path, exist_ok=True)
    suffix = f"_{artifact_suffix}" if artifact_suffix else ""
    return os.path.join(path, f"{method}{suffix}_{gender}.csv")


def _bracket_path(year: int, method: str, gender: str, artifact_suffix: str = "") -> str:
    method_dir = f"{method}_{artifact_suffix}" if artifact_suffix else method
    path = os.path.join(OUTPUT_ROOT, str(year), method_dir)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"bracket_{gender}.png")


def _bracket_html_path(year: int, method: str, gender: str, artifact_suffix: str = "") -> str:
    method_dir = f"{method}_{artifact_suffix}" if artifact_suffix else method
    path = os.path.join(OUTPUT_ROOT, str(year), method_dir)
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, f"bracket_{gender}.html")


_NOTEBOOK_METHODS = {
    "baseline_massey",
    "xgb_ensemble",
    "xgb_ensemble_v2",
    "massey_direct",
    "seed_spline_only",
    "recency_xgb",
    "massey_blend",
    "poisson_margin",
    "meta_ensemble",
    "seed_matchup_calibration",
}


def _generate_submission(predictor, data_dir: str, season: int, method: str, gender: str) -> pd.DataFrame:
    """Generate a submission DataFrame for all M or W matchups."""
    if method in _NOTEBOOK_METHODS:
        sub = generate_notebook_submission(method, data_dir, season, gender)
    else:
        sub = predictor.generate_predictions(method=method)
    if "ID" not in sub.columns or "Pred" not in sub.columns:
        raise ValueError(f"generate_predictions() returned unexpected columns: {sub.columns.tolist()}")
    return sub


def _generate_bracket_png(predictor, season: int, method: str, out_path: str):
    if not _HAS_VIZ:
        return
    try:
        fig = visualize_bracket(predictor=predictor, season=season, method=method, output_path=out_path)
        plt.close(fig)
    except Exception as e:
        print(f"      [warn] bracket PNG failed: {e}")


def _generate_bracket_html(predictor, season: int, method: str, out_path: str):
    if not _HAS_VIZ:
        return
    try:
        visualize_bracket_html(
            predictor=predictor,
            season=season,
            method=method,
            output_path=out_path,
        )
    except Exception as e:
        print(f"      [warn] bracket HTML failed: {e}")
        traceback.print_exc()


def _build_submission_backed_predictor(
    data_dir: str,
    gender: str,
    season: int,
    submission_df: pd.DataFrame,
):
    dm = MarchMadnessDataManager(data_dir, gender=gender, current_season=season)
    dm.load_data()
    return SimplePredictor(
        teams_df=dm.data["teams"].copy(),
        seeds_df=dm.data["tourney_seeds"].copy(),
        slots_df=dm.data["tourney_slots"].copy(),
        predictions_df=submission_df[["ID", "Pred"]].copy(),
        results_df=dm.data.get("tourney_results"),
        current_season=season,
    )


def _score_submission(sub_df, data_dir, season, gender) -> dict:
    """Return scoring stats dict or empty dict on failure."""
    try:
        results_df = load_tournament_results(data_dir, season, gender)
        if len(results_df) == 0:
            return {}
        stats = score_submission(sub_df, results_df, season, gender)
        return stats
    except FileNotFoundError:
        return {}


# ── main ─────────────────────────────────────────────────────────────────────

def run(
    years,
    genders,
    methods,
    base_data_dir,
    skip_existing=True,
    skip_brackets=False,
    lookback_years=8,
    calibrate_ml=False,
    artifact_suffix="",
):
    rows = []

    for gender in genders:
        print(f"\n{'='*60}")
        print(f"Gender: {gender}")
        print(f"{'='*60}")

        for year in years:
            print(f"\n  Season: {year}")
            prediction_data_dir = _resolve_year_data_dir(base_data_dir, year)
            scoring_data_dir = _resolve_scoring_data_dir(base_data_dir, year)
            predictor = None

            for method in methods:
                sub_path = _submission_path(
                    year, method, gender, artifact_suffix=artifact_suffix
                )
                tag = f"{year}/{method}/{gender}"

                # ── submission CSV ───────────────────────────────────────────
                if skip_existing and os.path.exists(sub_path):
                    print(f"    [skip] {tag} - submission exists, loading from disk")
                    sub_df = pd.read_csv(sub_path)
                else:
                    needs_predictor = method not in _NOTEBOOK_METHODS
                    if predictor is None and needs_predictor:
                        try:
                            print(
                                f"    Building predictor (season={year}, "
                                f"gender={gender}, data={prediction_data_dir})..."
                            )
                            predictor = _build_predictor(
                                prediction_data_dir,
                                gender,
                                year,
                                methods,
                                lookback_years,
                                calibrate_ml,
                            )
                        except Exception as e:
                            print(f"    [ERROR] Could not build predictor: {e}")
                            continue  # skip this method but try others

                    try:
                        print(f"    Generating {tag} ...")
                        sub_df = _generate_submission(
                            predictor,
                            prediction_data_dir,
                            year,
                            method,
                            gender,
                        )
                        sub_df.to_csv(sub_path, index=False)
                        print(f"    [OK]  saved {len(sub_df)} rows → {sub_path}")
                    except Exception as e:
                        print(f"    [ERROR] {tag} submission failed: {e}")
                        continue

                # ── bracket artifacts ───────────────────────────────────────
                if not skip_brackets:
                    bracket_path = _bracket_path(
                        year, method, gender, artifact_suffix=artifact_suffix
                    )
                    bracket_html_path = _bracket_html_path(
                        year, method, gender, artifact_suffix=artifact_suffix
                    )
                    if (
                        skip_existing
                        and os.path.exists(bracket_path)
                        and os.path.exists(bracket_html_path)
                    ):
                        print(f"    [skip] {tag} - bracket artifacts exist")
                    else:
                        bracket_predictor = predictor
                        if predictor is None and method not in _NOTEBOOK_METHODS:
                            try:
                                predictor = _build_predictor(
                                    prediction_data_dir,
                                    gender,
                                    year,
                                    methods,
                                    lookback_years,
                                    calibrate_ml,
                                )
                            except Exception as e:
                                print(f"    [warn] predictor unavailable for bracket: {e}")
                            bracket_predictor = predictor
                        if bracket_predictor is None:
                            try:
                                print(f"    Building submission-backed bracket predictor for {tag}...")
                                bracket_predictor = _build_submission_backed_predictor(
                                    prediction_data_dir,
                                    gender,
                                    year,
                                    sub_df,
                                )
                            except Exception as e:
                                print(f"    [warn] submission-backed bracket unavailable: {e}")
                        if bracket_predictor is not None:
                            _generate_bracket_html(
                                bracket_predictor,
                                year,
                                method,
                                bracket_html_path,
                            )
                            if os.path.exists(bracket_html_path):
                                print(f"    [OK]  bracket HTML -> {bracket_html_path}")
                            _generate_bracket_png(bracket_predictor, year, method, bracket_path)
                            if os.path.exists(bracket_path):
                                print(f"    [OK]  bracket PNG -> {bracket_path}")

                # ── score ────────────────────────────────────────────────────
                stats = _score_submission(sub_df, scoring_data_dir, year, gender) if scoring_data_dir else {}
                if stats.get("n_games", 0) > 0:
                    model_name = f"{method}_{artifact_suffix}" if artifact_suffix else method
                    row = {
                        "year": year,
                        "gender": gender,
                        "model": model_name,
                        "brier_score": round(stats["brier_score"], 5),
                        "accuracy": round(stats["accuracy"], 5),
                        "n_games": stats["n_games"],
                        "missing_games": stats.get("missing_games", 0),
                    }
                    # per-round Brier scores
                    for rnd, val in stats.get("per_round", {}).items():
                        col = "round_" + rnd.lower().replace(" ", "_").replace("of_", "")
                        row[col] = round(val, 5)
                    rows.append(row)
                    print(
                        f"    Brier: {stats['brier_score']:.4f}  "
                        f"Accuracy: {stats['accuracy']:.3f}  "
                        f"({stats['n_games']} games)"
                    )
                elif scoring_data_dir is None:
                    print(f"    [skip] no scoring data found for season {year + 1}")

    # ── also score any pre-existing archive submissions not covered above ────
    print(f"\n{'='*60}")
    print("Scoring all archive submissions (including pre-existing)...")
    archive_dir = OUTPUT_ROOT
    full_frames = []
    for year in years:
        scoring_data_dir = _resolve_scoring_data_dir(base_data_dir, year)
        if scoring_data_dir is None:
            continue
        scored = score_all_submissions(archive_dir, scoring_data_dir, seasons=[year])
        if not scored.empty:
            full_frames.append(scored)
    full_results = pd.concat(full_frames, ignore_index=True) if full_frames else pd.DataFrame()
    if not full_results.empty:
        print(full_results.to_string(index=False))

    # ── write aggregate scoring results ─────────────────────────────────────
    results_df = full_results.copy()
    if results_df.empty and rows:
        results_df = pd.DataFrame(rows)

    if not results_df.empty:
        results_df = results_df.sort_values(["year", "gender", "brier_score"])
        out_path = os.path.join(OUTPUT_ROOT, "scoring_results.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved scoring_results.csv → {out_path}  ({len(results_df)} rows)")
    else:
        print("\nNo scoring rows generated (no playable tournaments found or all skipped).")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate historical March Madness submission CSVs, bracket PNGs, and scoring results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025],
        help="Tournament seasons to process",
    )
    parser.add_argument(
        "--genders", nargs="+", choices=["M", "W"], default=["M", "W"],
        help="Genders to process",
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["elo", "elo_enhanced", "baseline_massey", "xgb_ensemble"],
        choices=[
            "elo", "elo_enhanced", "ensemble",
            "baseline_massey", "xgb_ensemble", "xgb_ensemble_v2", "massey_direct",
            "seed_spline_only", "recency_xgb", "massey_blend", "poisson_margin",
            "meta_ensemble", "seed_matchup_calibration",
        ],
        help="Prediction methods to run",
    )
    parser.add_argument(
        "--base-data-dir", default=os.path.join(PROJECT_ROOT, "data"),
        help="Root data directory containing season folders like data/2025 and data/2026",
    )
    parser.add_argument(
        "--lookback-years", type=int, default=8,
        help="Number of completed tournament seasons to use for ML training",
    )
    parser.add_argument(
        "--calibrate-ml", action="store_true",
        help="Enable slower isotonic calibration for elo_enhanced",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-generate even if submission CSV already exists",
    )
    parser.add_argument(
        "--skip-brackets", action="store_true",
        help="Skip bracket PNG generation (faster)",
    )
    parser.add_argument(
        "--artifact-suffix",
        default="",
        help="Optional suffix added to generated artifact names for side-by-side comparisons",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        years=args.years,
        genders=args.genders,
        methods=args.methods,
        base_data_dir=args.base_data_dir,
        skip_existing=not args.no_skip_existing,
        skip_brackets=args.skip_brackets,
        lookback_years=args.lookback_years,
        calibrate_ml=args.calibrate_ml,
        artifact_suffix=args.artifact_suffix,
    )
