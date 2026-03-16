"""
Generate bracket PNGs from existing submission CSVs — no model retraining required.

Scans output/{year}/submissions/ for {method}_{gender}.csv files and generates
bracket images using the PIL-based BracketSimulator.

Usage (from project root):
    poetry run python utils/generate_brackets_from_submissions.py
    poetry run python utils/generate_brackets_from_submissions.py \\
        --years 2025 --genders M --methods xgb_ensemble
    poetry run python utils/generate_brackets_from_submissions.py \\
        --years 2022 2023 2024 2025 --no-skip-existing

Outputs:
    output/{year}/brackets/{method}/{gender}/bracket.png
    output/{year}/brackets/{method}/{gender}/bracket_historical.png  (for 2022-2025)
"""

import argparse
import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
sys.path.insert(0, PROJECT_ROOT)

from src.data_classes.processing.DataManager import MarchMadnessDataManager
from src.data_classes.simple_predictor import SimplePredictor
from src.data_classes.bracket.BracketGenerator import BracketSimulator


DEFAULT_YEARS = [2022, 2023, 2024, 2025]
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "2026")
HISTORICAL_YEARS = {2022, 2023, 2024, 2025}


def _parse_submission_filename(filename: str):
    """
    Parse method and gender from filename like 'xgb_ensemble_M.csv'.
    Gender is the last char before .csv; method is everything before that.
    Returns (method, gender) or (None, None) if unparseable.
    """
    if not filename.endswith(".csv"):
        return None, None
    stem = filename[:-4]  # remove .csv
    if len(stem) < 3 or stem[-1] not in ("M", "W") or stem[-2] != "_":
        return None, None
    gender = stem[-1]
    method = stem[:-2]  # remove _M or _W
    return method, gender


def _discover_submissions(output_root: str, years: list, genders: list, methods: list):
    """
    Scan output/{year}/submissions/ and return list of (year, method, gender, path).
    Filtered by genders and methods if specified.
    """
    found = []
    for year in years:
        sub_dir = os.path.join(output_root, str(year), "submissions")
        if not os.path.isdir(sub_dir):
            continue
        for fname in sorted(os.listdir(sub_dir)):
            method, gender = _parse_submission_filename(fname)
            if method is None:
                continue
            if genders and gender not in genders:
                continue
            if methods and method not in methods:
                continue
            fpath = os.path.join(sub_dir, fname)
            found.append((year, method, gender, fpath))
    return found


def _build_predictor(data_dir: str, gender: str, season: int, submission_df: pd.DataFrame) -> SimplePredictor:
    """Build a SimplePredictor backed by a pre-generated submission CSV."""
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


def _bracket_dir(output_root: str, year: int, method: str, gender: str) -> str:
    path = os.path.join(output_root, str(year), "brackets", method, gender)
    os.makedirs(path, exist_ok=True)
    return path


def generate_bracket(
    year: int,
    method: str,
    gender: str,
    submission_path: str,
    data_dir: str,
    output_root: str,
    skip_existing: bool = True,
):
    out_dir = _bracket_dir(output_root, year, method, gender)
    predicted_path = os.path.join(out_dir, "bracket.png")
    historical_path = os.path.join(out_dir, "bracket_historical.png")

    need_predicted = not skip_existing or not os.path.exists(predicted_path)
    need_historical = (year in HISTORICAL_YEARS) and (not skip_existing or not os.path.exists(historical_path))

    if not need_predicted and not need_historical:
        print(f"    [skip] {year}/{method}/{gender} — already exists")
        return True

    try:
        submission_df = pd.read_csv(submission_path)
        # Filter to the target season
        season_prefix = f"{year}_"
        submission_df = submission_df[submission_df["ID"].str.startswith(season_prefix)]
        if len(submission_df) == 0:
            print(f"    [warn] {year}/{method}/{gender} — no rows for season {year} in submission")
            return False
    except Exception as e:
        print(f"    [error] reading {submission_path}: {e}")
        return False

    try:
        predictor = _build_predictor(data_dir, gender, year, submission_df)
    except Exception as e:
        print(f"    [error] building predictor for {year}/{method}/{gender}: {e}")
        return False

    sim = BracketSimulator(predictor=predictor)
    try:
        sim.use_predictor_data(season=year)
        sim.build_bracket_tree(season=year)
    except Exception as e:
        print(f"    [error] building bracket tree for {year}/{method}/{gender}: {e}")
        traceback.print_exc()
        return False

    success = True

    if need_predicted:
        try:
            sim.visualize_bracket(method="ensemble", output_path=predicted_path, show_plot=False)
            print(f"    [ok]   {predicted_path}")
        except Exception as e:
            print(f"    [warn] predicted bracket failed for {year}/{method}/{gender}: {e}")
            traceback.print_exc()
            success = False

    if need_historical:
        try:
            sim.visualize_historical_bracket(season=year, method="ensemble", output_path=historical_path, show_plot=False)
            print(f"    [ok]   {historical_path}")
        except Exception as e:
            print(f"    [warn] historical bracket failed for {year}/{method}/{gender}: {e}")
            success = False

    return success


def run(
    years=None,
    genders=None,
    methods=None,
    data_dir=None,
    output_root=None,
    skip_existing=True,
):
    if years is None:
        years = DEFAULT_YEARS
    if genders is None:
        genders = ["M", "W"]
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    if output_root is None:
        output_root = OUTPUT_ROOT

    submissions = _discover_submissions(output_root, years, genders, methods)

    if not submissions:
        print("No matching submissions found.")
        return

    print(f"Found {len(submissions)} submission(s) to process.\n")

    ok = 0
    fail = 0
    for year, method, gender, path in submissions:
        print(f"  {year}/{method}/{gender}")
        result = generate_bracket(
            year=year,
            method=method,
            gender=gender,
            submission_path=path,
            data_dir=data_dir,
            output_root=output_root,
            skip_existing=skip_existing,
        )
        if result:
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate bracket PNGs from existing submission CSVs."
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=DEFAULT_YEARS,
        help="Seasons to process (default: 2022 2023 2024 2025)",
    )
    parser.add_argument(
        "--genders", nargs="+", choices=["M", "W"], default=["M", "W"],
        help="Genders to process (default: M W)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Specific methods to process (default: all found in submissions dir)",
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help=f"Cumulative data root (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_ROOT,
        help=f"Output root directory (default: {OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Regenerate brackets even if bracket.png already exists",
    )
    args = parser.parse_args()

    run(
        years=args.years,
        genders=args.genders,
        methods=args.methods,
        data_dir=args.data_dir,
        output_root=args.output_dir,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
