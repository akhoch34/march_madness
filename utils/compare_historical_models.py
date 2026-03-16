import argparse
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def load_scores(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing scoring file: {path}")
    df = pd.read_csv(path)
    if "accuracy" not in df.columns:
        df["accuracy"] = float("nan")
    return df


def summarize(df: pd.DataFrame, years=None, genders=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if years:
        df = df[df["year"].isin(years)]
    if genders:
        df = df[df["gender"].isin(genders)]

    model_summary = (
        df.groupby(["model", "gender"], as_index=False)
        .agg(
            avg_brier=("brier_score", "mean"),
            avg_accuracy=("accuracy", "mean"),
            seasons=("year", "nunique"),
        )
        .sort_values(["gender", "avg_brier", "avg_accuracy"], ascending=[True, True, False])
    )

    season_summary = (
        df.sort_values(["year", "gender", "brier_score", "accuracy"], ascending=[True, True, True, False])
        .groupby(["year", "gender"], as_index=False)
        .first()
    )

    return model_summary, season_summary


def main():
    parser = argparse.ArgumentParser(description="Summarize historical March Madness model scores.")
    parser.add_argument(
        "--scores-path",
        default=os.path.join(PROJECT_ROOT, "output", "scoring_results.csv"),
        help="Path to scoring_results.csv",
    )
    parser.add_argument("--years", nargs="+", type=int, help="Optional season filter")
    parser.add_argument("--genders", nargs="+", choices=["M", "W"], help="Optional gender filter")
    args = parser.parse_args()

    df = load_scores(args.scores_path)
    model_summary, season_summary = summarize(df, years=args.years, genders=args.genders)

    print("\nBest model by season")
    if season_summary.empty:
        print("No rows matched the requested filters.")
    else:
        print(season_summary.to_string(index=False))

    print("\nAverage performance by model")
    if model_summary.empty:
        print("No rows matched the requested filters.")
    else:
        print(model_summary.to_string(index=False))


if __name__ == "__main__":
    main()
