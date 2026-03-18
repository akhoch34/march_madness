"""
Brier score evaluation for March Madness submissions.

Scoring metric: Brier score = mean((pred - actual)^2) over all played games.
Lower is better. Equivalent to MSE.

Submission format: ID,Pred
  ID = {season}_{lower_teamid}_{higher_teamid}
  Pred = P(lower ID team wins)

Only played tournament games are scored; the full submission covers all possible matchups.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# Round labels by day number — separate tables for men's and women's tournaments.
# Women's games run on later calendar days than men's (roughly +5 days).
ROUND_DAY_MAP = {
    "M": {
        "Play-In": (132, 135),
        "Round of 64": (136, 137),
        "Round of 32": (138, 139),
        "Sweet 16": (143, 145),
        "Elite 8": (146, 148),
        "Final Four": (152, 153),
        "Championship": (154, 155),
    },
    "W": {
        "Play-In": (136, 140),
        "Round of 64": (141, 145),
        "Round of 32": (144, 147),
        "Sweet 16": (148, 153),
        "Elite 8": (152, 156),
        "Final Four": (156, 160),
        "Championship": (159, 163),
    },
}


def _resolve_results_path(data_dir: str, filename: str) -> str:
    direct = Path(data_dir) / filename
    if direct.exists():
        return str(direct)

    matches = sorted(Path(data_dir).rglob(filename), reverse=True)
    if matches:
        return str(matches[0])

    raise FileNotFoundError(f"Could not find '{filename}' under {data_dir}")


def _day_to_round(day_num: int, gender: str = "M") -> str:
    day_map = ROUND_DAY_MAP.get(gender, ROUND_DAY_MAP["M"])
    for round_name, (low, high) in day_map.items():
        if low <= day_num <= high:
            return round_name
    return "Unknown"


def load_tournament_results(
    data_dir: str, season: int, gender: str = "M"
) -> pd.DataFrame:
    """
    Load actual NCAA tournament game results for a given season.

    Parameters
    ----------
    data_dir : str
        Root data directory (e.g. "data/2026").
    season : int
        Tournament season year (e.g. 2025).
    gender : str
        "M" or "W".

    Returns
    -------
    pd.DataFrame with columns: Season, DayNum, WTeamID, LTeamID, WScore, LScore
        Filtered to the requested season.
    """
    results_path = _resolve_results_path(
        data_dir, f"{gender}NCAATourneyCompactResults.csv"
    )
    df = pd.read_csv(results_path)
    season_df = df[df["Season"] == season].copy()
    season_df["Round"] = season_df["DayNum"].apply(lambda d: _day_to_round(d, gender))
    return season_df


def score_submission(
    submission_df: pd.DataFrame,
    results_df: pd.DataFrame,
    season: int,
    gender: str = "M",
) -> dict:
    """
    Compute Brier score for played games in a submission.

    For each played game:
      - Build ID = {season}_{min(w,l)}_{max(w,l)}
      - actual = 1.0 if lower_id == WTeamID, 0.0 if lower_id == LTeamID
      - brier_per_game = (pred - actual)^2

    Parameters
    ----------
    submission_df : pd.DataFrame
        Submission with columns [ID, Pred].
    results_df : pd.DataFrame
        Actual results (from load_tournament_results), filtered to one gender.
    season : int
        Season year (used for filtering submission rows).
    gender : str
        "M" or "W" — only used for per-round labeling.

    Returns
    -------
    dict with keys:
        brier_score : float
        n_games : int
        per_round : dict[str, float]  (round name -> Brier score)
        missing_games : int  (games in results but not found in submission)
    """
    # Build lookup from submission
    pred_lookup = dict(zip(submission_df["ID"], submission_df["Pred"]))

    brier_values = []
    accuracy_values = []
    round_brier = {}
    missing = 0

    for _, game in results_df.iterrows():
        w_id = int(game["WTeamID"])
        l_id = int(game["LTeamID"])
        low_id = min(w_id, l_id)
        high_id = max(w_id, l_id)
        game_id = f"{season}_{low_id}_{high_id}"

        if game_id not in pred_lookup:
            missing += 1
            continue

        pred = float(pred_lookup[game_id])
        # actual = 1 if lower_id team won
        actual = 1.0 if low_id == w_id else 0.0
        brier = (pred - actual) ** 2
        brier_values.append(brier)
        accuracy_values.append(float((pred >= 0.5) == bool(actual)))

        round_name = game.get("Round", "Unknown")
        round_brier.setdefault(round_name, []).append(brier)

    per_round_scores = {
        r: float(np.mean(v)) for r, v in round_brier.items()
    }

    return {
        "brier_score": float(np.mean(brier_values)) if brier_values else float("nan"),
        "accuracy": float(np.mean(accuracy_values)) if accuracy_values else float("nan"),
        "n_games": len(brier_values),
        "per_round": per_round_scores,
        "missing_games": missing,
    }


def score_all_submissions(
    archive_dir: str,
    data_dir: str,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """
    Score all archived submission CSVs across all scorable years.

    Walks archive_dir looking for CSV files organized as either:
      archive_dir/{year}/{model_name}.csv
      archive_dir/{year}/submissions/{model_name}.csv

    Each CSV must have columns [ID, Pred]. IDs that don't match the season
    are ignored, so a combined men's+women's file works fine.

    Parameters
    ----------
    archive_dir : str
        Path to historical outputs, typically output/.
    data_dir : str
        Root data directory containing year-specific tournament results
        (e.g. "data/2026" which has cumulative results through 2025).
    seasons : list[int], optional
        Seasons to score. Defaults to [2022, 2023, 2024, 2025].

    Returns
    -------
    pd.DataFrame with columns: year, model, gender, brier_score, n_games
    """
    if seasons is None:
        seasons = [2022, 2023, 2024, 2025]

    archive_path = Path(archive_dir)
    rows = []

    for season in seasons:
        print(f"Scoring submissions for season {season}...")
        year_dir = archive_path / str(season)
        if not year_dir.exists():
            print(f"  No directory found for {season} under {archive_dir}")
            continue

        # Collect CSVs from:
        #   1. year_dir/{method}/{file}.csv  — current per-method flat structure
        #   2. year_dir/submissions/{file}.csv  — legacy archive structure
        #   3. year_dir/{file}.csv  — top-level combined submissions
        _SKIP_DIRS = {"features"}
        csv_files = []
        for child in sorted(year_dir.iterdir()):
            if child.name in _SKIP_DIRS:
                continue
            if child.is_dir():
                legacy_sub = child / "submissions"
                scan_dir = legacy_sub if legacy_sub.exists() else child
                csv_files.extend(sorted(scan_dir.glob("*.csv")))
            elif child.suffix == ".csv":
                csv_files.append(child)

        if not csv_files:
            print(f"  No submission files found for {season} under {archive_dir}")
            continue
        print(f"  Found {len(csv_files)} submission files")
        for file_index, csv_file in enumerate(csv_files, start=1):
            model_name = csv_file.stem
            print(f"  [{file_index}/{len(csv_files)}] Loading {csv_file.name}...")
            try:
                sub_df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"  Warning: could not read {csv_file}: {e}")
                continue

            if "ID" not in sub_df.columns or "Pred" not in sub_df.columns:
                print(f"  Skipping {csv_file}: missing ID or Pred columns")
                continue

            # Score separately for men's and women's
            for gender in ("M", "W"):
                try:
                    results_df = load_tournament_results(data_dir, season, gender)
                except FileNotFoundError:
                    print(f"    No {gender} results found for season {season} in {data_dir}")
                    continue

                if len(results_df) == 0:
                    print(f"    No played {gender} tournament games found for season {season}")
                    continue

                stats = score_submission(sub_df, results_df, season, gender)

                if stats["n_games"] == 0:
                    print(f"    {model_name} has no scorable {gender} rows for {season}")
                    continue  # Submission doesn't cover this gender

                rows.append(
                    {
                        "year": season,
                        "model": model_name,
                        "gender": gender,
                        "brier_score": stats["brier_score"],
                        "accuracy": stats["accuracy"],
                        "n_games": stats["n_games"],
                        "missing_games": stats["missing_games"],
                    }
                )
                print(
                    f"    {gender}: brier={stats['brier_score']:.4f} "
                    f"accuracy={stats['accuracy']:.3f} games={stats['n_games']}"
                )

    return pd.DataFrame(rows)


def generate_leaderboard(results_df: pd.DataFrame, sort_by: str = "avg_brier") -> pd.DataFrame:
    """
    Pivot scoring results to a wide leaderboard table.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of score_all_submissions().
    sort_by : str
        Column to sort by. Options: "avg_brier", "std_brier", "avg_rank", "n_years".

    Returns
    -------
    pd.DataFrame with columns: model, gender, avg_brier, std_brier, avg_rank, n_years,
        plus one column per year.
    """
    if results_df.empty:
        return pd.DataFrame()

    rows = []
    for (model, gender), grp in results_df.groupby(["model", "gender"]):
        years = sorted(grp["year"].unique())
        brier_vals = grp.set_index("year")["brier_score"].to_dict()
        avg_b = float(np.mean(list(brier_vals.values())))
        std_b = float(np.std(list(brier_vals.values()))) if len(brier_vals) > 1 else 0.0
        row: dict = {"model": model, "gender": gender, "avg_brier": round(avg_b, 5),
                     "std_brier": round(std_b, 5), "n_years": len(years)}
        for y in years:
            row[str(y)] = round(brier_vals[y], 5)
        rows.append(row)

    lb = pd.DataFrame(rows)
    if lb.empty:
        return lb

    # Compute avg_rank per gender group
    for gender, gdf in lb.groupby("gender"):
        lb.loc[lb["gender"] == gender, "avg_rank"] = lb.loc[lb["gender"] == gender, "avg_brier"].rank()

    year_cols = [c for c in lb.columns if c.isdigit()]
    col_order = ["model", "gender", "avg_brier", "std_brier", "avg_rank", "n_years"] + sorted(year_cols)
    lb = lb[[c for c in col_order if c in lb.columns]]

    if sort_by in lb.columns:
        lb = lb.sort_values(sort_by)
    return lb.reset_index(drop=True)


def generate_head_to_head(results_df: pd.DataFrame, gender: str = "M") -> pd.DataFrame:
    """
    Build a square matrix of head-to-head Brier score wins.

    Entry [A, B] = number of years where model A had a lower Brier score than model B.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of score_all_submissions().
    gender : str
        "M" or "W".

    Returns
    -------
    pd.DataFrame — square matrix indexed and columned by model name.
    """
    gdf = results_df[results_df["gender"] == gender].copy()
    if gdf.empty:
        return pd.DataFrame()

    models = sorted(gdf["model"].unique())
    matrix = pd.DataFrame(0, index=models, columns=models)

    for year, year_grp in gdf.groupby("year"):
        brier_by_model = year_grp.set_index("model")["brier_score"].to_dict()
        for m_a in models:
            for m_b in models:
                if m_a == m_b:
                    continue
                if m_a in brier_by_model and m_b in brier_by_model:
                    if brier_by_model[m_a] < brier_by_model[m_b]:
                        matrix.loc[m_a, m_b] += 1

    return matrix


def generate_round_breakdown(results_df: pd.DataFrame, gender: str = "M") -> str:
    """
    Formatted table of per-round Brier scores averaged across years.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of score_all_submissions(). Should contain round_* columns
        if generated by generate_historical_artifacts.py.
    gender : str
        "M" or "W".

    Returns
    -------
    str : Formatted table string.
    """
    gdf = results_df[results_df["gender"] == gender].copy()
    if gdf.empty:
        return f"No data for gender={gender}."

    round_cols = [c for c in gdf.columns if c.startswith("round_")]
    if not round_cols:
        return "No per-round columns found in results_df. Run with generate_historical_artifacts.py to include them."

    agg = gdf.groupby("model")[round_cols].mean()
    agg["avg_brier"] = gdf.groupby("model")["brier_score"].mean()
    agg = agg.sort_values("avg_brier")

    gender_label = "Men's" if gender == "M" else "Women's"
    lines = [
        "=" * 90,
        f"ROUND BREAKDOWN — {gender_label}",
        "-" * 90,
    ]
    pretty_rounds = [c.replace("round_", "").replace("_", " ").title() for c in round_cols]
    header = f"{'Model':<35}" + "".join(f"{r:>12}" for r in pretty_rounds) + f"{'Overall':>12}"
    lines.append(header)
    lines.append("-" * 90)
    for model, row in agg.iterrows():
        line = f"{model:<35}"
        for col in round_cols:
            val = row.get(col, float("nan"))
            line += f"{val:>12.4f}" if not np.isnan(val) else f"{'—':>12}"
        line += f"{row['avg_brier']:>12.4f}"
        lines.append(line)
    lines.append("=" * 90)
    return "\n".join(lines)


def generate_scoring_report(results_df: pd.DataFrame) -> str:
    """
    Pretty-print a table of all models × years with Brier scores.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of score_all_submissions().

    Returns
    -------
    str : Formatted table string.
    """
    if results_df.empty:
        return "No scoring results available."

    lines = []
    lines.append("=" * 72)
    lines.append("MARCH MADNESS HISTORICAL BRIER SCORE REPORT")
    lines.append("Lower is better. Perfect predictions = 0.0, Random = 0.25")
    lines.append("=" * 72)

    for gender in ("M", "W"):
        gender_label = "Men's" if gender == "M" else "Women's"
        gdf = results_df[results_df["gender"] == gender]
        if gdf.empty:
            continue

        lines.append(f"\n{gender_label} Tournament")
        lines.append("-" * 72)

        pivot = gdf.pivot_table(
            index="model", columns="year", values="brier_score", aggfunc="first"
        )
        pivot["avg"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("avg")

        # Header
        years = sorted(gdf["year"].unique())
        header = f"{'Model':<35}" + "".join(f"{y:>10}" for y in years) + f"{'Avg':>10}"
        lines.append(header)
        lines.append("-" * 72)

        for model, row in pivot.iterrows():
            line = f"{model:<35}"
            for y in years:
                val = row.get(y, float("nan"))
                if np.isnan(val):
                    line += f"{'—':>10}"
                else:
                    line += f"{val:>10.4f}"
            line += f"{row['avg']:>10.4f}"
            lines.append(line)

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)
