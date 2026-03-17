"""Tests for the Brier score evaluation module."""

import numpy as np
import pandas as pd
import pytest
from conftest import requires_data

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.scorer import score_submission, load_tournament_results, _day_to_round


# ── unit tests (no real data needed) ────────────────────────────────────────

def _make_results(season, games):
    """Build a minimal results DataFrame for testing."""
    rows = []
    for w, l, day in games:
        rows.append({"Season": season, "DayNum": day, "WTeamID": w, "LTeamID": l})
    df = pd.DataFrame(rows)
    df["Round"] = df["DayNum"].apply(lambda d: _day_to_round(d, "M"))
    return df


def _make_submission(season, pairs_preds):
    rows = [
        {"ID": f"{season}_{min(a,b)}_{max(a,b)}", "Pred": p}
        for a, b, p in pairs_preds
    ]
    return pd.DataFrame(rows)


def test_brier_all_half():
    """All 0.5 predictions should give Brier = 0.25 (random baseline)."""
    season = 9999
    results = _make_results(season, [(1, 2, 136), (3, 4, 136)])
    sub = _make_submission(season, [(1, 2, 0.5), (3, 4, 0.5)])
    stats = score_submission(sub, results, season)
    assert abs(stats["brier_score"] - 0.25) < 1e-9


def test_brier_perfect_predictions():
    """Perfect predictions should give Brier = 0.0."""
    season = 9998
    # lower-id team wins both games
    results = _make_results(season, [(1, 2, 136), (3, 4, 136)])
    # pred = 1.0 means lower-id team wins — matches actuals
    sub = _make_submission(season, [(1, 2, 1.0), (3, 4, 1.0)])
    stats = score_submission(sub, results, season)
    assert abs(stats["brier_score"] - 0.0) < 1e-9
    assert abs(stats["accuracy"] - 1.0) < 1e-9


def test_brier_worst_predictions():
    """Completely wrong predictions (pred=1.0 but higher-id team wins) → Brier = 1.0."""
    season = 9997
    # higher-id team wins (WTeamID > LTeamID)
    results = _make_results(season, [(2, 1, 136)])
    # pred = 1.0 means team with lower ID wins — but lower-id (1) actually lost
    sub = _make_submission(season, [(1, 2, 1.0)])
    stats = score_submission(sub, results, season)
    assert abs(stats["brier_score"] - 1.0) < 1e-9
    assert abs(stats["accuracy"] - 0.0) < 1e-9


def test_missing_games_counted():
    season = 9996
    results = _make_results(season, [(1, 2, 136), (3, 4, 136)])
    sub = _make_submission(season, [(1, 2, 0.5)])  # missing game (3,4)
    stats = score_submission(sub, results, season)
    assert stats["missing_games"] == 1
    assert stats["n_games"] == 1


def test_day_to_round_men():
    assert _day_to_round(136, "M") == "Round of 64"
    assert _day_to_round(154, "M") == "Championship"
    assert _day_to_round(1, "M") == "Unknown"


def test_day_to_round_women():
    # Women's ranges are later — day 142 should be in "Round of 64"
    r = _day_to_round(142, "W")
    assert r != "Unknown", f"Day 142 should map to a women's round, got {r!r}"


# ── integration tests (require data/2026) ───────────────────────────────────

@requires_data
def test_load_tournament_results(data_dir):
    results = load_tournament_results(data_dir, 2024, "M")
    assert len(results) > 60  # typical men's tournament has 63-67 games
    assert "Round" in results.columns
    assert (results["Round"] != "Unknown").any()


@requires_data
def test_score_archive_submission(data_dir):
    """Score a known archive submission and check the Brier is in a sane range."""
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    archive_base = os.path.join(repo_root, "output", "historical", "submissions")
    if not os.path.isdir(archive_base):
        archive_base = os.path.join(repo_root, "archive", "submissions")
    csv_candidates = []
    for year in [2024, 2025]:
        year_dir = os.path.join(archive_base, str(year))
        if os.path.isdir(year_dir):
            for f in os.listdir(year_dir):
                if f.endswith(".csv"):
                    csv_candidates.append((year, os.path.join(year_dir, f)))
            if csv_candidates:
                break

    if not csv_candidates:
        pytest.skip("No archive submission CSVs found")

    season, csv_path = csv_candidates[0]
    sub = pd.read_csv(csv_path)
    if "ID" not in sub.columns or "Pred" not in sub.columns:
        pytest.skip("Archive CSV lacks ID/Pred columns")

    results = load_tournament_results(data_dir, season, "M")
    stats = score_submission(sub, results, season, "M")
    assert stats["n_games"] > 0
    # Brier score for any remotely reasonable model should be < 0.25 (random)
    assert stats["brier_score"] < 0.30, f"Unexpectedly high Brier: {stats['brier_score']}"
