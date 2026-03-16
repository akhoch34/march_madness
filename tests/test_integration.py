"""
End-to-end integration test: load data → ELO predict → score 2025 tourney.

Marked as slow; skip by default with: pytest -m "not slow"
Run with: pytest tests/test_integration.py -v -m slow
"""

import pandas as pd
import pytest
from conftest import requires_data, DATA_DIR


@pytest.mark.slow
@requires_data
def test_elo_predictions_brier_below_random(men_data_manager, men_elo):
    """ELO-only predictions for 2025 men's tournament should beat random (Brier < 0.25)."""
    from itertools import combinations
    import numpy as np
    from src.evaluation.scorer import score_submission, load_tournament_results

    dm = men_data_manager
    elo = men_elo
    season = 2025

    seeds_df = dm.data["tourney_seeds"]
    season_seeds = seeds_df[seeds_df["Season"] == season]
    team_ids = season_seeds["TeamID"].values

    rows = []
    for t1, t2 in combinations(sorted(team_ids), 2):
        pred = elo.predict_game(t1, t2, day_num=132, season=season)
        pred = float(np.clip(pred, 0.025, 0.975))
        rows.append({"ID": f"{season}_{t1}_{t2}", "Pred": pred})

    sub = pd.DataFrame(rows)
    results = load_tournament_results(DATA_DIR, season, "M")
    stats = score_submission(sub, results, season, "M")

    assert stats["n_games"] > 0, "No games were scored"
    brier = stats["brier_score"]
    assert brier < 0.25, f"ELO Brier {brier:.4f} is not better than random (0.25)"
    print(f"\nELO Brier 2025 Men's: {brier:.4f}  ({stats['n_games']} games)")
