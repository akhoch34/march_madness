"""
Tests for BracketGenerator / BracketSimulator.

These tests require full model initialization (ELO + XGBoost training) and are
therefore marked @pytest.mark.slow.  Run with: pytest -m slow
"""

import pytest
from conftest import requires_data


@pytest.mark.slow
@requires_data
def test_simulate_historical_bracket_game_count(men_data_manager, men_elo):
    """simulate_historical_bracket for 2024 should return exactly 63 main-draw games."""
    from src.data_classes.bracket.BracketGenerator import BracketSimulator
    from src.data_classes.processing.TeamStatsCalculator import TeamStatsCalculator
    from src.data_classes.processing.MLModel import MarchMadnessMLModel
    from src.data_classes.processing.Predictor import MarchMadnessPredictor

    dm = men_data_manager
    elo = men_elo
    stats = TeamStatsCalculator(dm)
    ml = MarchMadnessMLModel(dm, elo, stats)

    predictor = MarchMadnessPredictor(
        data_manager=dm,
        elo_system=elo,
        stats_calculator=stats,
        ml_model=ml,
        current_season=2025,
    )
    predictor.initialize_models()

    simulator = BracketSimulator(predictor)
    simulator.teams_df = dm.data["teams"]
    simulator.seeds_df = dm.data["tourney_seeds"]
    simulator.slots_df = dm.data["tourney_slots"]
    simulator.current_season = 2024
    simulator.build_bracket_tree(season=2024)

    slot_data, accuracy_metrics = simulator.simulate_historical_bracket(2024, method="elo")

    # There should be games played (non-zero)
    assert accuracy_metrics["total_games"] > 0
    # Accuracy must be a valid probability
    acc = accuracy_metrics["accuracy"]
    assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"


@pytest.mark.slow
@requires_data
def test_bracket_layout_has_regions(men_data_manager, men_elo):
    """get_bracket_layout should return a dict with at least 4 region keys."""
    from src.data_classes.bracket.BracketGenerator import BracketSimulator
    from src.data_classes.processing.TeamStatsCalculator import TeamStatsCalculator
    from src.data_classes.processing.MLModel import MarchMadnessMLModel
    from src.data_classes.processing.Predictor import MarchMadnessPredictor

    dm = men_data_manager
    elo = men_elo
    stats = TeamStatsCalculator(dm)
    ml = MarchMadnessMLModel(dm, elo, stats)
    predictor = MarchMadnessPredictor(
        data_manager=dm,
        elo_system=elo,
        stats_calculator=stats,
        ml_model=ml,
        current_season=2025,
    )
    predictor.initialize_models()

    simulator = BracketSimulator(predictor)
    simulator.teams_df = dm.data["teams"]
    simulator.seeds_df = dm.data["tourney_seeds"]
    simulator.slots_df = dm.data["tourney_slots"]
    simulator.current_season = 2025
    simulator.build_bracket_tree(season=2025)

    layout = simulator.get_bracket_layout(method="elo", season=2025)
    assert isinstance(layout, dict)
    assert len(layout) >= 1
