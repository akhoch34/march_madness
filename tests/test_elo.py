"""Tests for EloRatingSystem."""

import pytest
from conftest import requires_data


@requires_data
def test_win_prob_symmetry(men_elo):
    """P(A beats B) + P(B beats A) == 1.0 for any two teams."""
    elo = men_elo
    dm = elo.data_manager
    teams = dm.data["teams"]["TeamID"].values
    t1, t2 = teams[0], teams[1]
    season = dm.current_season
    p_ab = elo.predict_game(t1, t2, day_num=132, season=season)
    p_ba = elo.predict_game(t2, t1, day_num=132, season=season)
    assert abs(p_ab + p_ba - 1.0) < 1e-9


@requires_data
def test_win_prob_self(men_elo):
    """P(A beats A) == 0.5."""
    elo = men_elo
    teams = elo.data_manager.data["teams"]["TeamID"].values
    t = teams[0]
    season = elo.data_manager.current_season
    p = elo.predict_game(t, t, day_num=132, season=season)
    assert abs(p - 0.5) < 1e-9


@requires_data
def test_higher_elo_higher_win_prob(men_elo):
    """The team with the higher ELO rating should have > 50 % win probability."""
    elo = men_elo
    season = elo.data_manager.current_season
    tourney_seeds = elo.data_manager.data["tourney_seeds"]
    season_seeds = tourney_seeds[tourney_seeds["Season"] == season]
    if len(season_seeds) < 2:
        pytest.skip("Not enough tournament teams for this season")

    seed_df = season_seeds.copy()
    seed_df["SeedNum"] = seed_df["Seed"].str[1:3].astype(int)
    # Pick seed 1 vs seed 16 — seed 1 should always have higher ELO
    s1 = seed_df[seed_df["SeedNum"] == 1]
    s16 = seed_df[seed_df["SeedNum"] == 16]
    if len(s1) == 0 or len(s16) == 0:
        pytest.skip("Missing seed 1 or seed 16 for this season")

    t1 = s1["TeamID"].iloc[0]
    t16 = s16["TeamID"].iloc[0]
    p = elo.predict_game(t1, t16, day_num=132, season=season)
    assert p > 0.5, f"Seed-1 team should beat seed-16 team (got {p:.3f})"


@requires_data
def test_elo_ratings_populated(men_elo):
    """After calculate_elo_ratings, the rating dict must be non-empty."""
    assert len(men_elo.team_elo_ratings) > 0
