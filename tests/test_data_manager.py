"""Tests for MarchMadnessDataManager."""

import pytest
from conftest import requires_data


@requires_data
def test_data_loads(men_data_manager):
    dm = men_data_manager
    assert "teams" in dm.data
    assert "regular_season" in dm.data
    assert "tourney_seeds" in dm.data
    assert len(dm.data["teams"]) > 0


@requires_data
def test_seed_parsing_standard(men_data_manager):
    """Seeds like 'W01a' → numeric 1, 'X16b' → numeric 16."""
    seeds_df = men_data_manager.data["tourney_seeds"]
    # Parse seed number from seed string (characters 1-2 after region letter)
    seed_nums = seeds_df["Seed"].str[1:3].astype(int)
    assert seed_nums.min() >= 1
    assert seed_nums.max() <= 16


@requires_data
def test_get_team_name_returns_string(men_data_manager):
    teams = men_data_manager.data["teams"]
    any_team_id = teams["TeamID"].iloc[0]
    name = men_data_manager.get_team_name(any_team_id)
    assert isinstance(name, str)
    assert len(name) > 0


@requires_data
def test_get_team_name_unknown_id(men_data_manager):
    result = men_data_manager.get_team_name(99999)
    assert "99999" in result or result == "Team 99999"


@requires_data
def test_seed_lookup_roundtrip(men_data_manager):
    """seed_lookup stores the numeric seed (int), keyed by (season, team_id)."""
    seeds_df = men_data_manager.data["tourney_seeds"]
    row = seeds_df.iloc[0]
    season, team_id, seed_str = row["Season"], row["TeamID"], row["Seed"]
    seed_num = int(seed_str[1:3])
    assert men_data_manager.seed_lookup.get((season, team_id)) == seed_num


@requires_data
def test_women_data_loads(data_dir):
    """Women's DataManager should load without error from data/2026."""
    from src.data_classes.processing.DataManager import MarchMadnessDataManager
    dm = MarchMadnessDataManager(data_dir, gender="W", current_season=2025)
    dm.load_data()
    assert len(dm.data["teams"]) > 0
    assert len(dm.data["tourney_seeds"]) > 0
