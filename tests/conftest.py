"""
Shared fixtures for March Madness tests.
All tests that need real data use data/2026 (cumulative, has results through 2025).
"""

import os
import sys

import pytest

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "2026")


def _data_dir_available():
    return os.path.isdir(DATA_DIR) and os.path.isfile(
        os.path.join(DATA_DIR, "MNCAATourneySeeds.csv")
    )


# Skip all data-dependent tests when data/2026 is absent (CI without data)
requires_data = pytest.mark.skipif(
    not _data_dir_available(),
    reason="data/2026 not present — skipping data-dependent test",
)


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR


@pytest.fixture(scope="session")
def men_data_manager():
    """Session-scoped MarchMadnessDataManager for men's data (2026 dir, season 2025)."""
    if not _data_dir_available():
        pytest.skip("data/2026 not available")
    from src.data_classes.processing.DataManager import MarchMadnessDataManager
    dm = MarchMadnessDataManager(DATA_DIR, gender="M", current_season=2025)
    dm.load_data()
    return dm


@pytest.fixture(scope="session")
def men_elo(men_data_manager):
    """Pre-computed ELO ratings for the men's data manager."""
    from src.data_classes.processing.EloRatingSystem import EloRatingSystem
    elo = EloRatingSystem(men_data_manager)
    elo.calculate_elo_ratings(start_year=2010, k_factor=30, carry_over_factor=0.75)
    return elo
