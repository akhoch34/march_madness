"""
Bradley-Terry pairwise ranking model for March Madness predictions.

Unlike ELO (sequential updates), Bradley-Terry estimates a strength parameter
for every team simultaneously across all games using MLE. Often outperforms
ELO because it sees the whole season at once.

P(team_i beats team_j) = 1 / (1 + exp(-(beta_i - beta_j)))
"""

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.linear_model import LogisticRegression

from .DataManager import MarchMadnessDataManager


class BradleyTerryModel:
    """
    Bradley-Terry pairwise ranking model.

    Fits a strength parameter per team using logistic regression with
    team-indicator features. Strength difference maps directly to win probability
    via the logistic (sigmoid) function.
    """

    def __init__(self, data_manager: MarchMadnessDataManager):
        self.data_manager = data_manager
        # team_strengths[season][team_id] = beta coefficient
        self.team_strengths: dict[int, dict[int, float]] = {}
        self._team_index: dict[int, dict[int, int]] = {}  # season -> {team_id -> column_idx}
        self._fitted_seasons: set[int] = set()

    def fit(
        self,
        seasons: list[int] | None = None,
        min_season: int = 2003,
        regularization: float = 1.0,
    ) -> None:
        """
        Fit Bradley-Terry strength parameters for each season.

        For each season, trains on all regular-season games from min_season up
        to and including that season. This gives more training data for later
        seasons while maintaining temporal ordering.

        Parameters
        ----------
        seasons : list[int], optional
            Seasons to fit. Defaults to all seasons in regular season data.
        min_season : int
            Earliest season to include in training data.
        regularization : float
            Inverse regularization strength for logistic regression (higher = less regularization).
        """
        all_games = self.data_manager.data["regular_season"].copy()
        all_games = all_games[all_games["Season"] >= min_season]

        available_seasons = sorted(all_games["Season"].unique())
        if seasons is None:
            seasons = available_seasons

        # Also add tournament games for strength estimation
        tourney = self.data_manager.data["tourney_results"].copy()
        tourney = tourney[tourney["Season"] >= min_season]
        all_games_with_tourney = pd.concat([all_games, tourney], ignore_index=True)

        for target_season in seasons:
            # Train on all games up to and including this season
            train_games = all_games_with_tourney[
                all_games_with_tourney["Season"] <= target_season
            ]

            if len(train_games) == 0:
                continue

            # Build team index for this season
            all_team_ids = sorted(
                set(train_games["WTeamID"].unique()) | set(train_games["LTeamID"].unique())
            )
            team_idx = {tid: i for i, tid in enumerate(all_team_ids)}
            self._team_index[target_season] = team_idx
            n_teams = len(team_idx)

            # Build sparse feature matrix: each row = one game
            # +1 for winner column, -1 for loser column
            n_games = len(train_games)
            X = lil_matrix((n_games * 2, n_teams), dtype=np.float32)
            y = np.zeros(n_games * 2, dtype=np.float32)

            for row_idx, (_, game) in enumerate(train_games.iterrows()):
                w_col = team_idx[game["WTeamID"]]
                l_col = team_idx[game["LTeamID"]]
                # Winner row
                X[row_idx * 2, w_col] = 1.0
                X[row_idx * 2, l_col] = -1.0
                y[row_idx * 2] = 1.0
                # Loser row (augmented symmetric pair)
                X[row_idx * 2 + 1, w_col] = -1.0
                X[row_idx * 2 + 1, l_col] = 1.0
                y[row_idx * 2 + 1] = 0.0

            X_csr = csr_matrix(X)

            # Fit logistic regression with no intercept (intercept would cancel anyway)
            clf = LogisticRegression(
                fit_intercept=False,
                C=regularization,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )
            clf.fit(X_csr, y)

            # Store team strength coefficients
            betas = clf.coef_[0]
            self.team_strengths[target_season] = {
                tid: float(betas[i]) for tid, i in team_idx.items()
            }
            self._fitted_seasons.add(target_season)

        print(f"Bradley-Terry fitted for {len(self._fitted_seasons)} seasons.")

    def get_team_strength(self, team_id: int, season: int) -> float:
        """Return the Bradley-Terry strength coefficient for a team in a season."""
        if season not in self.team_strengths:
            # Fall back to most recent fitted season
            fitted = [s for s in self._fitted_seasons if s <= season]
            if not fitted:
                return 0.0
            season = max(fitted)
        return self.team_strengths[season].get(team_id, 0.0)

    def predict_game(self, team1_id: int, team2_id: int, season: int) -> float:
        """
        Predict P(team1 wins) using Bradley-Terry strength parameters.

        Parameters
        ----------
        team1_id, team2_id : int
        season : int

        Returns
        -------
        float : Win probability for team1.
        """
        beta1 = self.get_team_strength(team1_id, season)
        beta2 = self.get_team_strength(team2_id, season)
        return float(1.0 / (1.0 + np.exp(-(beta1 - beta2))))

    def get_strength_features(
        self, team1_id: int, team2_id: int, season: int
    ) -> dict:
        """Return Bradley-Terry strength features for a matchup."""
        beta1 = self.get_team_strength(team1_id, season)
        beta2 = self.get_team_strength(team2_id, season)
        return {
            "Team1_BTStrength": beta1,
            "Team2_BTStrength": beta2,
            "BTStrength_Diff": beta1 - beta2,
            "BTWinProb": float(1.0 / (1.0 + np.exp(-(beta1 - beta2)))),
        }
