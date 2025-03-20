from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .DataManager import MarchMadnessDataManager
from .EloRatingSystem import EloRatingSystem
from .TeamStatsCalculator import TeamStatsCalculator


class MarchMadnessMLModel:
    def __init__(
        self,
        data_manager: MarchMadnessDataManager,
        elo_system: EloRatingSystem,
        stats_calculator: TeamStatsCalculator,
    ):
        self.data_manager = data_manager
        self.elo_system = elo_system
        self.stats_calculator = stats_calculator
        self.model = None
        self.feature_df = None
        self.feature_columns = None
        self.exclude_columns = [
            "Result",
            "Season",
            "Team1ID",
            "Team2ID",
            "ELOWinProb",
            "ELO_residual",
        ]
        self._feature_dataset_created = False

    def create_feature_dataset(
        self,
        train_years_range=(2010, 2024),
        include_elo=True,
        include_advanced_stats=True,
    ):
        """
        Create a dataset with features for training and prediction.

        Parameters:
        train_years_range (tuple): Range of years to use for training (inclusive)
        include_elo (bool): Whether to include ELO rating features
        include_advanced_stats (bool): Whether to include advanced box score stats
        """
        # Prevent infinite recursion
        if self._feature_dataset_created:
            print("Feature dataset already created, skipping")
            return self.feature_df

        print("Creating feature dataset...")

        # Process seeds first
        self.data_manager.preprocess_seeds()

        # Calculate ELO ratings if needed
        if include_elo and (
            not hasattr(self.elo_system, "team_elo_ratings")
            or not self.elo_system.team_elo_ratings
        ):
            self.elo_system.calculate_elo_ratings(
                start_year=min(train_years_range[0] - 2, 2003)
            )

        # Calculate advanced stats if needed
        advanced_stats_calculated = (
            hasattr(self.stats_calculator, "advanced_team_stats")
            and self.stats_calculator.advanced_team_stats
        )

        if include_advanced_stats and not advanced_stats_calculated:
            self.stats_calculator.calculate_advanced_team_stats(
                start_season=min(train_years_range[0], 2003)
            )

        # Get all tournament matchups from historical data
        tourney_games = self.data_manager.data["tourney_results"].copy()

        # Create features for each historical matchup
        features = []

        for _, game in tourney_games.iterrows():
            season = game["Season"]

            # Skip if outside our training range
            if season < train_years_range[0] or season > train_years_range[1]:
                continue

            team1_id = game["WTeamID"]  # Winner
            team2_id = game["LTeamID"]  # Loser
            day_num = game["DayNum"]

            # Get seed information
            team1_seed = self.data_manager.seed_lookup.get(
                (season, team1_id), 16
            )  # Default to 16 if not found
            team2_seed = self.data_manager.seed_lookup.get((season, team2_id), 16)

            # Basic features
            game_features = {
                "Season": season,
                "Team1ID": team1_id,
                "Team2ID": team2_id,
                "Team1Seed": team1_seed,
                "Team2Seed": team2_seed,
                "SeedDiff": team2_seed - team1_seed,
                "Result": 1,  # Team1 won
            }

            # Add season performance metrics
            game_features.update(self._get_season_stats(season, team1_id, team2_id))

            # Add ranking features if available
            if self.data_manager.rankings_available:
                game_features.update(
                    self._get_ranking_features(season, team1_id, team2_id)
                )

            # Add ELO rating features and residual
            if include_elo:
                # Get ELO ratings just before this tournament game
                # We use day_num - 1 to ensure we don't leak future information
                team1_elo = self.elo_system.get_team_elo(season, team1_id, day_num - 1)
                team2_elo = self.elo_system.get_team_elo(season, team2_id, day_num - 1)

                # Calculate win probability
                elo_win_prob = self.elo_system.elo_win_probability(team1_elo, team2_elo)

                # Calculate ELO residual (actual - predicted)
                elo_residual = 1 - elo_win_prob  # Team1 won, so Result=1

                game_features.update(
                    {
                        "Team1ELO": team1_elo,
                        "Team2ELO": team2_elo,
                        "ELODiff": team1_elo - team2_elo,
                        "ELOWinProb": elo_win_prob,
                        "ELO_residual": elo_residual,
                    }
                )

            # Add advanced stats features if available
            if include_advanced_stats and (
                hasattr(self.stats_calculator, "advanced_team_stats")
                and self.stats_calculator.advanced_team_stats
            ):
                game_features.update(
                    self._get_advanced_stats_features(season, team1_id, team2_id)
                )

            features.append(game_features)

            # Also add the reversed matchup (with opposite result)
            reversed_features = self._create_reversed_features(game_features)
            features.append(reversed_features)

        # Create DataFrame with all features
        self.feature_df = pd.DataFrame(features)
        print(f"Created feature dataset with {len(self.feature_df)} samples")

        # Define feature columns (excluding outcome and identifiers)
        self.feature_columns = [
            col for col in self.feature_df.columns if col not in self.exclude_columns
        ]

        # Set flag to avoid infinite recursion
        self._feature_dataset_created = True
        self.feature_df.to_csv(
            f"output/{self.data_manager.gender}_feature_dataset.csv", index=False
        )

        return self.feature_df

    def _create_reversed_features(self, game_features):
        """Create reversed features (swap team1 and team2)"""
        reversed_features = game_features.copy()

        team1_id = game_features["Team1ID"]
        team2_id = game_features["Team2ID"]
        team1_seed = game_features["Team1Seed"]
        team2_seed = game_features["Team2Seed"]

        reversed_features["Team1ID"] = team2_id
        reversed_features["Team2ID"] = team1_id
        reversed_features["Team1Seed"] = team2_seed
        reversed_features["Team2Seed"] = team1_seed
        reversed_features["SeedDiff"] = team1_seed - team2_seed
        reversed_features["Result"] = 0  # Team1 lost

        # Reverse any asymmetric stat features
        if "Team1WinPct" in reversed_features:
            reversed_features["Team1WinPct"] = game_features["Team2WinPct"]
            reversed_features["Team2WinPct"] = game_features["Team1WinPct"]
            reversed_features["WinPctDiff"] = -game_features["WinPctDiff"]

        # Reverse strength of schedule features if present
        if "Team1SOS" in reversed_features:
            reversed_features["Team1SOS"] = game_features["Team2SOS"]
            reversed_features["Team2SOS"] = game_features["Team1SOS"]
            reversed_features["SOSDiff"] = -game_features["SOSDiff"]

        # Reverse last 10 features if present
        if "Team1Last10" in reversed_features:
            reversed_features["Team1Last10"] = game_features["Team2Last10"]
            reversed_features["Team2Last10"] = game_features["Team1Last10"]
            reversed_features["Last10Diff"] = -game_features["Last10Diff"]

        # Reverse exponentially weighted win percentage features
        if "Team1ExpWinPct" in reversed_features:
            reversed_features["Team1ExpWinPct"] = game_features["Team2ExpWinPct"]
            reversed_features["Team2ExpWinPct"] = game_features["Team1ExpWinPct"]
            reversed_features["ExpWinPctDiff"] = -game_features["ExpWinPctDiff"]

        # Reverse momentum features
        if "Team1Momentum" in reversed_features:
            reversed_features["Team1Momentum"] = game_features["Team2Momentum"]
            reversed_features["Team2Momentum"] = game_features["Team1Momentum"]
            reversed_features["MomentumDiff"] = -game_features["MomentumDiff"]

        # Reverse scoring trend features
        if "Team1ScoringTrend" in reversed_features:
            reversed_features["Team1ScoringTrend"] = game_features["Team2ScoringTrend"]
            reversed_features["Team2ScoringTrend"] = game_features["Team1ScoringTrend"]
            reversed_features["ScoringTrendDiff"] = -game_features["ScoringTrendDiff"]

        # Reverse recent margin features
        if "Team1RecentMargin" in reversed_features:
            reversed_features["Team1RecentMargin"] = game_features["Team2RecentMargin"]
            reversed_features["Team2RecentMargin"] = game_features["Team1RecentMargin"]
            reversed_features["RecentMarginDiff"] = -game_features["RecentMarginDiff"]

        # Reverse streak features
        if "Team1Streak" in reversed_features:
            reversed_features["Team1Streak"] = game_features["Team2Streak"]
            reversed_features["Team2Streak"] = game_features["Team1Streak"]
            reversed_features["StreakDiff"] = -game_features["StreakDiff"]

        # Reverse conference tournament features
        if "Team1ConfWinPct" in reversed_features:
            reversed_features["Team1ConfWinPct"] = game_features["Team2ConfWinPct"]
            reversed_features["Team2ConfWinPct"] = game_features["Team1ConfWinPct"]
            reversed_features["ConfWinPctDiff"] = -game_features["ConfWinPctDiff"]

        if "Team1ConfDepth" in reversed_features:
            reversed_features["Team1ConfDepth"] = game_features["Team2ConfDepth"]
            reversed_features["Team2ConfDepth"] = game_features["Team1ConfDepth"]
            reversed_features["ConfDepthDiff"] = -game_features["ConfDepthDiff"]

        # Reverse late season features
        if "Team1LateWinPct" in reversed_features:
            reversed_features["Team1LateWinPct"] = game_features["Team2LateWinPct"]
            reversed_features["Team2LateWinPct"] = game_features["Team1LateWinPct"]
            reversed_features["LateWinPctDiff"] = -game_features["LateWinPctDiff"]

        # Reverse ELO features if present
        if "Team1ELO" in reversed_features:
            reversed_features["Team1ELO"] = game_features["Team2ELO"]
            reversed_features["Team2ELO"] = game_features["Team1ELO"]
            reversed_features["ELODiff"] = -game_features["ELODiff"]
            reversed_features["ELOWinProb"] = 1.0 - game_features["ELOWinProb"]
            if "ELO_residual" in game_features:
                # Reverse residual (actual - predicted) for losing team
                # Team2 lost, so actual=0
                reversed_features["ELO_residual"] = 0 - (
                    1.0 - game_features["ELOWinProb"]
                )

        # Reverse advanced stats features if present
        for key in list(reversed_features.keys()):
            # Look for keys with Team1_ prefix that need to be swapped
            if (
                key.startswith("Team1_")
                and key.replace("Team1_", "Team2_") in reversed_features
            ):
                team1_key = key
                team2_key = key.replace("Team1_", "Team2_")
                reversed_features[team1_key] = game_features[team2_key]
                reversed_features[team2_key] = game_features[team1_key]

            # Flip the sign of all difference features
            if key.endswith("_Diff") and key not in [
                "SeedDiff",
                "ELODiff",
                "WinPctDiff",
                "SOSDiff",
                "Last10Diff",
                "ExpWinPctDiff",
                "MomentumDiff",
                "ScoringTrendDiff",
                "RecentMarginDiff",
                "StreakDiff",
                "ConfWinPctDiff",
                "ConfDepthDiff",
                "LateWinPctDiff",
            ]:
                reversed_features[key] = -game_features[key]

        return reversed_features

    def generate_features_for_matchup(self, team1_id, team2_id, season, day_num=132):
        """Generate features for a new matchup"""
        # Get seed information if available
        team1_seed = self.data_manager.seed_lookup.get((season, team1_id), 16)
        team2_seed = self.data_manager.seed_lookup.get((season, team2_id), 16)

        # Basic features
        game_features = {
            "Season": season,
            "Team1ID": team1_id,
            "Team2ID": team2_id,
            "Team1Seed": team1_seed,
            "Team2Seed": team2_seed,
            "SeedDiff": team2_seed - team1_seed,
        }

        # Add season performance metrics
        game_features.update(self._get_season_stats(season, team1_id, team2_id))

        # Add ranking features if available
        if self.data_manager.rankings_available:
            game_features.update(self._get_ranking_features(season, team1_id, team2_id))

        # Add ELO rating features
        if (
            hasattr(self.elo_system, "team_elo_ratings")
            and self.elo_system.team_elo_ratings
        ):
            # Get ELO ratings for tournament
            team1_elo = self.elo_system.get_team_elo(season, team1_id, day_num)
            team2_elo = self.elo_system.get_team_elo(season, team2_id, day_num)

            # Calculate win probability
            elo_win_prob = self.elo_system.elo_win_probability(team1_elo, team2_elo)

            game_features.update(
                {
                    "Team1ELO": team1_elo,
                    "Team2ELO": team2_elo,
                    "ELODiff": team1_elo - team2_elo,
                    "ELOWinProb": elo_win_prob,
                }
            )

        # Add advanced stats features if available
        if (
            hasattr(self.stats_calculator, "advanced_team_stats")
            and self.stats_calculator.advanced_team_stats
        ):
            game_features.update(
                self._get_advanced_stats_features(season, team1_id, team2_id)
            )

        return game_features

    def get_matchup_features(
        self, team1_id, team2_id, season, day_num=132
    ) -> pd.DataFrame:
        """Get features for a specific matchup, generating them if needed"""
        # Ensure feature dataset exists
        if self.feature_df is None:
            self.create_feature_dataset()

        # Check if this matchup exists in our feature dataset
        existing_features = self.feature_df[
            (self.feature_df["Team1ID"] == team1_id)
            & (self.feature_df["Team2ID"] == team2_id)
            & (self.feature_df["Season"] == season)
        ]

        # If we have existing features, return them
        if len(existing_features) > 0:
            return existing_features

        # Otherwise, generate new features for this matchup
        new_features = self.generate_features_for_matchup(
            team1_id, team2_id, season, day_num
        )

        # Create a single row DataFrame with these features
        df_new = pd.DataFrame([new_features])

        # Make sure all required columns are present
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in df_new.columns:
                    # Add missing column with default value 0
                    df_new[col] = 0

        return df_new

    def predict(self, team1_id, team2_id, season, day_num=132):
        """
        Make a prediction for a specific matchup using the ELO-enhanced model.
        First gets base ELO prediction, then applies ML correction if available.
        """
        # Get base ELO prediction
        elo_pred = self.elo_system.predict_game(team1_id, team2_id, day_num, season)

        # If we don't have a trained ML model, just return the ELO prediction
        if self.model is None:
            return elo_pred

        try:
            # Get features for this matchup
            matchup_features = self.get_matchup_features(
                team1_id, team2_id, season, day_num
            )

            if len(matchup_features) == 0:
                print(
                    f"Warning: No features found for {team1_id} vs {team2_id}. Using ELO fallback."
                )
                return elo_pred

            # Extract just the feature columns we need
            features = matchup_features[self.feature_columns]

            # Make ML prediction (which predicts residual)
            ml_correction = self.model.predict(features)[0]

            # Apply correction to ELO prediction
            final_pred = np.clip(elo_pred + ml_correction, 0.01, 0.999)

            return final_pred

        except Exception as e:
            print(f"Error making prediction: {e}")
            # Fallback to ELO if ML prediction fails
            return elo_pred

    def train_model(self, model_type="xgboost", test_size=0.2, random_state=42):
        """
        Train a model to predict residuals (corrections) to ELO predictions
        """
        print("Training ELO-enhanced ML model...")

        # Create or ensure feature dataset exists
        if self.feature_df is None or not self._feature_dataset_created:
            self.create_feature_dataset()

        if "ELO_residual" not in self.feature_df.columns:
            print("Error: ELO_residual column not found in feature dataset.")
            return None

        # Prepare features and target (residual)
        X = self.feature_df[self.feature_columns]
        y = self.feature_df["ELO_residual"]  # Target is ELO residual

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Get baseline ELO predictions for evaluation
        elo_train = self.feature_df.loc[y_train.index, "ELOWinProb"]
        elo_test = self.feature_df.loc[y_test.index, "ELOWinProb"]
        y_actual_train = self.feature_df.loc[y_train.index, "Result"]
        y_actual_test = self.feature_df.loc[y_test.index, "Result"]

        # Initialize model
        if model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=50,
                learning_rate=0.03,
                max_depth=2,
                min_child_weight=3,
                subsample=0.7,
                colsample_bytree=0.7,
                objective="reg:squarederror",
                random_state=random_state,
            )

        # Train model on residuals
        self.model.fit(X_train, y_train)

        # Make predictions
        train_residuals = self.model.predict(X_train)
        test_residuals = self.model.predict(X_test)

        # Apply corrections to get final predictions
        train_preds = np.clip(elo_train + train_residuals, 0.01, 0.999)
        test_preds = np.clip(elo_test + test_residuals, 0.01, 0.999)

        # Evaluate baseline ELO performance
        elo_train_acc = accuracy_score(y_actual_train, elo_train > 0.5)
        elo_test_acc = accuracy_score(y_actual_test, elo_test > 0.5)
        elo_train_brier = brier_score_loss(y_actual_train, elo_train)
        elo_test_brier = brier_score_loss(y_actual_test, elo_test)
        elo_train_log_loss_val = log_loss(y_actual_train, elo_train)
        elo_test_log_loss_val = log_loss(y_actual_test, elo_test)

        # Evaluate combined model performance
        train_acc = accuracy_score(y_actual_train, train_preds > 0.5)
        test_acc = accuracy_score(y_actual_test, test_preds > 0.5)
        train_brier = brier_score_loss(y_actual_train, train_preds)
        test_brier = brier_score_loss(y_actual_test, test_preds)
        train_log_loss_val = log_loss(y_actual_train, train_preds)
        test_log_loss_val = log_loss(y_actual_test, test_preds)

        # Calculate MSE on residuals
        train_mse = mean_squared_error(y_train, train_residuals)
        test_mse = mean_squared_error(y_test, test_residuals)

        # Print comparison results
        print("\nPerformance Comparison - ELO vs ELO-Enhanced ML:")
        print(
            f"{'Metric':<20} {'ELO Train':<12} {'ELO Test':<12} {'Enhanced Train':<12} {'Enhanced Test':<12} {'Improvement':<12}"
        )
        print("-" * 80)
        print(
            f"{'Accuracy':<20} {elo_train_acc:.4f}{'':<8} {elo_test_acc:.4f}{'':<8} {train_acc:.4f}{'':<8} {test_acc:.4f}{'':<8} {test_acc - elo_test_acc:.4f}"
        )
        print(
            f"{'Brier Score':<20} {elo_train_brier:.4f}{'':<8} {elo_test_brier:.4f}{'':<8} {train_brier:.4f}{'':<8} {test_brier:.4f}{'':<8} {elo_test_brier - test_brier:.4f}"
        )
        print(
            f"{'Log Loss':<20} {elo_train_log_loss_val:.4f}{'':<8} {elo_test_log_loss_val:.4f}{'':<8} {train_log_loss_val:.4f}{'':<8} {test_log_loss_val:.4f}{'':<8} {elo_test_log_loss_val - test_log_loss_val:.4f}"
        )
        print(
            f"{'MSE on Residuals':<20} {'N/A':<12} {'N/A':<12} {train_mse:.4f}{'':<8} {test_mse:.4f}"
        )

        # Feature importance
        if hasattr(self.model, "feature_importances_") and self.feature_columns:
            self._display_feature_importance()

        return self.model

    def _display_feature_importance(self):
        """Display feature importance from the model"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Only show top 20 features for clarity
        top_n = min(20, len(self.feature_columns))

        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance (Top 20)")
        plt.bar(range(top_n), importances[indices][:top_n], align="center")
        plt.xticks(
            range(top_n),
            [self.feature_columns[i] for i in indices][:top_n],
            rotation=90,
        )
        plt.tight_layout()
        plt.show()

    def _get_season_stats(self, season, team1_id, team2_id):
        """Get season performance stats for both teams with enhanced recency metrics"""
        # Filter regular season games for this season
        season_games = self.data_manager.data["regular_season"][
            self.data_manager.data["regular_season"]["Season"] == season
        ]

        # Also get tournament games (will include conference tournaments)
        tournament_games = self.data_manager.data["tourney_results"][
            self.data_manager.data["tourney_results"]["Season"] == season
        ]

        # Combine regular season and tournament games
        all_games = pd.concat([season_games, tournament_games])

        # --- TRADITIONAL SEASON-LONG METRICS (KEEPING EXISTING CODE) ---
        # Team1 stats
        team1_wins = season_games[season_games["WTeamID"] == team1_id].shape[0]
        team1_losses = season_games[season_games["LTeamID"] == team1_id].shape[0]
        team1_win_pct = (
            team1_wins / (team1_wins + team1_losses)
            if (team1_wins + team1_losses) > 0
            else 0
        )

        # Team2 stats
        team2_wins = season_games[season_games["WTeamID"] == team2_id].shape[0]
        team2_losses = season_games[season_games["LTeamID"] == team2_id].shape[0]
        team2_win_pct = (
            team2_wins / (team2_wins + team2_losses)
            if (team2_wins + team2_losses) > 0
            else 0
        )

        # --- ENHANCED RECENCY FEATURES ---

        # Get all games for each team sorted by day
        team1_games = pd.concat(
            [
                all_games[all_games["WTeamID"] == team1_id].assign(
                    Result=1,
                    ScoreMargin=all_games["WScore"] - all_games["LScore"],
                    isWin=1,
                ),
                all_games[all_games["LTeamID"] == team1_id].assign(
                    Result=0,
                    ScoreMargin=all_games["LScore"] - all_games["WScore"],
                    isWin=0,
                ),
            ]
        ).sort_values(
            "DayNum", ascending=True
        )  # Sorting in ascending order for time series

        team2_games = pd.concat(
            [
                all_games[all_games["WTeamID"] == team2_id].assign(
                    Result=1,
                    ScoreMargin=all_games["WScore"] - all_games["LScore"],
                    isWin=1,
                ),
                all_games[all_games["LTeamID"] == team2_id].assign(
                    Result=0,
                    ScoreMargin=all_games["LScore"] - all_games["WScore"],
                    isWin=0,
                ),
            ]
        ).sort_values(
            "DayNum", ascending=True
        )  # Sorting in ascending order for time series

        # 1. Better Last10 features (basic version already exists)
        team1_last_10 = team1_games.tail(10) if len(team1_games) > 0 else pd.DataFrame()
        team2_last_10 = team2_games.tail(10) if len(team2_games) > 0 else pd.DataFrame()

        team1_last_10_win_pct = (
            team1_last_10["Result"].mean() if len(team1_last_10) > 0 else team1_win_pct
        )
        team2_last_10_win_pct = (
            team2_last_10["Result"].mean() if len(team2_last_10) > 0 else team2_win_pct
        )

        # 2. Exponentially weighted recent win percentage (more weight to recent games)
        # Get the last 10 games with exponential weights (most recent games weighted more)
        def get_exp_weighted_win_pct(team_games, window=10, halflife=3):
            """Calculate exponentially weighted win percentage for recent games"""
            if len(team_games) == 0:
                return 0.0

            # Take last n games
            recent_games = team_games.tail(window)
            if len(recent_games) == 0:
                return 0.0

            # Apply exponential weights
            weights = np.exp(np.arange(len(recent_games)) / halflife)
            weights = weights / weights.sum()  # Normalize weights

            # Calculate weighted average
            weighted_win_pct = (recent_games["Result"] * weights).sum()
            return weighted_win_pct

        team1_exp_win_pct = get_exp_weighted_win_pct(team1_games)
        team2_exp_win_pct = get_exp_weighted_win_pct(team2_games)

        # 3. Last 5 vs Previous 5 (momentum indicator)
        team1_last_5_win_pct = (
            team1_games.tail(5)["Result"].mean()
            if len(team1_games) >= 5
            else team1_win_pct
        )
        team1_prev_5_win_pct = (
            team1_games.iloc[-10:-5]["Result"].mean()
            if len(team1_games) >= 10
            else team1_win_pct
        )
        team1_momentum = (
            team1_last_5_win_pct - team1_prev_5_win_pct
        )  # Positive = improving, Negative = declining

        team2_last_5_win_pct = (
            team2_games.tail(5)["Result"].mean()
            if len(team2_games) >= 5
            else team2_win_pct
        )
        team2_prev_5_win_pct = (
            team2_games.iloc[-10:-5]["Result"].mean()
            if len(team2_games) >= 10
            else team2_win_pct
        )
        team2_momentum = team2_last_5_win_pct - team2_prev_5_win_pct

        # 4. Scoring trend features (are they scoring more or less lately?)
        def get_scoring_trend(team_games, window=5):
            """Calculate scoring trend by comparing recent games to season average"""
            if len(team_games) < window:
                return 0.0

            # For wins, use WScore; for losses, use LScore
            recent_scores = []
            for _, game in team_games.tail(window).iterrows():
                if game["isWin"] == 1:
                    recent_scores.append(game["WScore"] if "WScore" in game else 0)
                else:
                    recent_scores.append(game["LScore"] if "LScore" in game else 0)

            recent_avg = np.mean(recent_scores) if recent_scores else 0

            all_scores = []
            for _, game in team_games.iterrows():
                if game["isWin"] == 1:
                    all_scores.append(game["WScore"] if "WScore" in game else 0)
                else:
                    all_scores.append(game["LScore"] if "LScore" in game else 0)

            season_avg = np.mean(all_scores) if all_scores else 0

            return recent_avg - season_avg  # Positive = scoring more lately

        team1_scoring_trend = get_scoring_trend(team1_games)
        team2_scoring_trend = get_scoring_trend(team2_games)

        # 5. Recent margin of victory
        team1_recent_margin = (
            team1_games.tail(5)["ScoreMargin"].mean() if len(team1_games) >= 5 else 0
        )
        team2_recent_margin = (
            team2_games.tail(5)["ScoreMargin"].mean() if len(team2_games) >= 5 else 0
        )

        # 6. Winning/losing streak
        def get_current_streak(team_games):
            """Calculate current winning or losing streak"""
            if len(team_games) == 0:
                return 0

            results = team_games["isWin"].values

            if len(results) == 0:
                return 0

            current_result = results[-1]
            streak = 0

            # Count consecutive same results from the end
            for i in range(len(results) - 1, -1, -1):
                if results[i] == current_result:
                    if current_result == 1:
                        streak += 1  # Winning streak (positive)
                    else:
                        streak -= 1  # Losing streak (negative)
                else:
                    break

            return streak

        team1_streak = get_current_streak(team1_games)
        team2_streak = get_current_streak(team2_games)

        # 7. Conference tournament specific features
        # Based on day_num, conference tournaments are typically days 118-132
        conf_tourney_start = 118

        # Extract conference tournament games
        team1_conf_games = team1_games[
            (team1_games["DayNum"] >= conf_tourney_start)
            & (team1_games["DayNum"] < 134)
        ]
        team2_conf_games = team2_games[
            (team2_games["DayNum"] >= conf_tourney_start)
            & (team2_games["DayNum"] < 134)
        ]

        team1_conf_win_pct = (
            team1_conf_games["Result"].mean()
            if len(team1_conf_games) > 0
            else team1_win_pct
        )
        team2_conf_win_pct = (
            team2_conf_games["Result"].mean()
            if len(team2_conf_games) > 0
            else team2_win_pct
        )

        # How deep did they go in conference tournament (approx by last day played)
        team1_conf_depth = (
            team1_conf_games["DayNum"].max() - conf_tourney_start
            if len(team1_conf_games) > 0
            else 0
        )
        team2_conf_depth = (
            team2_conf_games["DayNum"].max() - conf_tourney_start
            if len(team2_conf_games) > 0
            else 0
        )

        # 8. Late season performance (February onward, ~ day 70+)
        late_season_start = 70

        team1_late_games = team1_games[team1_games["DayNum"] >= late_season_start]
        team2_late_games = team2_games[team2_games["DayNum"] >= late_season_start]

        team1_late_win_pct = (
            team1_late_games["Result"].mean()
            if len(team1_late_games) > 0
            else team1_win_pct
        )
        team2_late_win_pct = (
            team2_late_games["Result"].mean()
            if len(team2_late_games) > 0
            else team2_win_pct
        )

        # Calculate strength of schedule (keeping from original code)
        if (
            hasattr(self.stats_calculator, "advanced_team_stats")
            and self.stats_calculator.advanced_team_stats
            and season in self.stats_calculator.advanced_team_stats
        ):
            # Get list of opponents and their net efficiency
            team1_opponents = []
            team2_opponents = []

            # Get opponents from wins
            for _, game in season_games[season_games["WTeamID"] == team1_id].iterrows():
                team1_opponents.append(game["LTeamID"])

            for _, game in season_games[season_games["WTeamID"] == team2_id].iterrows():
                team2_opponents.append(game["LTeamID"])

            # Get opponents from losses
            for _, game in season_games[season_games["LTeamID"] == team1_id].iterrows():
                team1_opponents.append(game["WTeamID"])

            for _, game in season_games[season_games["LTeamID"] == team2_id].iterrows():
                team2_opponents.append(game["WTeamID"])

            # Calculate average opponent net efficiency
            season_stats = self.stats_calculator.advanced_team_stats[season]

            team1_opp_net_eff = [
                season_stats.get(opp, {}).get("NetEff", 0) for opp in team1_opponents
            ]
            team2_opp_net_eff = [
                season_stats.get(opp, {}).get("NetEff", 0) for opp in team2_opponents
            ]

            team1_sos = np.mean(team1_opp_net_eff) if team1_opp_net_eff else 0
            team2_sos = np.mean(team2_opp_net_eff) if team2_opp_net_eff else 0

            # Build feature dictionary with original features plus new recency features
            return {
                # Original features
                "Team1WinPct": team1_win_pct,
                "Team2WinPct": team2_win_pct,
                "WinPctDiff": team1_win_pct - team2_win_pct,
                "Team1SOS": team1_sos,
                "Team2SOS": team2_sos,
                "SOSDiff": team1_sos - team2_sos,
                "Team1Last10": team1_last_10_win_pct,
                "Team2Last10": team2_last_10_win_pct,
                "Last10Diff": team1_last_10_win_pct - team2_last_10_win_pct,
                # New recency features
                "Team1ExpWinPct": team1_exp_win_pct,
                "Team2ExpWinPct": team2_exp_win_pct,
                "ExpWinPctDiff": team1_exp_win_pct - team2_exp_win_pct,
                "Team1Momentum": team1_momentum,
                "Team2Momentum": team2_momentum,
                "MomentumDiff": team1_momentum - team2_momentum,
                "Team1ScoringTrend": team1_scoring_trend,
                "Team2ScoringTrend": team2_scoring_trend,
                "ScoringTrendDiff": team1_scoring_trend - team2_scoring_trend,
                "Team1RecentMargin": team1_recent_margin,
                "Team2RecentMargin": team2_recent_margin,
                "RecentMarginDiff": team1_recent_margin - team2_recent_margin,
                "Team1Streak": team1_streak,
                "Team2Streak": team2_streak,
                "StreakDiff": team1_streak - team2_streak,
                "Team1ConfWinPct": team1_conf_win_pct,
                "Team2ConfWinPct": team2_conf_win_pct,
                "ConfWinPctDiff": team1_conf_win_pct - team2_conf_win_pct,
                "Team1ConfDepth": team1_conf_depth,
                "Team2ConfDepth": team2_conf_depth,
                "ConfDepthDiff": team1_conf_depth - team2_conf_depth,
                "Team1LateWinPct": team1_late_win_pct,
                "Team2LateWinPct": team2_late_win_pct,
                "LateWinPctDiff": team1_late_win_pct - team2_late_win_pct,
            }
        else:
            # Basic stats if advanced stats aren't available
            return {
                "Team1WinPct": team1_win_pct,
                "Team2WinPct": team2_win_pct,
                "WinPctDiff": team1_win_pct - team2_win_pct,
                # Include new recency features even without advanced stats
                "Team1Last10": team1_last_10_win_pct,
                "Team2Last10": team2_last_10_win_pct,
                "Last10Diff": team1_last_10_win_pct - team2_last_10_win_pct,
                "Team1ExpWinPct": team1_exp_win_pct,
                "Team2ExpWinPct": team2_exp_win_pct,
                "ExpWinPctDiff": team1_exp_win_pct - team2_exp_win_pct,
                "Team1Momentum": team1_momentum,
                "Team2Momentum": team2_momentum,
                "MomentumDiff": team1_momentum - team2_momentum,
                "Team1Streak": team1_streak,
                "Team2Streak": team2_streak,
                "StreakDiff": team1_streak - team2_streak,
                "Team1ConfWinPct": team1_conf_win_pct,
                "Team2ConfWinPct": team2_conf_win_pct,
                "ConfWinPctDiff": team1_conf_win_pct - team2_conf_win_pct,
            }

    def _get_ranking_features(self, season, team1_id, team2_id):
        """Get pre-tournament ranking features for both teams"""
        if not self.data_manager.rankings_available:
            return {}

        # Get final rankings before tournament (RankingDayNum = 133)
        rankings = self.data_manager.data["rankings"]
        pre_tourney_rankings = rankings[
            (rankings["Season"] == season) & (rankings["RankingDayNum"] == 133)
        ]

        # Aggregate rankings across systems (use mean)
        team_ranks = {}
        for _, row in pre_tourney_rankings.iterrows():
            team_id = row["TeamID"]
            if team_id not in team_ranks:
                team_ranks[team_id] = []
            team_ranks[team_id].append(row["OrdinalRank"])

        # Calculate average ranking for each team
        team1_avg_rank = (
            np.mean(team_ranks.get(team1_id, [353])) if team1_id in team_ranks else 353
        )
        team2_avg_rank = (
            np.mean(team_ranks.get(team2_id, [353])) if team2_id in team_ranks else 353
        )

        return {
            "Team1AvgRank": team1_avg_rank,
            "Team2AvgRank": team2_avg_rank,
            "RankDiff": team2_avg_rank
            - team1_avg_rank,  # Positive if team1 is ranked better
        }

    def _get_advanced_stats_features(self, season, team1_id, team2_id):
        """Get advanced stats features for both teams"""
        if (
            not hasattr(self.stats_calculator, "advanced_team_stats")
            or not self.stats_calculator.advanced_team_stats
        ):
            self.stats_calculator.calculate_advanced_team_stats()

        # Get stats for this season
        if season not in self.stats_calculator.advanced_team_stats:
            # If season not available, return empty dict
            return {}

        season_stats = self.stats_calculator.advanced_team_stats[season]

        # Get stats for both teams
        team1_stats = season_stats.get(team1_id, {})
        team2_stats = season_stats.get(team2_id, {})

        # Skip if either team doesn't have stats
        if not team1_stats or not team2_stats:
            return {}

        # Create features dictionary
        features = {}

        # Four Factors - the most predictive advanced metrics
        for factor in ["eFG%", "TOV%", "ORB%", "FTRate"]:
            # Offensive factors
            features[f"Team1_{factor}"] = team1_stats.get(factor, 0)
            features[f"Team2_{factor}"] = team2_stats.get(factor, 0)
            features[f"{factor}_Diff"] = team1_stats.get(factor, 0) - team2_stats.get(
                factor, 0
            )

            # Defensive factors (opponent's numbers)
            opp_factor = f"Opp{factor}"
            if opp_factor in team1_stats:
                features[f"Team1_Def_{factor}"] = team1_stats.get(opp_factor, 0)
                features[f"Team2_Def_{factor}"] = team2_stats.get(opp_factor, 0)
                features[f"Def_{factor}_Diff"] = team1_stats.get(
                    opp_factor, 0
                ) - team2_stats.get(opp_factor, 0)

        # Efficiency metrics
        for metric in ["OffEff", "DefEff", "NetEff"]:
            features[f"Team1_{metric}"] = team1_stats.get(metric, 0)
            features[f"Team2_{metric}"] = team2_stats.get(metric, 0)
            features[f"{metric}_Diff"] = team1_stats.get(metric, 0) - team2_stats.get(
                metric, 0
            )

        # Tempo/Pace
        features["Team1_Pace"] = team1_stats.get("Pace", 0)
        features["Team2_Pace"] = team2_stats.get("Pace", 0)
        features["Pace_Diff"] = team1_stats.get("Pace", 0) - team2_stats.get("Pace", 0)

        # Shooting percentages
        for pct in ["FG%", "3P%", "FT%"]:
            features[f"Team1_{pct}"] = team1_stats.get(pct, 0)
            features[f"Team2_{pct}"] = team2_stats.get(pct, 0)
            features[f"{pct}_Diff"] = team1_stats.get(pct, 0) - team2_stats.get(pct, 0)

            # Defensive (opponent shooting percentages)
            opp_pct = f"Opp{pct}"
            features[f"Team1_Def_{pct}"] = team1_stats.get(opp_pct, 0)
            features[f"Team2_Def_{pct}"] = team2_stats.get(opp_pct, 0)
            features[f"Def_{pct}_Diff"] = team1_stats.get(opp_pct, 0) - team2_stats.get(
                opp_pct, 0
            )

        # Other key stats per game
        for stat in [
            "PointsPerGame",
            "PointsAllowedPerGame",
            "AstRate",
            "BlkRate",
            "StlRate",
        ]:
            if stat in team1_stats:
                features[f"Team1_{stat}"] = team1_stats.get(stat, 0)
                features[f"Team2_{stat}"] = team2_stats.get(stat, 0)
                features[f"{stat}_Diff"] = team1_stats.get(stat, 0) - team2_stats.get(
                    stat, 0
                )

        return features
