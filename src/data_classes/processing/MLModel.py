import os
import time

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
)
from sklearn.model_selection import GroupKFold
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

try:
    import xgboost as xgb
except ImportError:
    xgb = None

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
            "ELO_residual",
        ]
        self._feature_dataset_created = False
        self.calibrator = None   # isotonic regression calibrator (optional)
        self.bt_model = None     # BradleyTerry model (optional, set externally)
        self._feature_lookup = None
        self._generated_matchup_cache = {}
        self._season_team_stats_cache = {}
        self._season_rank_cache = {}
        self.seed_spline = None  # spline for seed-based calibration
        self.spline_blend = 0.3  # fraction of ML vs spline (0.3 ML + 0.7 spline)

    def _log_every(self, index, total, label, every=250):
        if total <= 0:
            return
        if index == total or index % every == 0:
            print(f"{label}: {index}/{total} ({index / total:.1%})")

    def _feature_cache_path(self):
        path = os.path.join(
            "output",
            str(self.data_manager.current_season),
            "features",
            self.data_manager.gender,
        )
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, "feature_dataset.csv")

    def create_feature_dataset(
        self,
        train_years_range=None,
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
        if train_years_range is None:
            train_years_range = (
                max(2003, self.data_manager.current_season - 8),
                self.data_manager.current_season - 1,
            )

        # Prevent infinite recursion
        if self._feature_dataset_created:
            print("Feature dataset already created, skipping")
            return self.feature_df

        cache_path = self._feature_cache_path()
        if os.path.exists(cache_path):
            self.feature_df = pd.read_csv(cache_path)
            self.feature_columns = [
                col for col in self.feature_df.columns if col not in self.exclude_columns
            ]
            self._initialize_feature_lookup()
            self._feature_dataset_created = True
            print(f"Loaded cached feature dataset from {cache_path}")
            return self.feature_df

        print(
            f"Creating feature dataset for {self.data_manager.gender} "
            f"{self.data_manager.current_season} using tournaments "
            f"{train_years_range[0]}-{train_years_range[1]}..."
        )

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
                start_season=min(train_years_range[0], 2003),
                include_tourney=False,
            )

        # Get all tournament matchups from historical data
        tourney_games = self.data_manager.data["tourney_results"].copy()

        # Create features for each historical matchup
        features = []
        eligible_games = tourney_games[
            (tourney_games["Season"] >= train_years_range[0])
            & (tourney_games["Season"] <= train_years_range[1])
        ]
        total_games = len(eligible_games)
        print(f"Building training rows from {total_games} tournament games...")

        processed_games = 0
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

            # Add coach quality features (9d)
            game_features.update(self._get_coach_features(season, team1_id, team2_id))

            # Add Bradley-Terry strength features (9a)
            if self.bt_model is not None:
                game_features.update(
                    self.bt_model.get_strength_features(team1_id, team2_id, season)
                )

            features.append(game_features)

            # Also add the reversed matchup (with opposite result)
            reversed_features = self._create_reversed_features(game_features)
            features.append(reversed_features)
            processed_games += 1
            self._log_every(
                processed_games,
                total_games,
                "  Feature rows built",
                every=100,
            )

        # Create DataFrame with all features
        self.feature_df = pd.DataFrame(features)
        print(f"Created feature dataset with {len(self.feature_df)} samples")

        # Define feature columns (excluding outcome and identifiers)
        self.feature_columns = [
            col for col in self.feature_df.columns if col not in self.exclude_columns
        ]
        self._initialize_feature_lookup()

        # Set flag to avoid infinite recursion
        self._feature_dataset_created = True
        self.feature_df.to_csv(cache_path, index=False)

        return self.feature_df

    def _initialize_feature_lookup(self):
        if self.feature_df is None or self.feature_df.empty:
            self._feature_lookup = {}
            return

        lookup = {}
        for row in self.feature_df.itertuples(index=False):
            key = (int(row.Season), int(row.Team1ID), int(row.Team2ID))
            lookup[key] = row._asdict()
        self._feature_lookup = lookup

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

        # Add coach quality features (9d)
        game_features.update(self._get_coach_features(season, team1_id, team2_id))

        # Add Bradley-Terry strength features (9a)
        if self.bt_model is not None:
            game_features.update(
                self.bt_model.get_strength_features(team1_id, team2_id, season)
            )

        return game_features

    def get_matchup_features(
        self, team1_id, team2_id, season, day_num=132
    ) -> pd.DataFrame:
        """Get features for a specific matchup, generating them if needed"""
        # Ensure feature dataset exists
        if self.feature_df is None:
            self.create_feature_dataset()

        cache_key = (int(season), int(team1_id), int(team2_id), int(day_num))

        if cache_key in self._generated_matchup_cache:
            return self._generated_matchup_cache[cache_key]

        if self._feature_lookup is None:
            self._initialize_feature_lookup()

        existing_row = None if self._feature_lookup is None else self._feature_lookup.get(
            (int(season), int(team1_id), int(team2_id))
        )

        # If we have existing features, return them
        if existing_row is not None:
            df_existing = pd.DataFrame([existing_row])
            self._generated_matchup_cache[cache_key] = df_existing
            return df_existing

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
            df_new = df_new[self.feature_columns]

        self._generated_matchup_cache[cache_key] = df_new

        return df_new

    def predict(self, team1_id, team2_id, season, day_num=132):
        """
        Make a prediction for a specific matchup using the ELO-enhanced model.
        Returns blend of ML classifier probability and seed-spline anchor.
        Falls back to ELO if model is not available.
        """
        elo_pred = self.elo_system.predict_game(team1_id, team2_id, day_num, season)

        if self.model is None:
            return elo_pred

        try:
            matchup_features = self.get_matchup_features(
                team1_id, team2_id, season, day_num
            )

            if len(matchup_features) == 0:
                print(
                    f"Warning: No features found for {team1_id} vs {team2_id}. Using ELO fallback."
                )
                return elo_pred

            features = matchup_features[self.feature_columns].fillna(0)
            ml_pred = float(self.model.predict_proba(features)[0, 1])

            # Spline-based seed anchor
            if self.seed_spline is not None:
                t1s = self.data_manager.seed_lookup.get((season, team1_id), 8)
                t2s = self.data_manager.seed_lookup.get((season, team2_id), 8)
                seed_diff = t2s - t1s
                spline_pred = float(np.clip(self.seed_spline(seed_diff), 0.025, 0.975))
                raw_pred = self.spline_blend * ml_pred + (1 - self.spline_blend) * spline_pred
            else:
                raw_pred = ml_pred

            raw_pred = float(np.clip(raw_pred, 0.01, 0.999))

            if self.calibrator is not None:
                return float(np.clip(self.calibrator.predict([raw_pred])[0], 0.01, 0.999))
            return raw_pred

        except Exception as e:
            print(f"Error making prediction: {e}")
            return elo_pred

    def predict_many(self, matchups, season, day_num=132):
        """Bulk prediction path for a list of (team1_id, team2_id) matchups."""
        if self.model is None:
            return [
                self.elo_system.predict_game(team1_id, team2_id, day_num, season)
                for team1_id, team2_id in matchups
            ]

        rows = []
        seed_diffs = []
        total = len(matchups)
        print(f"Preparing bulk feature rows for {total} matchups...")

        for index, (team1_id, team2_id) in enumerate(matchups, start=1):
            t1s = self.data_manager.seed_lookup.get((season, team1_id), 8)
            t2s = self.data_manager.seed_lookup.get((season, team2_id), 8)
            seed_diffs.append(t2s - t1s)
            matchup_features = self.get_matchup_features(
                team1_id, team2_id, season, day_num
            )
            rows.append(matchup_features.iloc[0].to_dict())
            self._log_every(index, total, "  Bulk features prepared", every=250)

        features_df = pd.DataFrame(rows)
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[self.feature_columns].fillna(0)
        print(
            f"Running model inference on shape={features_df.shape} "
            f"for season {season}..."
        )

        ml_preds = self.model.predict_proba(features_df)[:, 1]
        print("Bulk model inference complete.")

        if self.seed_spline is not None:
            spline_preds = np.clip(self.seed_spline(np.array(seed_diffs)), 0.025, 0.975)
            raw_preds = self.spline_blend * ml_preds + (1 - self.spline_blend) * spline_preds
        else:
            raw_preds = ml_preds

        raw_preds = np.clip(raw_preds, 0.01, 0.999)

        if self.calibrator is not None:
            print("Applying calibrated probability mapping...")
            return np.clip(self.calibrator.predict(raw_preds), 0.01, 0.999).tolist()

        return raw_preds.tolist()

    def train_model(
        self,
        model_type="xgboost",
        test_size=0.2,
        random_state=42,
        train_years_range=None,
        calibrate=False,
    ):
        """
        Train a direct binary classifier for tournament win probability.
        Uses GroupKFold(Season) to prevent same-season row leakage.
        Blends ML prediction with a seed-spline anchor (0.3 ML + 0.7 spline).
        """
        print("Training ELO-enhanced ML model (direct classifier + spline blend)...")

        if train_years_range is None:
            train_years_range = (
                max(2003, self.data_manager.current_season - 8),
                self.data_manager.current_season - 1,
            )
        print(
            f"Training window: {train_years_range[0]}-{train_years_range[1]} "
            f"for season {self.data_manager.current_season}"
        )

        # Create or ensure feature dataset exists
        if self.feature_df is None or not self._feature_dataset_created:
            self.create_feature_dataset(train_years_range=train_years_range)

        if "Result" not in self.feature_df.columns:
            print("Error: Result column not found in feature dataset.")
            return None

        # Prepare features and target (direct classification)
        X = self.feature_df[self.feature_columns].fillna(0)
        y = self.feature_df["Result"]
        groups = self.feature_df["Season"]

        print(f"Feature matrix shape: {X.shape}")
        print(f"Training seasons: {sorted(groups.unique().tolist())}")

        # Fit seed-spline on all training data
        seed_diffs = self.feature_df["SeedDiff"].values
        results = y.values
        sort_idx = np.argsort(seed_diffs)
        try:
            self.seed_spline = UnivariateSpline(
                seed_diffs[sort_idx],
                results[sort_idx],
                s=len(results),
                ext=3,
            )
            print("Seed spline fitted.")
        except Exception as e:
            print(f"Warning: spline fitting failed ({e}). No spline blend will be used.")
            self.seed_spline = None

        # Initialize classifier
        def _make_clf():
            if xgb is None:
                return HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.05,
                    max_depth=3,
                    min_samples_leaf=20,
                    random_state=random_state,
                )
            return xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                min_child_weight=3,
                subsample=0.7,
                colsample_bytree=0.7,
                eval_metric="logloss",
                random_state=random_state,
                verbosity=0,
            )

        # GroupKFold CV for evaluation (prevents same-season row leakage)
        gkf = GroupKFold(n_splits=5)
        oof_ml = np.zeros(len(X))
        oof_spline = np.zeros(len(X))
        print("Running GroupKFold(5) cross-validation by Season...")
        for fold_num, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
            tmp = _make_clf()
            tmp.fit(X.iloc[train_idx], y.iloc[train_idx])
            oof_ml[val_idx] = tmp.predict_proba(X.iloc[val_idx])[:, 1]
            if self.seed_spline is not None:
                oof_spline[val_idx] = np.clip(
                    self.seed_spline(self.feature_df["SeedDiff"].values[val_idx]),
                    0.025, 0.975,
                )
            else:
                oof_spline[val_idx] = 0.5
            print(f"  CV fold {fold_num}/5 complete")

        # Evaluate CV performance
        oof_blended = self.spline_blend * oof_ml + (1 - self.spline_blend) * oof_spline
        y_arr = y.values
        cv_brier = brier_score_loss(y_arr, oof_blended)
        cv_acc = accuracy_score(y_arr, oof_blended > 0.5)
        cv_logloss = log_loss(y_arr, np.clip(oof_blended, 0.01, 0.999))
        print(
            f"\nCV (GroupKFold/Season) — Brier: {cv_brier:.4f}  "
            f"Accuracy: {cv_acc:.4f}  LogLoss: {cv_logloss:.4f}"
        )

        # Train final model on all data
        print(f"Fitting final {model_type} classifier on all {len(X)} samples...")
        self.model = _make_clf()
        self.model.fit(X, y)
        print("Final model fit complete.")

        # Optional isotonic calibration on OOF predictions
        if calibrate:
            try:
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(oof_blended, y_arr)
                self.calibrator = calibrator
                print("Isotonic calibration fit on OOF blended predictions.")
            except Exception as e:
                print(f"Warning: calibration fitting failed ({e}). Skipping.")
                self.calibrator = None
        else:
            self.calibrator = None

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

    def _build_season_team_stats_cache(self, season):
        if season in self._season_team_stats_cache:
            return
        start = time.time()
        print(f"Building season stats cache for {season}...")

        season_games = self.data_manager.data["regular_season"][
            self.data_manager.data["regular_season"]["Season"] == season
        ].copy()

        all_games = [season_games]
        if (
            hasattr(self.data_manager, "secondary_tourney_available")
            and self.data_manager.secondary_tourney_available
            and "secondary_tourney" in self.data_manager.data
        ):
            all_games.append(
                self.data_manager.data["secondary_tourney"][
                    self.data_manager.data["secondary_tourney"]["Season"] == season
                ].copy()
            )

        conf_season = pd.DataFrame()
        if (
            hasattr(self.data_manager, "conf_tourney_available")
            and self.data_manager.conf_tourney_available
            and "conf_tourney" in self.data_manager.data
        ):
            conf_season = self.data_manager.data["conf_tourney"][
                self.data_manager.data["conf_tourney"]["Season"] == season
            ].copy()
            all_games.append(conf_season)

        all_games = pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()
        team_game_rows = []
        for _, game in all_games.iterrows():
            team_game_rows.append(
                {
                    "TeamID": int(game["WTeamID"]),
                    "OpponentID": int(game["LTeamID"]),
                    "DayNum": int(game["DayNum"]),
                    "Result": 1.0,
                    "isWin": 1,
                    "ScoreMargin": float(game["WScore"] - game["LScore"]),
                    "ScoreFor": float(game["WScore"]),
                }
            )
            team_game_rows.append(
                {
                    "TeamID": int(game["LTeamID"]),
                    "OpponentID": int(game["WTeamID"]),
                    "DayNum": int(game["DayNum"]),
                    "Result": 0.0,
                    "isWin": 0,
                    "ScoreMargin": float(game["LScore"] - game["WScore"]),
                    "ScoreFor": float(game["LScore"]),
                }
            )

        team_games_df = pd.DataFrame(team_game_rows)
        if not team_games_df.empty:
            team_games_df = team_games_df.sort_values(["TeamID", "DayNum"])

        reg_wins = season_games.groupby("WTeamID").size().to_dict()
        reg_losses = season_games.groupby("LTeamID").size().to_dict()
        season_stats = (
            self.stats_calculator.advanced_team_stats.get(season, {})
            if hasattr(self.stats_calculator, "advanced_team_stats")
            else {}
        )

        conf_champs = set()
        if not conf_season.empty:
            if "ConfAbbrev" in conf_season.columns:
                for _, grp in conf_season.groupby("ConfAbbrev"):
                    conf_champs.update(grp[grp["DayNum"] == grp["DayNum"].max()]["WTeamID"].astype(int).tolist())
            else:
                conf_champs.update(
                    conf_season[conf_season["DayNum"] == conf_season["DayNum"].max()]["WTeamID"].astype(int).tolist()
                )

        cache = {}
        team_ids = set(team_games_df["TeamID"].unique().tolist()) if not team_games_df.empty else set()
        team_ids.update(int(t) for t in season_games["WTeamID"].unique().tolist())
        team_ids.update(int(t) for t in season_games["LTeamID"].unique().tolist())

        for team_id in team_ids:
            team_games = (
                team_games_df[team_games_df["TeamID"] == team_id].copy()
                if not team_games_df.empty
                else pd.DataFrame()
            )
            wins = float(reg_wins.get(team_id, 0))
            losses = float(reg_losses.get(team_id, 0))
            win_pct = wins / (wins + losses) if (wins + losses) > 0 else 0.0

            def recent_mean(values, default=0.0):
                return float(values.mean()) if len(values) > 0 else default

            last10 = team_games.tail(10)
            last10_win_pct = recent_mean(last10["Result"], win_pct)

            if len(last10) > 0:
                weights = np.exp(np.arange(len(last10)) / 3)
                weights = weights / weights.sum()
                exp_win_pct = float((last10["Result"].to_numpy() * weights).sum())
            else:
                exp_win_pct = 0.0

            last5_win_pct = recent_mean(team_games.tail(5)["Result"], win_pct)
            prev5 = team_games.iloc[-10:-5]
            prev5_win_pct = recent_mean(prev5["Result"], win_pct)
            momentum = last5_win_pct - prev5_win_pct

            recent_scores = recent_mean(team_games.tail(5)["ScoreFor"], 0.0)
            season_scores = recent_mean(team_games["ScoreFor"], 0.0)
            scoring_trend = recent_scores - season_scores if len(team_games) >= 5 else 0.0

            recent_margin = recent_mean(team_games.tail(5)["ScoreMargin"], 0.0) if len(team_games) >= 5 else 0.0
            avg_mov = recent_mean(team_games["ScoreMargin"], 0.0)

            streak = 0
            if len(team_games) > 0:
                results = team_games["isWin"].to_numpy()
                current_result = results[-1]
                for idx in range(len(results) - 1, -1, -1):
                    if results[idx] != current_result:
                        break
                    streak += 1 if current_result == 1 else -1

            conf_games = team_games[(team_games["DayNum"] >= 118) & (team_games["DayNum"] < 134)]
            conf_win_pct = recent_mean(conf_games["Result"], win_pct)
            conf_depth = float(conf_games["DayNum"].max() - 118) if len(conf_games) > 0 else 0.0

            late_games = team_games[team_games["DayNum"] >= 70]
            late_win_pct = recent_mean(late_games["Result"], win_pct)

            opp_ids = (
                season_games[season_games["WTeamID"] == team_id]["LTeamID"].tolist()
                + season_games[season_games["LTeamID"] == team_id]["WTeamID"].tolist()
            )
            opp_net_eff = [season_stats.get(int(opp), {}).get("NetEff", 0) for opp in opp_ids]
            sos = float(np.mean(opp_net_eff)) if opp_net_eff else 0.0

            cache[int(team_id)] = {
                "WinPct": win_pct,
                "SOS": sos,
                "Last10": last10_win_pct,
                "ExpWinPct": exp_win_pct,
                "Momentum": momentum,
                "ScoringTrend": scoring_trend,
                "RecentMargin": recent_margin,
                "Streak": float(streak),
                "ConfWinPct": conf_win_pct,
                "ConfDepth": conf_depth,
                "LateWinPct": late_win_pct,
                "AvgMOV": avg_mov,
                "ConfChamp": float(team_id in conf_champs),
            }

        self._season_team_stats_cache[season] = cache
        print(
            f"Season stats cache ready for {season}: {len(cache)} teams "
            f"in {time.time() - start:.1f}s"
        )

    def _build_season_rank_cache(self, season):
        if season in self._season_rank_cache:
            return
        start = time.time()

        if not self.data_manager.rankings_available:
            self._season_rank_cache[season] = {}
            return

        print(f"Building ranking cache for {season}...")

        rankings = self.data_manager.data["rankings"]
        season_rankings = rankings[rankings["Season"] == season]
        team_ids = season_rankings["TeamID"].unique().tolist()
        cache = {}

        def checkpoint_mean(team_id, day_min, day_max, default):
            sub = season_rankings[
                (season_rankings["TeamID"] == team_id)
                & (season_rankings["RankingDayNum"] >= day_min)
                & (season_rankings["RankingDayNum"] <= day_max)
            ]
            return float(sub["OrdinalRank"].mean()) if len(sub) > 0 else default

        for team_id in team_ids:
            late = checkpoint_mean(team_id, 133, 133, 353.0)
            if late == 353.0:
                late = checkpoint_mean(team_id, 120, 133, 353.0)
            early = checkpoint_mean(team_id, 30, 60, 200.0)
            mid = checkpoint_mean(team_id, 80, 100, 200.0)
            team_sub = season_rankings[season_rankings["TeamID"] == team_id]
            consistency = float(team_sub["OrdinalRank"].std()) if len(team_sub) > 1 else 50.0
            cache[int(team_id)] = {
                "AvgRank": late,
                "EarlyRank": early,
                "MidRank": mid,
                "RankTraj": early - late,
                "RankConsistency": consistency,
            }

        self._season_rank_cache[season] = cache
        print(
            f"Ranking cache ready for {season}: {len(cache)} teams "
            f"in {time.time() - start:.1f}s"
        )

    def _get_season_stats(self, season, team1_id, team2_id):
        """Get season performance stats for both teams with enhanced recency metrics"""
        self._build_season_team_stats_cache(season)
        team1_stats = self._season_team_stats_cache.get(season, {}).get(int(team1_id), {})
        team2_stats = self._season_team_stats_cache.get(season, {}).get(int(team2_id), {})

        def get_stat(team_stats, key, default=0.0):
            return float(team_stats.get(key, default))

        return {
            "Team1WinPct": get_stat(team1_stats, "WinPct"),
            "Team2WinPct": get_stat(team2_stats, "WinPct"),
            "WinPctDiff": get_stat(team1_stats, "WinPct") - get_stat(team2_stats, "WinPct"),
            "Team1SOS": get_stat(team1_stats, "SOS"),
            "Team2SOS": get_stat(team2_stats, "SOS"),
            "SOSDiff": get_stat(team1_stats, "SOS") - get_stat(team2_stats, "SOS"),
            "Team1Last10": get_stat(team1_stats, "Last10"),
            "Team2Last10": get_stat(team2_stats, "Last10"),
            "Last10Diff": get_stat(team1_stats, "Last10") - get_stat(team2_stats, "Last10"),
            "Team1ExpWinPct": get_stat(team1_stats, "ExpWinPct"),
            "Team2ExpWinPct": get_stat(team2_stats, "ExpWinPct"),
            "ExpWinPctDiff": get_stat(team1_stats, "ExpWinPct") - get_stat(team2_stats, "ExpWinPct"),
            "Team1Momentum": get_stat(team1_stats, "Momentum"),
            "Team2Momentum": get_stat(team2_stats, "Momentum"),
            "MomentumDiff": get_stat(team1_stats, "Momentum") - get_stat(team2_stats, "Momentum"),
            "Team1ScoringTrend": get_stat(team1_stats, "ScoringTrend"),
            "Team2ScoringTrend": get_stat(team2_stats, "ScoringTrend"),
            "ScoringTrendDiff": get_stat(team1_stats, "ScoringTrend") - get_stat(team2_stats, "ScoringTrend"),
            "Team1RecentMargin": get_stat(team1_stats, "RecentMargin"),
            "Team2RecentMargin": get_stat(team2_stats, "RecentMargin"),
            "RecentMarginDiff": get_stat(team1_stats, "RecentMargin") - get_stat(team2_stats, "RecentMargin"),
            "Team1Streak": get_stat(team1_stats, "Streak"),
            "Team2Streak": get_stat(team2_stats, "Streak"),
            "StreakDiff": get_stat(team1_stats, "Streak") - get_stat(team2_stats, "Streak"),
            "Team1ConfWinPct": get_stat(team1_stats, "ConfWinPct"),
            "Team2ConfWinPct": get_stat(team2_stats, "ConfWinPct"),
            "ConfWinPctDiff": get_stat(team1_stats, "ConfWinPct") - get_stat(team2_stats, "ConfWinPct"),
            "Team1ConfDepth": get_stat(team1_stats, "ConfDepth"),
            "Team2ConfDepth": get_stat(team2_stats, "ConfDepth"),
            "ConfDepthDiff": get_stat(team1_stats, "ConfDepth") - get_stat(team2_stats, "ConfDepth"),
            "Team1LateWinPct": get_stat(team1_stats, "LateWinPct"),
            "Team2LateWinPct": get_stat(team2_stats, "LateWinPct"),
            "LateWinPctDiff": get_stat(team1_stats, "LateWinPct") - get_stat(team2_stats, "LateWinPct"),
            "Team1_AvgMOV": get_stat(team1_stats, "AvgMOV"),
            "Team2_AvgMOV": get_stat(team2_stats, "AvgMOV"),
            "AvgMOV_Diff": get_stat(team1_stats, "AvgMOV") - get_stat(team2_stats, "AvgMOV"),
            "Team1_ConfChamp": get_stat(team1_stats, "ConfChamp"),
            "Team2_ConfChamp": get_stat(team2_stats, "ConfChamp"),
            "ConfChamp_Diff": get_stat(team1_stats, "ConfChamp") - get_stat(team2_stats, "ConfChamp"),
        }

    def _get_ranking_features(self, season, team1_id, team2_id):
        """Get pre-tournament ranking features for both teams"""
        if not self.data_manager.rankings_available:
            return {}

        self._build_season_rank_cache(season)
        season_cache = self._season_rank_cache.get(season, {})
        team1_rank = season_cache.get(int(team1_id), {})
        team2_rank = season_cache.get(int(team2_id), {})

        team1_avg_rank = float(team1_rank.get("AvgRank", 353.0))
        team2_avg_rank = float(team2_rank.get("AvgRank", 353.0))
        t1_early = float(team1_rank.get("EarlyRank", 200.0))
        t2_early = float(team2_rank.get("EarlyRank", 200.0))
        t1_mid = float(team1_rank.get("MidRank", 200.0))
        t2_mid = float(team2_rank.get("MidRank", 200.0))
        t1_traj = float(team1_rank.get("RankTraj", t1_early - team1_avg_rank))
        t2_traj = float(team2_rank.get("RankTraj", t2_early - team2_avg_rank))
        t1_consistency = float(team1_rank.get("RankConsistency", 50.0))
        t2_consistency = float(team2_rank.get("RankConsistency", 50.0))

        return {
            "Team1AvgRank": team1_avg_rank,
            "Team2AvgRank": team2_avg_rank,
            "RankDiff": team2_avg_rank - team1_avg_rank,
            "Team1_EarlyRank": t1_early,
            "Team2_EarlyRank": t2_early,
            "EarlyRank_Diff": t2_early - t1_early,
            "Team1_MidRank": t1_mid,
            "Team2_MidRank": t2_mid,
            "MidRank_Diff": t2_mid - t1_mid,
            "Team1_RankTraj": t1_traj,
            "Team2_RankTraj": t2_traj,
            "RankTraj_Diff": t1_traj - t2_traj,
            "Team1_RankConsistency": t1_consistency,
            "Team2_RankConsistency": t2_consistency,
            "RankConsistency_Diff": t1_consistency - t2_consistency,
        }

    def _get_advanced_stats_features(self, season, team1_id, team2_id):
        """Get advanced stats features for both teams"""
        if (
            not hasattr(self.stats_calculator, "advanced_team_stats")
            or not self.stats_calculator.advanced_team_stats
        ):
            self.stats_calculator.calculate_advanced_team_stats(include_tourney=False)

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

    def _get_coach_features(self, season, team1_id, team2_id):
        """
        Get coach quality features for both teams.
        Requires TeamStatsCalculator.calculate_coach_features() to have been called.
        Returns empty dict if coach features are not available (e.g., women's tournament).
        """
        if (
            not hasattr(self.stats_calculator, "coach_features")
            or not self.stats_calculator.coach_features
        ):
            return {}

        def get_cf(team_id):
            return self.stats_calculator.coach_features.get((season, team_id), {})

        cf1 = get_cf(team1_id)
        cf2 = get_cf(team2_id)

        if not cf1 and not cf2:
            return {}

        def safe(d, key, default=0.0):
            return d.get(key, default)

        return {
            "Team1_CoachTenure": safe(cf1, "tenure"),
            "Team2_CoachTenure": safe(cf2, "tenure"),
            "CoachTenure_Diff": safe(cf1, "tenure") - safe(cf2, "tenure"),
            "Team1_CoachTourneyApps": safe(cf1, "tourney_apps"),
            "Team2_CoachTourneyApps": safe(cf2, "tourney_apps"),
            "CoachTourneyApps_Diff": safe(cf1, "tourney_apps") - safe(cf2, "tourney_apps"),
            "Team1_CoachTourneyWinRate": safe(cf1, "tourney_win_rate"),
            "Team2_CoachTourneyWinRate": safe(cf2, "tourney_win_rate"),
            "CoachTourneyWinRate_Diff": safe(cf1, "tourney_win_rate") - safe(cf2, "tourney_win_rate"),
            "Team1_IsFirstYearCoach": float(safe(cf1, "is_first_year", False)),
            "Team2_IsFirstYearCoach": float(safe(cf2, "is_first_year", False)),
            "IsFirstYearCoach_Diff": float(safe(cf1, "is_first_year", False)) - float(safe(cf2, "is_first_year", False)),
        }
