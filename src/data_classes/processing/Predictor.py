from .DataManager import MarchMadnessDataManager
from .EloRatingSystem import EloRatingSystem
from .TeamStatsCalculator import TeamStatsCalculator
from .TournamentVisualizer import TournamentVisualizer
from .MLModel import MarchMadnessMLModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class MarchMadnessPredictor:
    """Master class that orchestrates the tournament prediction process"""

    def __init__(self, data_dir, gender="M", current_season=2025):
        """
        Initialize the March Madness predictor

        Parameters:
        data_dir (str): Directory containing the data files
        gender (str): 'M' for men's tournament, 'W' for women's
        current_season (int): The current season year (for prediction)
        """
        # Initialize data manager
        self.data_manager = MarchMadnessDataManager(data_dir, gender, current_season)
        self.data_manager.load_data()

        # Initialize component systems
        self.elo_system = EloRatingSystem(self.data_manager)
        self.stats_calculator = TeamStatsCalculator(self.data_manager)
        self.visualizer = TournamentVisualizer(self.data_manager)

        # Initialize ML model
        self.ml_model = MarchMadnessMLModel(
            self.data_manager, self.elo_system, self.stats_calculator
        )

        # Store current season
        self.current_season = current_season

    def initialize_models(
        self, calculate_elo=True, calculate_stats=True, train_ml=False
    ):
        """Initialize all prediction models"""
        if calculate_elo:
            self.elo_system.calculate_elo_ratings()

        if calculate_stats and self.data_manager.detailed_stats_available:
            self.stats_calculator.calculate_advanced_team_stats()

        if train_ml:
            self.ml_model = MarchMadnessMLModel(
                self.data_manager, self.elo_system, self.stats_calculator
            )
            self.ml_model.train_model()

    def predict_game(
        self, team1_id, team2_id, day_num=134, season=None, method="ensemble"
    ):
        """
        Predict the outcome of a game

        Parameters:
        team1_id, team2_id: Team IDs
        day_num: Day number (defaults to first round of tournament)
        season: Season (defaults to current_season)
        method: Prediction method ('elo', 'ml', or 'ensemble')

        Returns:
        float: Probability of team1 winning
        """
        if season is None:
            season = self.current_season

        # Get ELO prediction
        elo_pred = self.elo_system.predict_game(team1_id, team2_id, day_num, season)

        # Get ML prediction if available and requested
        ml_pred = None
        if method in ["ml", "ensemble"] and self.ml_model is not None:
            ml_pred = self.ml_model.predict(team1_id, team2_id, day_num, season)

        # Return appropriate prediction
        if method == "elo":
            return elo_pred
        elif method == "ml" and ml_pred is not None:
            return ml_pred
        elif method == "ensemble" and ml_pred is not None:
            # Simple weighted ensemble
            return 0.6 * ml_pred + 0.4 * elo_pred
        else:
            # Default to ELO if ML not available
            return elo_pred

    def generate_predictions(
        self, submission_file=None, method="ensemble", get_all_matchups=False
    ):
        """
        Generate predictions for the current tournament

        Parameters:
        submission_file (str): Path to save the submission file
        method: Prediction method ('elo', 'ml', or 'ensemble')

        Returns:
        DataFrame: Prediction results
        """
        if get_all_matchups:
            team_ids = self.data_manager.data["teams"]["TeamID"].unique()
        # Get current season seeds
        else:
            current_seeds = self.data_manager.data["processed_seeds"][
                self.data_manager.data["processed_seeds"]["Season"]
                == self.current_season
            ]

            if len(current_seeds) == 0:
                raise ValueError(f"No seed data found for season {self.current_season}")

            # Generate all possible matchups
            team_ids = current_seeds["TeamID"].unique()
        matchups = []

        for i, team1_id in enumerate(team_ids):
            for team2_id in team_ids[i + 1 :]:
                # Create ID in required format
                matchup_id = f"{self.current_season}_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"

                # Make prediction
                if team1_id < team2_id:
                    pred = self.predict_game(team1_id, team2_id, method=method)
                else:
                    pred = 1.0 - self.predict_game(team2_id, team1_id, method=method)

                matchups.append(
                    {
                        "ID": matchup_id,
                        "Pred": pred,
                        "Team1Name": self.data_manager.get_team_name(team1_id),
                        "Team2Name": self.data_manager.get_team_name(team2_id),
                        "Team1ELO": self.elo_system.get_team_elo(
                            self.current_season, team1_id
                        ),
                        "Team2ELO": self.elo_system.get_team_elo(
                            self.current_season, team2_id
                        ),
                    }
                )

        # Create submission DataFrame
        submission_df = pd.DataFrame(matchups)

        # Save to file if requested
        if submission_file:
            if os.path.exists(submission_file):
                existing_df = pd.read_csv(submission_file)
                submission_df = pd.concat([existing_df, submission_df])
            submission_df.to_csv(submission_file, index=False)
            print(f"Saved {len(submission_df)} predictions to {submission_file}")

        return submission_df

    def backtest_tournament(self, test_season, method="ensemble", visualize=True):
        """
        Backtest predictions on a historical tournament

        Parameters:
        test_season: Season to test on
        method: Prediction method to test
        visualize: Whether to visualize results

        Returns:
        dict: Evaluation metrics
        """
        print(f"Backtesting on {test_season} tournament using {method} method...")

        # Get tournament games for the season
        tourney_games = self.data_manager.data["tourney_results"]
        test_games = tourney_games[tourney_games["Season"] == test_season]

        if len(test_games) == 0:
            print(f"No games found for {test_season} tournament")
            return None

        # Generate predictions and evaluate
        predictions = []
        actuals = []
        game_details = []

        for _, game in test_games.iterrows():
            day_num = game["DayNum"]
            team1_id = game["WTeamID"]  # Winner
            team2_id = game["LTeamID"]  # Loser

            # Get prediction
            pred = self.predict_game(
                team1_id, team2_id, day_num, test_season, method=method
            )

            # Store prediction and result
            predictions.append(pred)
            actuals.append(1)  # Team1 won

            # Get seeds
            team1_seed = self.data_manager.seed_lookup.get(
                (test_season, team1_id), None
            )
            team2_seed = self.data_manager.seed_lookup.get(
                (test_season, team2_id), None
            )

            # Store game details
            game_details.append(
                {
                    "DayNum": day_num,
                    "Round": self.data_manager.get_tournament_round(day_num),
                    "Team1ID": team1_id,
                    "Team2ID": team2_id,
                    "Team1Seed": team1_seed,
                    "Team2Seed": team2_seed,
                    "SeedDiff": (
                        team2_seed - team1_seed if team1_seed and team2_seed else None
                    ),
                    "Team1Score": game["WScore"],
                    "Team2Score": game["LScore"],
                    "ScoreDiff": game["WScore"] - game["LScore"],
                    "Prediction": pred,
                    "Actual": 1,
                    "Correct": pred >= 0.5,  # Prediction was correct if >= 0.5
                }
            )

            # Also add reversed matchup for evaluation
            predictions.append(1 - pred)
            actuals.append(0)  # Team2 lost

            # Store reversed game details
            game_details.append(
                {
                    "DayNum": day_num,
                    "Round": self.data_manager.get_tournament_round(day_num),
                    "Team1ID": team2_id,
                    "Team2ID": team1_id,
                    "Team1Seed": team2_seed,
                    "Team2Seed": team1_seed,
                    "SeedDiff": (
                        team1_seed - team2_seed if team1_seed and team2_seed else None
                    ),
                    "Team1Score": game["LScore"],
                    "Team2Score": game["WScore"],
                    "ScoreDiff": game["LScore"] - game["WScore"],
                    "Prediction": 1 - pred,
                    "Actual": 0,
                    "Correct": (1 - pred) < 0.5,  # Prediction was correct if < 0.5
                }
            )

        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Brier score (mean squared error)
        brier_score = np.mean((predictions - actuals) ** 2)

        # Accuracy
        accuracy = np.mean((predictions > 0.5) == actuals)

        # Log loss
        epsilon = 1e-15  # Prevent log(0)
        predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
        log_loss_value = -np.mean(
            actuals * np.log(predictions_clipped)
            + (1 - actuals) * np.log(1 - predictions_clipped)
        )

        # Store results
        results = {
            "season": test_season,
            "method": method,
            "num_games": len(test_games),
            "brier_score": brier_score,
            "accuracy": accuracy,
            "log_loss": log_loss_value,
            "game_details": game_details,
        }

        # Print summary
        print(f"Brier Score: {brier_score:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Log Loss: {log_loss_value:.4f}")

        # Visualize if requested
        if visualize:
            self.visualizer.visualize_backtest(results)

        return results

    def backtest_multiple_seasons(
        self, seasons=None, method="ensemble", visualize=True
    ):
        """
        Backtest on multiple tournament seasons

        Parameters:
        seasons: List of seasons to test (default: last 5 available seasons)
        method: Prediction method to test
        visualize: Whether to visualize aggregate results

        Returns:
        dict: Aggregate and per-season metrics
        """
        if seasons is None:
            # Use last 5 seasons by default
            all_seasons = sorted(
                self.data_manager.data["tourney_results"]["Season"].unique()
            )
            seasons = all_seasons[-5:]

        print(
            f"Backtesting {method} predictions on {len(seasons)} tournament seasons: {seasons}"
        )

        # Run backtests
        all_results = []
        for season in seasons:
            result = self.backtest_tournament(season, method=method, visualize=False)
            if result:
                all_results.append(result)

        if not all_results:
            print("No valid backtest results")
            return None

        # Calculate aggregate metrics
        agg_metrics = {
            "num_seasons": len(all_results),
            "method": method,
            "brier_score": np.mean([r["brier_score"] for r in all_results]),
            "accuracy": np.mean([r["accuracy"] for r in all_results]),
            "log_loss": np.mean([r["log_loss"] for r in all_results]),
        }

        # Print aggregate results
        print("\nAggregate Results:")
        print(f"Seasons: {len(all_results)}")
        print(f"Average Brier Score: {agg_metrics['brier_score']:.4f}")
        print(f"Average Accuracy: {agg_metrics['accuracy']:.4f}")
        print(f"Average Log Loss: {agg_metrics['log_loss']:.4f}")

        # Visualize if requested
        if visualize:
            self.visualizer.visualize_multiple_backtests(all_results)

        return {"aggregate": agg_metrics, "per_season": all_results}

    def compare_methods(self, test_seasons=None):
        """
        Compare different prediction methods on multiple seasons

        Parameters:
        test_seasons: List of seasons to test

        Returns:
        DataFrame: Comparison of methods
        """
        methods = ["elo"]
        if self.ml_model is not None:
            methods.extend(["ml", "ensemble"])

        results = []

        for method in methods:
            result = self.backtest_multiple_seasons(
                seasons=test_seasons, method=method, visualize=False
            )
            if result:
                agg = result["aggregate"]
                results.append(
                    {
                        "Method": method,
                        "Accuracy": agg["accuracy"],
                        "Brier Score": agg["brier_score"],
                        "Log Loss": agg["log_loss"],
                        "Seasons": agg["num_seasons"],
                    }
                )

        # Convert to DataFrame
        comparison = pd.DataFrame(results)

        # Display results
        print("Method Comparison:")
        print(comparison)

        # Visualize
        plt.figure(figsize=(12, 6))

        metrics = ["Accuracy", "Brier Score", "Log Loss"]
        colors = ["green", "red", "blue"]

        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i + 1)

            if metric == "Accuracy":
                # Higher is better
                bars = plt.bar(
                    comparison["Method"], comparison[metric], color=colors[i], alpha=0.7
                )
                plt.ylabel(metric)
                plt.title(f"{metric} (higher is better)")
            else:
                # Lower is better
                bars = plt.bar(
                    comparison["Method"], comparison[metric], color=colors[i], alpha=0.7
                )
                plt.ylabel(metric)
                plt.title(f"{metric} (lower is better)")

            # Add values on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.005,
                    f"{height:.4f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.show()

        return comparison
