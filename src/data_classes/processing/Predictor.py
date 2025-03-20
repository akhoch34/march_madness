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

    def __init__(
        self,
        data_manager: MarchMadnessDataManager,
        elo_system: EloRatingSystem,
        stats_calculator: TeamStatsCalculator,
        ml_model: MarchMadnessMLModel,
        current_season=2025,
    ):
        """
        Initialize the March Madness predictor

        Parameters:
        data_manager: DataManager instance for loading data
        elo_system: EloRatingSystem instance for ELO ratings
        stats_calculator: TeamStatsCalculator instance for advanced metrics
        ml_model: MarchMadnessMLModel instance
        current_season (int): The current season year (for prediction)
        """
        # Initialize component systems
        self.data_manager = data_manager
        self.elo_system = elo_system
        self.stats_calculator = stats_calculator
        self.visualizer = TournamentVisualizer(self.data_manager)
        self.ml_model = ml_model
        self.current_season = current_season

    def initialize_models(
        self, calculate_elo=True, calculate_stats=True, train_ml=True
    ):
        """Initialize all prediction models"""
        if calculate_elo and not self.elo_system.team_elo_ratings:
            print("Calculating ELO ratings...")
            self.elo_system.calculate_elo_ratings()

        if calculate_stats and not self.stats_calculator.advanced_team_stats:
            print("Calculating advanced team statistics...")
            self.stats_calculator.calculate_advanced_team_stats()

        if train_ml:
            print("Training ML model...")
            self.ml_model.train_model(model_type="xgboost")

    def predict_game(
        self, team1_id, team2_id, day_num=134, season=None, method="elo_enhanced"
    ):
        """
        Predict the outcome of a game

        Parameters:
        team1_id, team2_id: Team IDs
        day_num: Day number (defaults to first round of tournament)
        season: Season (defaults to current_season)
        method: Prediction method ('elo', 'elo_enhanced')

        Returns:
        float: Probability of team1 winning
        """
        if season is None:
            season = self.current_season

        # Get ELO prediction
        elo_pred = self.elo_system.predict_game(team1_id, team2_id, day_num, season)

        # Return appropriate prediction
        if method == "elo":
            return elo_pred
        elif method == "elo_enhanced" and self.ml_model is not None:
            # Use ELO-enhanced model
            return self.ml_model.predict(team1_id, team2_id, season, day_num)
        else:
            # Default to ELO
            return elo_pred

    def generate_predictions(
        self,
        submission_file=None,
        method="elo_enhanced",
        get_all_matchups=False,
        tournament_teams_only=False,
    ):
        """
        Generate predictions for the tournament

        Parameters:
        submission_file (str): Optional path to save the submission file
        method (str): Prediction method ('elo', 'elo_enhanced')
        get_all_matchups (bool): Whether to generate predictions for all possible team matchups
        tournament_teams_only (bool): Whether to limit to only actual tournament teams

        Returns:
        DataFrame: Prediction results with team details
        """
        print(f"Generating predictions using {method} method...")
        print("new")

        # Determine which teams to include
        if get_all_matchups:
            # All teams in the dataset
            team_ids = self.data_manager.data["teams"]["TeamID"].unique()
            print(
                f"Generating predictions for all {len(team_ids)} teams ({len(team_ids)*(len(team_ids)-1)//2} matchups)"
            )
        elif tournament_teams_only:
            # Only teams in the tournament
            team_ids = self.data_manager.get_tournament_teams(self.current_season)[
                "TeamID"
            ].unique()
            print(
                f"Generating predictions for {len(team_ids)} tournament teams ({len(team_ids)*(len(team_ids)-1)//2} matchups)"
            )
        else:
            # Default: Use current season teams from seeds
            current_seeds = self.data_manager.data["processed_seeds"][
                self.data_manager.data["processed_seeds"]["Season"]
                == self.current_season
            ]

            if len(current_seeds) == 0:
                raise ValueError(f"No seed data found for season {self.current_season}")

            team_ids = current_seeds["TeamID"].unique()
            print(
                f"Generating predictions for {len(team_ids)} seeded teams ({len(team_ids)*(len(team_ids)-1)//2} matchups)"
            )

        # Generate all possible matchups
        matchups = []

        # Track progress for large datasets
        count = 0
        total = len(team_ids) * (len(team_ids) - 1) // 2

        for i, team1_id in enumerate(team_ids):
            for team2_id in team_ids[i + 1 :]:
                # Create ID in required format
                matchup_id = f"{self.current_season}_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"

                # Make prediction
                pred = self.predict_game(team1_id, team2_id, method=method)

                # Get seed info if available
                team1_seed = self.data_manager.seed_lookup.get(
                    (self.current_season, team1_id), None
                )
                team2_seed = self.data_manager.seed_lookup.get(
                    (self.current_season, team2_id), None
                )

                # Create matchup data
                matchup_data = {
                    "ID": matchup_id,
                    "Pred": pred,
                    "Team1ID": team1_id,
                    "Team2ID": team2_id,
                    "Team1Name": self.data_manager.get_team_name(team1_id),
                    "Team2Name": self.data_manager.get_team_name(team2_id),
                    "Team1ELO": self.elo_system.get_team_elo(
                        self.current_season, team1_id
                    ),
                    "Team2ELO": self.elo_system.get_team_elo(
                        self.current_season, team2_id
                    ),
                }

                # Add seed info if available
                if team1_seed is not None and team2_seed is not None:
                    matchup_data.update(
                        {"Team1Seed": team1_seed, "Team2Seed": team2_seed}
                    )

                matchups.append(matchup_data)

                # Show progress for large datasets
                count += 1
                if total > 1000 and count % 1000 == 0:
                    print(f"Processed {count}/{total} matchups ({count/total:.1%})")

        # Create DataFrame with predictions
        predictions_df = pd.DataFrame(matchups)

        # Save to submission file if requested
        if submission_file:
            # For Kaggle submission, only need ID and Pred columns
            submission_cols = ["ID", "Pred"]

            # Check if file exists and append if needed
            if os.path.exists(submission_file):
                existing_df = pd.read_csv(submission_file)
                submission_df = pd.concat(
                    [existing_df, predictions_df[submission_cols]]
                )
                submission_df = submission_df.drop_duplicates(subset=["ID"])
            else:
                submission_df = predictions_df[submission_cols]

            # Save the file
            submission_df.to_csv(submission_file, index=False)
            print(f"Saved {len(submission_df)} predictions to {submission_file}")

        return predictions_df

    def predict_tournament_bracket(self, method="elo_enhanced"):
        """
        Generate the complete tournament bracket predictions

        Parameters:
        method (str): Prediction method to use

        Returns:
        DataFrame: Tournament bracket predictions
        """
        # Check that we have tournament teams
        tournament_teams = self.data_manager.get_tournament_teams(self.current_season)

        if len(tournament_teams) == 0:
            raise ValueError(
                f"No tournament teams found for season {self.current_season}"
            )

        print(f"Predicting tournament bracket for {len(tournament_teams)} teams...")

        # Generate predictions just for tournament teams
        predictions_df = self.generate_predictions(
            method=method, tournament_teams_only=True
        )

        return predictions_df

    def backtest_tournament(self, test_season, method="elo_enhanced", visualize=True):
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

        # Visualize if requested
        if visualize:
            self.visualizer.visualize_backtest(results)

        return results

    def backtest_multiple_seasons(
        self, seasons=None, method="elo_enhanced", visualize=True
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

    def compare_methods(self, test_seasons=None, visualize=True):
        """
        Compare different prediction methods on multiple seasons

        Parameters:
        test_seasons: List of seasons to test

        Returns:
        DataFrame: Comparison of methods
        """
        methods = ["elo", "elo_enhanced"]

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

        if visualize:
            # Visualize
            plt.figure(figsize=(12, 6))

            metrics = ["Accuracy", "Brier Score", "Log Loss"]
            colors = ["green", "red", "blue"]

            for i, metric in enumerate(metrics):
                plt.subplot(1, 3, i + 1)

                if metric == "Accuracy":
                    # Higher is better
                    bars = plt.bar(
                        comparison["Method"],
                        comparison[metric],
                        color=colors[i],
                        alpha=0.7,
                    )
                    plt.ylabel(metric)
                    plt.title(f"{metric} (higher is better)")
                else:
                    # Lower is better
                    bars = plt.bar(
                        comparison["Method"],
                        comparison[metric],
                        color=colors[i],
                        alpha=0.7,
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

    def tune_elo_parameters(self, test_seasons=None, visualize=True):
        """
        Tune ELO parameters using grid search on historical tournaments

        Parameters:
        test_seasons: List of seasons to test (default: last 5 available seasons)
        visualize: Whether to visualize results

        Returns:
        DataFrame: Parameter tuning results sorted by performance
        """
        if test_seasons is None:
            # Use last 5 seasons by default
            all_seasons = sorted(
                self.data_manager.data["tourney_results"]["Season"].unique()
            )
            test_seasons = all_seasons[-5:]

        print(
            f"Tuning ELO parameters on {len(test_seasons)} tournament seasons: {test_seasons}"
        )

        # Define parameter grid
        k_factors = [10, 15, 20, 22, 25, 28, 30, 35, 40]
        recency_factors = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3]
        recency_windows = [10, 15, 20, 25, 30]
        carry_over_factors = [0.3, 0.4, 0.5, 0.6, 0.7]

        results = []
        best_log_loss = float("inf")
        best_params = None

        # Total combinations to test
        total_combos = (
            len(k_factors)
            * len(recency_factors)
            * len(recency_windows)
            * len(carry_over_factors)
        )
        combo_count = 0

        for k in k_factors:
            for rf in recency_factors:
                for rw in recency_windows:
                    for co in carry_over_factors:
                        combo_count += 1
                        print(
                            f"\nTesting combination {combo_count}/{total_combos}: k={k}, recency_factor={rf}, recency_window={rw}"
                        )

                        # Recalculate ELO ratings with these parameters
                        self.elo_system.calculate_elo_ratings(
                            start_year=2003,
                            k_factor=k,
                            recency_factor=rf,
                            recency_window=rw,
                            carry_over_factor=co,
                        )

                        # Run backtests for each season
                        season_results = []
                        for season in test_seasons:
                            result = self.backtest_tournament(
                                season, method="elo", visualize=False
                            )
                            if result:
                                season_results.append(result)

                        if not season_results:
                            print("No valid backtest results")
                            continue

                        # Calculate aggregate metrics
                        avg_accuracy = np.mean([r["accuracy"] for r in season_results])
                        avg_brier = np.mean([r["brier_score"] for r in season_results])
                        avg_log_loss = np.mean([r["log_loss"] for r in season_results])

                        # Store results
                        result_data = {
                            "k_factor": k,
                            "recency_factor": rf,
                            "recency_window": rw,
                            "carry_over_factor": co,
                            "accuracy": avg_accuracy,
                            "brier_score": avg_brier,
                            "log_loss": avg_log_loss,
                            "num_seasons": len(season_results),
                        }
                        results.append(result_data)

                        # Track best parameters
                        if avg_log_loss < best_log_loss:
                            best_log_loss = avg_log_loss
                            best_params = (k, rf, rw, co)

        # Convert to DataFrame and sort by performance
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("log_loss")

        print("\nParameter Tuning Results (Best Configurations):")
        print(results_df.head(5))

        if best_params:
            print(
                f"\nBest parameters: k_factor={best_params[0]}, recency_factor={best_params[1]}, recency_window={best_params[2]}, carry_over_factor={best_params[3]}"
            )
            print(f"Best log loss: {best_log_loss:.4f}")

        if visualize and len(results_df) > 0:
            self._visualize_parameter_tuning(results_df)

        return results_df

    def _visualize_parameter_tuning(self, results_df):
        """Helper method to visualize parameter tuning results"""
        plt.figure(figsize=(15, 10))

        # 1. Plot k_factor vs log_loss
        plt.subplot(2, 2, 1)
        k_factors = results_df["k_factor"].unique()
        k_loss_values = [
            results_df[results_df["k_factor"] == k]["log_loss"].mean()
            for k in k_factors
        ]
        plt.plot(k_factors, k_loss_values, "o-", linewidth=2)
        plt.xlabel("k_factor")
        plt.ylabel("Average Log Loss")
        plt.title("Effect of k_factor on Log Loss")
        plt.grid(True, alpha=0.3)

        # 2. Plot recency_factor vs log_loss
        plt.subplot(2, 2, 2)
        rf_factors = results_df["recency_factor"].unique()
        rf_loss_values = [
            results_df[results_df["recency_factor"] == rf]["log_loss"].mean()
            for rf in rf_factors
        ]
        plt.plot(rf_factors, rf_loss_values, "o-", linewidth=2)
        plt.xlabel("recency_factor")
        plt.ylabel("Average Log Loss")
        plt.title("Effect of recency_factor on Log Loss")
        plt.grid(True, alpha=0.3)

        # 3. Plot recency_window vs log_loss
        plt.subplot(2, 2, 3)
        rw_values = results_df["recency_window"].unique()
        rw_loss_values = [
            results_df[results_df["recency_window"] == rw]["log_loss"].mean()
            for rw in rw_values
        ]
        plt.plot(rw_values, rw_loss_values, "o-", linewidth=2)
        plt.xlabel("recency_window")
        plt.ylabel("Average Log Loss")
        plt.title("Effect of recency_window on Log Loss")
        plt.grid(True, alpha=0.3)

        # 4. Plot top 10 configurations
        plt.subplot(2, 2, 4)
        top10 = results_df.head(10).copy()
        top10["config"] = top10.apply(
            lambda x: f"k={x['k_factor']}, rf={x['recency_factor']}, rw={x['recency_window']}",
            axis=1,
        )
        plt.barh(top10["config"], top10["log_loss"], color="green", alpha=0.7)
        plt.xlabel("Log Loss")
        plt.title("Top 10 Parameter Configurations")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
