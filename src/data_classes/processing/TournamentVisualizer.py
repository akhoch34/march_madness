from .DataManager import MarchMadnessDataManager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TournamentVisualizer:
    """Class for visualizing tournament predictions and results"""

    def __init__(self, data_manager: MarchMadnessDataManager):
        """
        Initialize the tournament visualizer

        Parameters:
        data_manager: MarchMadnessDataManager instance
        """
        self.data_manager = data_manager

    def visualize_backtest(self, results):
        """Visualize backtest results for a single season"""
        # Get game details
        game_details = results["game_details"]

        # Get unique games (odd indices are duplicates with flipped teams)
        unique_games = [game for i, game in enumerate(game_details) if i % 2 == 0]

        # Convert to DataFrame
        df = pd.DataFrame(unique_games)

        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        # Add team names
        df["Team1Name"] = df["Team1ID"].apply(self.data_manager.get_team_name)
        df["Team2Name"] = df["Team2ID"].apply(self.data_manager.get_team_name)
        df["MatchupLabel"] = df.apply(
            lambda x: f"{x['Team1Name']} vs {x['Team2Name']}", axis=1
        )

        # 1. Predictions vs. Actual outcomes
        axs[0, 0].scatter(df["Prediction"], df["Actual"], alpha=0.7)
        axs[0, 0].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axs[0, 0].set_xlabel("Predicted Probability")
        axs[0, 0].set_ylabel("Actual Outcome")
        axs[0, 0].set_title("Prediction Calibration")
        axs[0, 0].grid(True, alpha=0.3)

        # 2. Prediction by seed difference
        df_with_seeds = df[df["SeedDiff"].notna()].copy()
        if len(df_with_seeds) > 0:
            df_with_seeds["AbsSeedDiff"] = df_with_seeds["SeedDiff"].abs()
            seed_groups = (
                df_with_seeds.groupby("AbsSeedDiff")
                .agg({"Prediction": "mean", "Actual": "mean", "Team1ID": "count"})
                .rename(columns={"Team1ID": "Count"})
                .reset_index()
            )

            x = seed_groups["AbsSeedDiff"]
            width = 0.35

            axs[0, 1].bar(
                x - width / 2,
                seed_groups["Prediction"],
                width,
                label="Predicted",
                color="blue",
                alpha=0.7,
            )
            axs[0, 1].bar(
                x + width / 2,
                seed_groups["Actual"],
                width,
                label="Actual",
                color="green",
                alpha=0.7,
            )

            for i, (_, row) in enumerate(seed_groups.iterrows()):
                axs[0, 1].text(
                    row["AbsSeedDiff"],
                    0.05,
                    f"n={row['Count']}",
                    ha="center",
                    fontsize=8,
                )

            axs[0, 1].set_xlabel("Absolute Seed Difference")
            axs[0, 1].set_ylabel("Win Rate")
            axs[0, 1].set_title("Prediction vs. Actual by Seed Difference")
            axs[0, 1].legend()
            axs[0, 1].grid(True, alpha=0.3)

        # 3. Performance by round
        round_order = [
            "Play-In",
            "Round 1",
            "Round 2",
            "Sweet 16",
            "Elite 8",
            "Final 4",
            "Championship",
        ]
        df["Round"] = pd.Categorical(df["Round"], categories=round_order, ordered=True)

        round_metrics = (
            df.groupby("Round")
            .agg({"Correct": "mean", "Team1ID": "count"})
            .rename(columns={"Correct": "Accuracy", "Team1ID": "Count"})
            .reset_index()
        )

        round_metrics = round_metrics.sort_values("Round")

        bars = axs[1, 0].bar(
            round_metrics["Round"], round_metrics["Accuracy"], color="skyblue"
        )

        # Add counts above bars
        for i, bar in enumerate(bars):
            count = round_metrics.iloc[i]["Count"]
            axs[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"n={count}",
                ha="center",
                fontsize=9,
            )

        axs[1, 0].set_ylim(0, 1.1)
        axs[1, 0].set_xlabel("Tournament Round")
        axs[1, 0].set_ylabel("Prediction Accuracy")
        axs[1, 0].set_title("Accuracy by Tournament Round")
        axs[1, 0].grid(True, alpha=0.3)
        plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha="right")

        # 4. Interesting games (upsets and close calls)
        # Find upsets (higher seed lost) and wrong predictions
        df_with_seeds["HigherSeedWon"] = df_with_seeds["SeedDiff"] < 0
        df_with_seeds["Upset"] = df_with_seeds["SeedDiff"] > 0 & (
            df_with_seeds["Actual"] == 1
        )

        # Filter for upsets or wrong predictions
        interesting_games = df_with_seeds[
            (df_with_seeds["Upset"]) | (~df_with_seeds["Correct"])
        ].copy()

        if len(interesting_games) > 0:
            interesting_games["AbsError"] = np.abs(
                interesting_games["Prediction"] - interesting_games["Actual"]
            )
            interesting_games = interesting_games.sort_values(
                "AbsError", ascending=False
            ).head(10)

            interesting_games["Label"] = interesting_games.apply(
                lambda x: f"{x['Team1Name']} vs {x['Team2Name']} "
                f"({x['Team1Seed']} vs {x['Team2Seed']}) - "
                f"Pred: {x['Prediction']:.2f}, {'✓' if x['Correct'] else '✗'}",
                axis=1,
            )

            y_pos = np.arange(len(interesting_games))
            colors = [
                "green" if correct else "red"
                for correct in interesting_games["Correct"]
            ]

            bars = axs[1, 1].barh(
                y_pos, interesting_games["AbsError"], color=colors, alpha=0.7
            )
            axs[1, 1].set_yticks(y_pos)
            axs[1, 1].set_yticklabels(interesting_games["Label"], fontsize=9)
            axs[1, 1].set_xlabel("Prediction Error")
            axs[1, 1].set_title("Most Interesting Games")
            axs[1, 1].grid(True, alpha=0.3)

        # Title
        fig.suptitle(
            f"Backtest Results for {results['season']} Tournament\n"
            f"Accuracy: {results['accuracy']:.4f}  Brier Score: {results['brier_score']:.4f}",
            fontsize=16,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    def visualize_multiple_backtests(self, results_list):
        """Visualize aggregate results from multiple backtests"""
        # Convert to DataFrame
        seasons = [r["season"] for r in results_list]
        accuracy = [r["accuracy"] for r in results_list]
        brier = [r["brier_score"] for r in results_list]
        log_loss = [r["log_loss"] for r in results_list]

        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))

        # 1. Accuracy by season
        axs[0, 0].plot(seasons, accuracy, "o-", color="blue", linewidth=2)
        axs[0, 0].set_xlabel("Season")
        axs[0, 0].set_ylabel("Accuracy")
        axs[0, 0].set_title("Prediction Accuracy by Season")
        axs[0, 0].grid(True, alpha=0.3)

        # Add mean line
        mean_acc = np.mean(accuracy)
        axs[0, 0].axhline(y=mean_acc, color="red", linestyle="--", alpha=0.7)
        axs[0, 0].text(
            seasons[0], mean_acc + 0.01, f"Mean: {mean_acc:.4f}", color="red"
        )

        # 2. Brier score by season
        axs[0, 1].plot(seasons, brier, "o-", color="green", linewidth=2)
        axs[0, 1].set_xlabel("Season")
        axs[0, 1].set_ylabel("Brier Score")
        axs[0, 1].set_title("Brier Score by Season")
        axs[0, 1].grid(True, alpha=0.3)

        # Add mean line
        mean_brier = np.mean(brier)
        axs[0, 1].axhline(y=mean_brier, color="red", linestyle="--", alpha=0.7)
        axs[0, 1].text(
            seasons[0], mean_brier + 0.01, f"Mean: {mean_brier:.4f}", color="red"
        )

        # Analyze round and seed performance
        self._visualize_round_analysis(axs[1, 0], results_list)
        self._visualize_seed_analysis(axs[1, 1], results_list)

        # Title
        fig.suptitle(
            f"Aggregate Backtest Results for {len(results_list)} Seasons\n"
            f"Average Accuracy: {np.mean(accuracy):.4f}  Average Brier Score: {np.mean(brier):.4f}",
            fontsize=16,
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    def _visualize_round_analysis(self, ax, results_list):
        """Helper method to visualize round analysis"""
        round_order = [
            "Play-In",
            "Round 1",
            "Round 2",
            "Sweet 16",
            "Elite 8",
            "Final 4",
            "Championship",
        ]
        round_data = {}

        # Extract round data from each season
        for result in results_list:
            # Get game details
            game_details = result["game_details"]

            # Get unique games
            unique_games = [game for i, game in enumerate(game_details) if i % 2 == 0]

            # Group by round
            for game in unique_games:
                round_name = game["Round"]
                if round_name not in round_data:
                    round_data[round_name] = {"correct": 0, "total": 0}

                round_data[round_name]["total"] += 1
                if game["Correct"]:
                    round_data[round_name]["correct"] += 1

        # Calculate accuracy by round
        round_accuracy = {
            round_name: data["correct"] / data["total"] if data["total"] > 0 else 0
            for round_name, data in round_data.items()
        }

        # Convert to DataFrame
        round_df = pd.DataFrame(
            {
                "Round": list(round_accuracy.keys()),
                "Accuracy": list(round_accuracy.values()),
                "Games": [round_data[r]["total"] for r in round_accuracy.keys()],
            }
        )

        # Sort by round order
        round_df["Round"] = pd.Categorical(
            round_df["Round"], categories=round_order, ordered=True
        )
        round_df = round_df.sort_values("Round")

        # Plot
        bars = ax.bar(
            round_df["Round"], round_df["Accuracy"], color="purple", alpha=0.7
        )

        # Add counts above bars
        for i, bar in enumerate(bars):
            games = round_df.iloc[i]["Games"]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"n={games}",
                ha="center",
                fontsize=9,
            )

        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Tournament Round")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Tournament Round (All Seasons)")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    def _visualize_seed_analysis(self, ax, results_list):
        """Helper method to visualize seed analysis"""
        seed_data = {}

        # Extract seed data from each season
        for result in results_list:
            # Get game details
            game_details = result["game_details"]

            # Get unique games
            unique_games = [game for i, game in enumerate(game_details) if i % 2 == 0]

            # Filter games with seed info
            seed_games = [
                game for game in unique_games if game.get("SeedDiff") is not None
            ]

            # Group by seed difference
            for game in seed_games:
                seed_diff = abs(game["SeedDiff"])

                if seed_diff not in seed_data:
                    seed_data[seed_diff] = {"correct": 0, "total": 0}

                seed_data[seed_diff]["total"] += 1
                if game["Correct"]:
                    seed_data[seed_diff]["correct"] += 1

        # Calculate accuracy by seed difference
        seed_accuracy = {
            diff: data["correct"] / data["total"] if data["total"] > 0 else 0
            for diff, data in seed_data.items()
        }

        # Convert to DataFrame
        seed_df = pd.DataFrame(
            {
                "SeedDiff": list(seed_accuracy.keys()),
                "Accuracy": list(seed_accuracy.values()),
                "Games": [seed_data[diff]["total"] for diff in seed_accuracy.keys()],
            }
        )

        # Sort by seed difference
        seed_df = seed_df.sort_values("SeedDiff")

        # Plot
        bars = ax.bar(
            seed_df["SeedDiff"], seed_df["Accuracy"], color="orange", alpha=0.7
        )

        # Add counts above bars
        for i, bar in enumerate(bars):
            games = seed_df.iloc[i]["Games"]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"n={games}",
                ha="center",
                fontsize=9,
            )

        ax.set_ylim(0, 1.1)
        ax.set_xlabel("Absolute Seed Difference")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by Seed Difference (All Seasons)")
        ax.grid(True, alpha=0.3)
