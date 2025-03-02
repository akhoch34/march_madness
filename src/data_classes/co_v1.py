import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import defaultdict

YEAR = 2024


class MarchMadnessPredictor:
    def __init__(self, data_dir, gender="M", current_season=2025):
        """
        Initialize the March Madness predictor.

        Parameters:
        data_dir (str): Directory containing the data files
        gender (str): 'M' for men's tournament, 'W' for women's
        current_season (int): The current season year (for prediction)
        """
        self.data_dir = data_dir
        self.gender = gender
        self.current_season = current_season
        self.data = {}
        self.model = None
        self.feature_cols = []
        self.team_elo_ratings = {}  # Store ELO ratings by (season, team_id)

    def load_data(self):
        """Load all necessary data files"""
        # Teams data
        self.data["teams"] = pd.read_csv(f"{self.data_dir}/{self.gender}Teams.csv")

        # Regular season results
        self.data["regular_season"] = pd.read_csv(
            f"{self.data_dir}/{self.gender}RegularSeasonCompactResults.csv"
        )

        # Tournament results (for training)
        self.data["tourney_results"] = pd.read_csv(
            f"{self.data_dir}/{self.gender}NCAATourneyCompactResults.csv"
        )

        # Tournament seeds
        self.data["tourney_seeds"] = pd.read_csv(
            f"{self.data_dir}/{self.gender}NCAATourneySeeds.csv"
        )

        # Try to load detailed results if available (for advanced features)
        try:
            self.data["regular_season_detailed"] = pd.read_csv(
                f"{self.data_dir}/{self.gender}RegularSeasonDetailedResults.csv"
            )
            self.detailed_stats_available = True
        except FileNotFoundError:
            self.detailed_stats_available = False

        # Try to load rankings data if available (men's only)
        if self.gender == "M":
            try:
                self.data["rankings"] = pd.read_csv(
                    f"{self.data_dir}/MMasseyOrdinals.csv"
                )
                self.rankings_available = True
            except FileNotFoundError:
                self.rankings_available = False
        else:
            self.rankings_available = False

        # Load secondary tournament results if available (for more games to train ELO)
        try:
            self.data["secondary_tourney"] = pd.read_csv(
                f"{self.data_dir}/{self.gender}SecondaryTourneyCompactResults.csv"
            )
            self.secondary_tourney_available = True
        except FileNotFoundError:
            self.secondary_tourney_available = False

        print(f"Loaded {len(self.data)} datasets")

    def preprocess_seeds(self):
        """Process tournament seeds to extract region and numeric seed value"""

        # Extract numeric seed from seed string
        def extract_seed_number(seed_str):
            # Remove region identifier and possible play-in indicator
            return int(seed_str[1:3])

        # Process seeds for easier use
        seeds_df = self.data["tourney_seeds"].copy()
        seeds_df["SeedNumber"] = seeds_df["Seed"].apply(extract_seed_number)
        seeds_df["SeedRegion"] = seeds_df["Seed"].str[0]

        # Create a dictionary for quick seed lookup
        seed_dict = {}
        for _, row in seeds_df.iterrows():
            key = (row["Season"], row["TeamID"])
            seed_dict[key] = row["SeedNumber"]

        self.data["processed_seeds"] = seeds_df
        self.seed_lookup = seed_dict

    def calculate_elo_ratings(
        self,
        start_year=2003,
        k_factor=20,
        home_advantage=100,
        carry_over_factor=0.75,
        new_team_rating=1500,
        reset_each_year=False,
    ):
        """
        Calculate ELO ratings for all teams across multiple seasons.

        Parameters:
        start_year (int): First year to calculate ELO ratings for
        k_factor (float): How much each game impacts ELO (higher = more impact)
        home_advantage (float): ELO points added for home court advantage
        carry_over_factor (float): How much of previous season's rating carries over (0-1)
        new_team_rating (float): Default rating for new teams
        reset_each_year (bool): Whether to reset ratings each season (if True, carry_over_factor is ignored)
        """
        print(f"Calculating ELO ratings from {start_year} to {self.current_season}...")

        # Initialize with default ratings
        self.team_elo_ratings = {}

        # Get all regular season games, sorted by season and day
        all_games = pd.concat(
            [self.data["regular_season"], self.data["tourney_results"]]
        )

        # Add secondary tournament games if available
        if self.secondary_tourney_available:
            all_games = pd.concat([all_games, self.data["secondary_tourney"]])

        # Sort by season and day
        all_games = all_games.sort_values(["Season", "DayNum"])

        # Get list of seasons and all teams
        seasons = all_games["Season"].unique()
        seasons.sort()
        seasons = seasons[seasons >= start_year]

        all_teams = set(self.data["teams"]["TeamID"].unique())

        # Process each season
        current_ratings = {team_id: new_team_rating for team_id in all_teams}

        for i, season in enumerate(seasons):
            # Apply carry-over from previous season (or reset)
            if i > 0 and not reset_each_year:
                for team_id in current_ratings:
                    # Regress toward the mean
                    current_ratings[team_id] = new_team_rating + carry_over_factor * (
                        current_ratings[team_id] - new_team_rating
                    )
            else:
                # Reset all ratings
                current_ratings = {team_id: new_team_rating for team_id in all_teams}

            # Apply preseason adjustments based on early rankings if available
            if self.rankings_available:
                early_ranks = self._get_early_season_rankings(season)

                # Map rankings to ELO adjustments
                # Teams in top 25 get a boost, lower ranked teams get smaller adjustment
                for team_id, rank in early_ranks.items():
                    if rank <= 25:
                        # Top 25 teams get bigger boost
                        adjustment = 100 - (rank - 1) * 4  # #1 gets +100, #25 gets +4
                    elif rank <= 100:
                        # Teams 26-100 get small boost
                        adjustment = max(
                            0, 10 - (rank - 25) * 0.1
                        )  # Linear decrease from +10 to 0
                    else:
                        # Teams outside top 100 get small penalty
                        adjustment = min(
                            0, -((rank - 100) * 0.05)
                        )  # Small penalty for very low ranked teams

                    if team_id in current_ratings:
                        current_ratings[team_id] += adjustment

            # Store initial season ratings
            for team_id, rating in current_ratings.items():
                self.team_elo_ratings[(season, team_id, 0)] = rating

            # Process each game in the season
            season_games = all_games[all_games["Season"] == season]

            for _, game in season_games.iterrows():
                w_team = game["WTeamID"]
                l_team = game["LTeamID"]
                day_num = game["DayNum"]
                w_loc = game["WLoc"]

                # Get current ratings
                w_rating = current_ratings.get(w_team, new_team_rating)
                l_rating = current_ratings.get(l_team, new_team_rating)

                # Adjust for home court advantage
                if w_loc == "H":
                    # Winner at home
                    adjusted_w_rating = w_rating + home_advantage
                    adjusted_l_rating = l_rating
                elif w_loc == "A":
                    # Winner away
                    adjusted_w_rating = w_rating
                    adjusted_l_rating = l_rating + home_advantage
                else:
                    # Neutral court
                    adjusted_w_rating = w_rating
                    adjusted_l_rating = l_rating

                # Calculate win probability based on ELO
                win_prob = 1.0 / (
                    1.0 + math.pow(10, (adjusted_l_rating - adjusted_w_rating) / 400.0)
                )

                # Update ratings
                rating_change = k_factor * (1.0 - win_prob)
                current_ratings[w_team] = w_rating + rating_change
                current_ratings[l_team] = l_rating - rating_change

                # Store updated ratings after each game
                self.team_elo_ratings[(season, w_team, day_num)] = current_ratings[
                    w_team
                ]
                self.team_elo_ratings[(season, l_team, day_num)] = current_ratings[
                    l_team
                ]
                print(f"Calculated ELO ratings for {len(seasons)} seasons")
                # Write team_elo_ratings to a readable format
                with open("team_elo_ratings.txt", "w") as f:
                    for (
                        season,
                        team_id,
                        day_num,
                    ), rating in self.team_elo_ratings.items():
                        f.write(
                            f"Season: {season}, Team ID: {team_id}, Day Num: {day_num}, Rating: {rating}\n"
                        )

                    print("team_elo_ratings written to team_elo_ratings.txt")

                return self.team_elo_ratings

    def get_team_elo(self, season, team_id, day_num=None):
        """Get a team's ELO rating for a specific season and day"""
        if day_num is None:
            # If no day specified, get rating before tournament (day 132)
            day_num = 132

        # Find the most recent day with a rating
        while day_num >= 0:
            if (season, team_id, day_num) in self.team_elo_ratings:
                return self.team_elo_ratings[(season, team_id, day_num)]
            day_num -= 1

        # If no rating found, return default
        return 1500

    def elo_win_probability(
        self, team1_elo, team2_elo, home_advantage=100, location=None
    ):
        """Calculate win probability based on ELO ratings"""
        # Adjust for home court if specified
        if location == "H":  # Team1 at home
            team1_elo += home_advantage
        elif location == "A":  # Team1 away
            team2_elo += home_advantage

        # Calculate win probability
        return 1.0 / (1.0 + math.pow(10, (team2_elo - team1_elo) / 400.0))

    def create_feature_dataset(self, train_years_range=(2010, 2024), include_elo=True):
        """
        Create a dataset with features for training and prediction.

        Parameters:
        train_years_range (tuple): Range of years to use for training (inclusive)
        include_elo (bool): Whether to include ELO rating features
        """
        print("Creating feature dataset...")

        # Process seeds first
        self.preprocess_seeds()

        # Calculate ELO ratings if needed
        if include_elo and not self.team_elo_ratings:
            self.calculate_elo_ratings(start_year=min(train_years_range[0] - 2, 2003))

        # Get all possible tournament matchups from historical data
        tourney_games = self.data["tourney_results"].copy()

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
            team1_seed = self.seed_lookup.get(
                (season, team1_id), 16
            )  # Default to 16 if not found
            team2_seed = self.seed_lookup.get((season, team2_id), 16)

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
            if self.rankings_available:
                game_features.update(
                    self._get_ranking_features(season, team1_id, team2_id)
                )

            # Add ELO rating features if available
            if include_elo and self.team_elo_ratings:
                # Get ELO ratings just before this tournament game
                # We use day_num - 1 to ensure we don't leak future information
                team1_elo = self.get_team_elo(season, team1_id, day_num - 1)
                team2_elo = self.get_team_elo(season, team2_id, day_num - 1)

                # Calculate win probability
                elo_win_prob = self.elo_win_probability(team1_elo, team2_elo)

                game_features.update(
                    {
                        "Team1ELO": team1_elo,
                        "Team2ELO": team2_elo,
                        "ELODiff": team1_elo - team2_elo,
                        "ELOWinProb": elo_win_prob,
                    }
                )

            features.append(game_features)

            # Also add the reversed matchup (with opposite result)
            reversed_features = game_features.copy()
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

            # Reverse ELO features if present
            if "Team1ELO" in reversed_features:
                reversed_features["Team1ELO"] = game_features["Team2ELO"]
                reversed_features["Team2ELO"] = game_features["Team1ELO"]
                reversed_features["ELODiff"] = -game_features["ELODiff"]
                reversed_features["ELOWinProb"] = 1.0 - game_features["ELOWinProb"]

            features.append(reversed_features)

        # Create DataFrame with all features
        self.feature_df = pd.DataFrame(features)
        self.feature_df.to_csv("./cache_datasets.csv")
        print(f"Created feature dataset with {len(self.feature_df)} samples")

        # Define feature columns (excluding outcome and identifiers)
        self.feature_cols = [
            col
            for col in self.feature_df.columns
            if col not in ["Result", "Season", "Team1ID", "Team2ID"]
        ]

        return self.feature_df

    def _get_season_stats(self, season, team1_id, team2_id):
        """Get season performance stats for both teams"""
        # Filter regular season games for this season
        season_games = self.data["regular_season"][
            self.data["regular_season"]["Season"] == season
        ]

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

        # If detailed stats are available, we could calculate more advanced metrics here

        return {
            "Team1WinPct": team1_win_pct,
            "Team2WinPct": team2_win_pct,
            "WinPctDiff": team1_win_pct - team2_win_pct,
        }

    def _get_ranking_features(self, season, team1_id, team2_id):
        """Get pre-tournament ranking features for both teams"""
        if not self.rankings_available:
            return {}

        # Get final rankings before tournament (RankingDayNum = 133)
        rankings = self.data["rankings"]
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

    def _get_early_season_rankings(self, season):
        """Get early season rankings (as a proxy for preseason rankings)"""
        if not self.rankings_available:
            return {}

        # Get rankings from early in the season (typically first 2-3 weeks)
        # Using RankingDayNum = 45 (roughly mid-December)
        rankings = self.data["rankings"]
        early_rankings = rankings[
            (rankings["Season"] == season) & (rankings["RankingDayNum"] <= 45)
        ]

        # Take the earliest available ranking for each system and team
        early_rankings = early_rankings.sort_values("RankingDayNum")
        early_rankings = early_rankings.drop_duplicates(
            subset=["Season", "SystemName", "TeamID"], keep="first"
        )

        # Aggregate across systems
        team_ranks = defaultdict(list)
        for _, row in early_rankings.iterrows():
            team_ranks[row["TeamID"]].append(row["OrdinalRank"])

        # Calculate average early ranking for each team
        avg_ranks = {team_id: np.mean(ranks) for team_id, ranks in team_ranks.items()}

        return avg_ranks

    def train_model(self):
        """Train a model using the feature dataset"""
        if not hasattr(self, "feature_df"):
            raise ValueError(
                "Feature dataset not created. Call create_feature_dataset() first."
            )

        # Split features and target
        X = self.feature_df[self.feature_cols]
        y = self.feature_df["Result"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize and train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate
        train_preds = self.model.predict_proba(X_train)[:, 1]
        test_preds = self.model.predict_proba(X_test)[:, 1]

        train_accuracy = accuracy_score(y_train, train_preds > 0.5)
        test_accuracy = accuracy_score(y_test, test_preds > 0.5)

        train_log_loss = log_loss(y_train, train_preds)
        test_log_loss = log_loss(y_test, test_preds)

        print(f"Train accuracy: {train_accuracy:.4f}, Log loss: {train_log_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}, Log loss: {test_log_loss:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame(
            {
                "Feature": self.feature_cols,
                "Importance": self.model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

        print("\nTop 10 important features:")
        print(feature_importance.head(10))

        return self.model

    def generate_predictions(
        self, submission_file="submission.csv", blend_elo=True, elo_weight=0.3
    ):
        """
        Generate predictions for the current tournament

        Parameters:
        submission_file (str): Path to save the submission file
        blend_elo (bool): Whether to blend the model predictions with ELO predictions
        elo_weight (float): Weight to give ELO predictions when blending (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Get current season seeds
        current_seeds = self.data["processed_seeds"][
            self.data["processed_seeds"]["Season"] == self.current_season
        ]

        if len(current_seeds) == 0 and self.current_season < YEAR:
            raise ValueError(f"No seed data found for season {self.current_season}")

        # Ensure we have ELO ratings if blending is requested
        if blend_elo and not self.team_elo_ratings:
            print("ELO ratings not found. Calculating now...")
            self.calculate_elo_ratings()

        # Generate all possible matchups
        team_ids = current_seeds["TeamID"].unique()
        matchups = []

        for i, team1_id in enumerate(team_ids):
            team1_seed = self.seed_lookup.get((self.current_season, team1_id), 16)
            for team2_id in team_ids[i + 1 :]:
                team2_seed = self.seed_lookup.get((self.current_season, team2_id), 16)

                # Create ID in required format
                matchup_id = f"{self.current_season}_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"

                # Prepare features for prediction
                if team1_id < team2_id:
                    features = {
                        "Team1ID": team1_id,
                        "Team2ID": team2_id,
                        "Team1Seed": team1_seed,
                        "Team2Seed": team2_seed,
                        "SeedDiff": team2_seed - team1_seed,
                    }
                    team1_is_first = True
                else:
                    features = {
                        "Team1ID": team2_id,
                        "Team2ID": team1_id,
                        "Team1Seed": team2_seed,
                        "Team2Seed": team1_seed,
                        "SeedDiff": team1_seed - team2_seed,
                    }
                    team1_is_first = False

                # Add season stats
                features.update(
                    self._get_season_stats(
                        self.current_season, features["Team1ID"], features["Team2ID"]
                    )
                )

                # Add ranking features if available
                if self.rankings_available:
                    features.update(
                        self._get_ranking_features(
                            self.current_season,
                            features["Team1ID"],
                            features["Team2ID"],
                        )
                    )

                # Add ELO rating features if available and using ELO in model
                if self.team_elo_ratings:
                    team1_elo = self.get_team_elo(
                        self.current_season, features["Team1ID"]
                    )
                    team2_elo = self.get_team_elo(
                        self.current_season, features["Team2ID"]
                    )
                    elo_win_prob = self.elo_win_probability(team1_elo, team2_elo)

                    if "Team1ELO" in self.feature_cols:
                        features.update(
                            {
                                "Team1ELO": team1_elo,
                                "Team2ELO": team2_elo,
                                "ELODiff": team1_elo - team2_elo,
                                "ELOWinProb": elo_win_prob,
                            }
                        )

                # Extract just the model features
                X_pred = pd.DataFrame(
                    [{col: features.get(col, 0) for col in self.feature_cols}]
                )

                # Get model prediction
                model_pred = self.model.predict_proba(X_pred)[0, 1]

                # Blend with ELO if requested
                if blend_elo and self.team_elo_ratings:
                    # Get ELO-based prediction
                    team1_elo = self.get_team_elo(
                        self.current_season, features["Team1ID"]
                    )
                    team2_elo = self.get_team_elo(
                        self.current_season, features["Team2ID"]
                    )
                    elo_pred = self.elo_win_probability(team1_elo, team2_elo)

                    # Blend predictions
                    pred = (1 - elo_weight) * model_pred + elo_weight * elo_pred
                else:
                    pred = model_pred

                # If team1 is not the first ID in the matchup_id, flip the prediction
                if not team1_is_first:
                    pred = 1 - pred

                matchups.append({"ID": matchup_id, "Pred": pred})

        # Create submission DataFrame
        submission_df = pd.DataFrame(matchups)

        # Save to CSV
        submission_df.to_csv(submission_file, index=False)
        print(f"Saved {len(submission_df)} predictions to {submission_file}")

        return submission_df

    def analyze_team_chances(self, team_id):
        """Analyze a specific team's chances against all other tournament teams"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Get current season seeds
        current_seeds = self.data["processed_seeds"][
            self.data["processed_seeds"]["Season"] == self.current_season
        ]

        if len(current_seeds) == 0:
            raise ValueError(f"No seed data found for season {self.current_season}")

        # Get team info
        team_info = self.data["teams"][self.data["teams"]["TeamID"] == team_id]
        if len(team_info) == 0:
            raise ValueError(f"Team ID {team_id} not found")

        team_name = team_info.iloc[0]["TeamName"]
        team_seed = self.seed_lookup.get((self.current_season, team_id), "Unknown")

        print(f"Analyzing {team_name} (Seed: {team_seed})")

        # Get all other tournament teams
        other_teams = current_seeds[current_seeds["TeamID"] != team_id]

        results = []
        for _, other_team in other_teams.iterrows():
            other_id = other_team["TeamID"]
            other_name = self.data["teams"][
                self.data["teams"]["TeamID"] == other_id
            ].iloc[0]["TeamName"]
            other_seed = other_team["SeedNumber"]

            # Prepare features for prediction (team_id as Team1)
            features = {
                "Team1ID": team_id,
                "Team2ID": other_id,
                "Team1Seed": team_seed if isinstance(team_seed, int) else 16,
                "Team2Seed": other_seed,
                "SeedDiff": other_seed
                - (team_seed if isinstance(team_seed, int) else 16),
            }

            # Add season stats
            features.update(
                self._get_season_stats(self.current_season, team_id, other_id)
            )

            # Add ranking features if available
            if self.rankings_available:
                features.update(
                    self._get_ranking_features(self.current_season, team_id, other_id)
                )

            # Extract just the model features
            X_pred = pd.DataFrame(
                [{col: features.get(col, 0) for col in self.feature_cols}]
            )

            # Get prediction
            win_prob = self.model.predict_proba(X_pred)[0, 1]

            results.append(
                {
                    "OpponentID": other_id,
                    "OpponentName": other_name,
                    "OpponentSeed": other_seed,
                    "WinProbability": win_prob,
                }
            )

        # Create and sort DataFrame
        results_df = pd.DataFrame(results).sort_values(
            "WinProbability", ascending=False
        )

        # Display results
        print(f"\nWin probabilities for {team_name}:")
        print(results_df[["OpponentName", "OpponentSeed", "WinProbability"]].head(10))

        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x="WinProbability",
            y="OpponentName",
            hue="OpponentSeed",
            data=results_df.head(15),
            palette="viridis",
        )
        plt.title(f"Win Probabilities for {team_name}")
        plt.xlabel("Probability")
        plt.ylabel("Opponent")
        plt.tight_layout()
        plt.show()

        return results_df

    # Example usage
    def visualize_elo_history(
        self, team_ids, seasons=None, title="Team ELO Rating History"
    ):
        """
        Visualize ELO rating history for selected teams.

        Parameters:
        team_ids (list): List of TeamIDs to visualize
        seasons (list, optional): List of seasons to include. If None, use all available.
        title (str): Plot title
        """
        if not self.team_elo_ratings:
            raise ValueError(
                "ELO ratings not calculated. Call calculate_elo_ratings() first."
            )

        # Get team names
        team_names = {}
        for team_id in team_ids:
            team_info = self.data["teams"][self.data["teams"]["TeamID"] == team_id]
            if len(team_info) > 0:
                team_names[team_id] = team_info.iloc[0]["TeamName"]
            else:
                team_names[team_id] = f"Team {team_id}"

        # Extract ELO history
        elo_history = defaultdict(list)

        for (season, team_id, day_num), rating in self.team_elo_ratings.items():
            if team_id in team_ids:
                if seasons is None or season in seasons:
                    elo_history[(season, team_id)].append((day_num, rating))

        # Setup plot
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10.colors

        # Plot each team's rating over time
        for i, team_id in enumerate(team_ids):
            color = colors[i % len(colors)]
            team_name = team_names[team_id]

            for season in sorted(
                set(s for (s, t), _ in elo_history.items() if t == team_id)
            ):
                # Get data for this team and season
                data = sorted(elo_history[(season, team_id)])
                if data:
                    days, ratings = zip(*data)

                    # Plot with season label for first point only
                    if i == 0:  # Only label seasons for the first team to avoid clutter
                        plt.plot(
                            days,
                            ratings,
                            "-",
                            color=color,
                            alpha=0.7,
                            linewidth=2,
                            label=f"{season}",
                        )
                    else:
                        plt.plot(
                            days, ratings, "-", color=color, alpha=0.7, linewidth=2
                        )

            # Add a dummy line for the team legend
            plt.plot([], [], "-", color=color, linewidth=3, label=team_name)

        # Add NCAA tournament markers
        plt.axvspan(132, 154, color="lightgray", alpha=0.3, label="NCAA Tournament")

        # Formatting
        plt.xlabel("Day Number")
        plt.ylabel("ELO Rating")
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # Create two legends
        handles, labels = plt.gca().get_legend_handles_labels()

        # Split into teams and seasons
        team_handles = [
            h
            for h, l in zip(handles, labels)
            if not l.isdigit() and l != "NCAA Tournament"
        ]
        team_labels = [l for l in labels if not l.isdigit() and l != "NCAA Tournament"]

        season_handles = [h for h, l in zip(handles, labels) if l.isdigit()]
        season_labels = [l for l in labels if l.isdigit()]

        tournament_handles = [
            h for h, l in zip(handles, labels) if l == "NCAA Tournament"
        ]
        tournament_labels = ["NCAA Tournament"] if tournament_handles else []

        # Place legends
        if team_handles:
            plt.legend(team_handles, team_labels, loc="upper left", title="Teams")

        if season_handles:
            plt.legend(
                season_handles + tournament_handles,
                season_labels + tournament_labels,
                loc="upper right",
                title="Seasons",
            )

        plt.tight_layout()
        plt.show()

    def analyze_elo_factors(
        self, k_values=[10, 20, 30], carry_over_values=[0.5, 0.75, 0.9]
    ):
        """
        Analyze how different ELO parameters affect predictive performance.

        Parameters:
        k_values (list): Different k-factor values to test
        carry_over_values (list): Different season-to-season carryover factors to test
        """
        # Get tournament games for testing
        test_games = self.data["tourney_results"].copy()
        test_games = test_games[test_games["Season"] >= 2015]  # Use recent seasons

        results = []

        for k in k_values:
            for carry_over in carry_over_values:
                # Calculate ELO ratings with these parameters
                self.calculate_elo_ratings(k_factor=k, carry_over_factor=carry_over)

                # Test accuracy on tournament games
                correct = 0
                total = 0

                for _, game in test_games.iterrows():
                    season = game["Season"]
                    w_team = game["WTeamID"]
                    l_team = game["LTeamID"]
                    day_num = game["DayNum"]

                    # Get ELO ratings before the game
                    w_elo = self.get_team_elo(season, w_team, day_num - 1)
                    l_elo = self.get_team_elo(season, l_team, day_num - 1)

                    # Predict winner based on ELO
                    predicted_winner = w_team if w_elo > l_elo else l_team

                    # Check if prediction was correct
                    if predicted_winner == w_team:
                        correct += 1

                    total += 1

                # Calculate accuracy
                accuracy = correct / total if total > 0 else 0

                results.append(
                    {
                        "k_factor": k,
                        "carry_over": carry_over,
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total,
                    }
                )

        # Convert to DataFrame for easy analysis
        results_df = pd.DataFrame(results)

        # Plot results
        plt.figure(figsize=(10, 6))

        # Create pivot table for heatmap
        pivot_data = results_df.pivot(
            index="k_factor", columns="carry_over", values="accuracy"
        )

        # Plot heatmap
        sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="viridis")
        plt.title("Tournament Prediction Accuracy by ELO Parameters")
        plt.xlabel("Carry Over Factor")
        plt.ylabel("K Factor")
        plt.tight_layout()
        plt.show()

        return results_df


if __name__ == "__main__":
    # Initialize predictor
    predictor = MarchMadnessPredictor(
        data_dir=f"../../data/{YEAR}", gender="M", current_season=YEAR
    )

    # Load data
    predictor.load_data()

    # Calculate ELO ratings
    predictor.calculate_elo_ratings(
        start_year=2003, k_factor=20, carry_over_factor=0.75
    )

    # Create features with ELO
    predictor.create_feature_dataset(include_elo=True)

    # Train model
    predictor.train_model()

    # Generate predictions (blending model with ELO)
    predictor.generate_predictions(blend_elo=True, elo_weight=0.3)

    # Analyze ELO parameters to find optimal values
    elo_analysis = predictor.analyze_elo_factors()

    # Visualize ELO history for some top teams
    predictor.visualize_elo_history(
        [1181, 1112, 1246, 1437], seasons=[2023, 2024]  # Duke, Kansas, Kentucky, UNC
    )

    # Analyze a specific team (Duke)
    predictor.analyze_team_chances(1181)  # Duke's TeamID
