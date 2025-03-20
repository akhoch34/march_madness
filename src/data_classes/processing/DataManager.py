import pandas as pd
import numpy as np
from collections import defaultdict


class MarchMadnessDataManager:
    """Class for loading and managing March Madness data"""

    def __init__(self, data_dir, gender="M", current_season=2025):
        """
        Initialize the data manager

        Parameters:
        data_dir (str): Directory containing the data files
        gender (str): 'M' for men's tournament, 'W' for women's
        current_season (int): The current season year
        """
        self.data_dir = data_dir
        self.gender = gender
        self.current_season = current_season
        self.data = {}
        self.seed_lookup = {}  # Store seed lookups by (season, team_id)

    def load_data(self):
        """Load all necessary data files"""
        # Teams data
        all_teams = pd.read_csv(f"{self.data_dir}/{self.gender}Teams.csv")

        # Women's teams don't have the first/last season cols for some reason
        if self.gender == "M":
            self.data["teams"] = all_teams[
                all_teams["LastD1Season"] >= self.current_season
            ]
        else:
            mens_teams = pd.read_csv(f"{self.data_dir}/MTeams.csv")
            self.data["teams"] = all_teams[
                all_teams["TeamName"].isin(
                    mens_teams[mens_teams["LastD1Season"] >= self.current_season][
                        "TeamName"
                    ]
                )
            ]

        # Regular season results
        self.data["regular_season"] = pd.read_csv(
            f"{self.data_dir}/{self.gender}RegularSeasonCompactResults.csv"
        )

        # Tournament results
        self.data["tourney_results"] = pd.read_csv(
            f"{self.data_dir}/{self.gender}NCAATourneyCompactResults.csv"
        )

        # Tournament seeds
        self.data["tourney_seeds"] = pd.read_csv(
            f"{self.data_dir}/{self.gender}NCAATourneySeeds.csv"
        )

        # Tournament seeds
        self.data["tourney_slots"] = pd.read_csv(
            f"{self.data_dir}/{self.gender}NCAATourneySlots.csv"
        )

        # Try to load detailed results if available (for advanced features)
        try:
            self.data["regular_season_detailed"] = pd.read_csv(
                f"{self.data_dir}/{self.gender}RegularSeasonDetailedResults.csv"
            )
            self.detailed_stats_available = True

            # Also load tournament detailed results
            self.data["tourney_detailed"] = pd.read_csv(
                f"{self.data_dir}/{self.gender}NCAATourneyDetailedResults.csv"
            )
        except FileNotFoundError:
            self.detailed_stats_available = False

        # Try to load rankings data if available
        try:
            self.data["rankings"] = pd.read_csv(
                f"{self.data_dir}/{self.gender}MasseyOrdinals.csv"
            )
            self.rankings_available = True
        except FileNotFoundError:
            self.rankings_available = False

        # Load secondary tournament results if available
        try:
            self.data["secondary_tourney"] = pd.read_csv(
                f"{self.data_dir}/{self.gender}SecondaryTourneyCompactResults.csv"
            )
            self.secondary_tourney_available = True
        except FileNotFoundError:
            self.secondary_tourney_available = False

        # Process seeds
        self.preprocess_seeds()

        print(f"Loaded {len(self.data)} datasets")
        return self.data

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

    def get_team_name(self, team_id):
        """Get team name from team ID"""
        team = self.data["teams"][self.data["teams"]["TeamID"] == team_id]
        if len(team) > 0:
            return team.iloc[0]["TeamName"]
        return f"Team {team_id}"

    def get_season_games(self, season, include_tournament=True):
        """Get all games for a specific season"""
        # Get regular season games
        games = self.data["regular_season"][
            self.data["regular_season"]["Season"] == season
        ].copy()

        # Add tournament games if requested
        if include_tournament:
            tourney_games = self.data["tourney_results"][
                self.data["tourney_results"]["Season"] == season
            ].copy()
            games = pd.concat([games, tourney_games])

            # Add secondary tournament games if available
            if (
                hasattr(self, "secondary_tourney_available")
                and self.secondary_tourney_available
            ):
                sec_games = self.data["secondary_tourney"][
                    self.data["secondary_tourney"]["Season"] == season
                ].copy()
                games = pd.concat([games, sec_games])

        # Sort by day number
        games = games.sort_values("DayNum")

        return games

    def get_tournament_teams(self, season, get_all_matchups=False):
        """Get all teams participating in a specific tournament season"""
        if get_all_matchups:
            return self.data['teams']
        return self.data["processed_seeds"][
            self.data["processed_seeds"]["Season"] == season
        ]


    def get_early_season_rankings(self, season):
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

    def get_tournament_round(self, day_num):
        """Determine tournament round from day number"""
        if day_num <= 135:
            return "Play-In"
        elif day_num <= 137:
            return "Round 1"
        elif day_num <= 139:
            return "Round 2"
        elif day_num <= 144:
            return "Sweet 16"
        elif day_num <= 146:
            return "Elite 8"
        elif day_num <= 152:
            return "Final 4"
        else:
            return "Championship"
