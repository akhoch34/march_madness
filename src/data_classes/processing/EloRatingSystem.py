from .DataManager import MarchMadnessDataManager
import pandas as pd
import math


class EloRatingSystem:
    """Class for calculating and managing ELO ratings"""

    def __init__(self, data_manager: MarchMadnessDataManager):
        """
        Initialize the ELO rating system

        Parameters:
        data_manager: MarchMadnessDataManager instance
        """
        self.data_manager = data_manager
        self.team_elo_ratings = {}  # Store ELO ratings by (season, team_id, day)

    def calculate_elo_ratings(
        self,
        start_year=2003,
        k_factor=30,
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
        print(
            f"Calculating ELO ratings from {start_year} to {self.data_manager.current_season}..."
        )

        # Initialize with default ratings
        self.team_elo_ratings = {}

        # Get all regular season games, sorted by season and day
        all_games = pd.concat(
            [
                self.data_manager.data["regular_season"],
                self.data_manager.data["tourney_results"],
            ]
        )

        # Add secondary tournament games if available
        if (
            hasattr(self.data_manager, "secondary_tourney_available")
            and self.data_manager.secondary_tourney_available
        ):
            all_games = pd.concat(
                [all_games, self.data_manager.data["secondary_tourney"]]
            )

        # Sort by season and day
        all_games = all_games.sort_values(["Season", "DayNum"])

        # Get list of seasons and all teams
        seasons = all_games["Season"].unique()
        seasons.sort()
        seasons = seasons[seasons >= start_year]

        all_teams = set(self.data_manager.data["teams"]["TeamID"].unique())

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
            if self.data_manager.rankings_available:
                early_ranks = self.data_manager.get_early_season_rankings(season)

                # Map rankings to ELO adjustments
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

    def predict_game(self, team1_id, team2_id, day_num, season, location=None):
        """
        Predict the outcome of a game between team1 and team2

        Parameters:
        team1_id: ID of the first team
        team2_id: ID of the second team
        day_num: Day number of the game (for tournament games)
        season: Season of the game
        location: Game location (H=team1 home, A=team1 away, N=neutral)

        Returns:
        float: Probability of team1 winning
        """
        # Get ELO ratings
        team1_elo = self.get_team_elo(season, team1_id, day_num - 1)
        team2_elo = self.get_team_elo(season, team2_id, day_num - 1)

        # Calculate win probability
        return self.elo_win_probability(team1_elo, team2_elo, location=location)

    def get_all_teams_elo(self, season, day_num=132):
        """Get ELO ratings for all teams at a specific point in time"""
        teams = self.data_manager.data["teams"]["TeamID"].unique()

        results = []
        for team_id in teams:
            elo = self.get_team_elo(season, team_id, day_num)
            team_name = self.data_manager.get_team_name(team_id)

            results.append({"TeamID": team_id, "TeamName": team_name, "ELO": elo})

        return pd.DataFrame(results).sort_values("ELO", ascending=False)
