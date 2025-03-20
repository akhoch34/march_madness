from .DataManager import MarchMadnessDataManager
import pandas as pd


class TeamStatsCalculator:
    """Class for calculating advanced team statistics"""

    def __init__(self, data_manager: MarchMadnessDataManager):
        """
        Initialize the team stats calculator

        Parameters:
        data_manager: MarchMadnessDataManager instance
        """
        self.data_manager = data_manager
        self.advanced_team_stats = {}

    def calculate_advanced_team_stats(self, start_season=2003):
        """
        Calculate advanced team statistics for all seasons where detailed data is available.
        These include:
        - Offensive/Defensive Efficiency
        - Four Factors (eFG%, TOV%, ORB%, FT Rate)
        - Pace
        - Shooting percentages
        - Advanced possession-based metrics
        """
        if not self.data_manager.detailed_stats_available:
            raise ValueError(
                "Detailed stats not available. Cannot calculate advanced metrics."
            )

        print(
            f"Calculating advanced team stats from {start_season} to {self.data_manager.current_season}..."
        )

        # Get all detailed game results
        all_detailed_games = pd.concat(
            [
                self.data_manager.data["regular_season_detailed"],
                (
                    self.data_manager.data["tourney_detailed"]
                    if "tourney_detailed" in self.data_manager.data
                    else pd.DataFrame()
                ),
            ]
        )

        # Filter for seasons we want
        all_detailed_games = all_detailed_games[
            all_detailed_games["Season"] >= start_season
        ]

        # Sort by season and day
        all_detailed_games = all_detailed_games.sort_values(["Season", "DayNum"])

        # Get list of seasons
        seasons = all_detailed_games["Season"].unique()

        # Get list of teams
        all_teams = self.data_manager.data["teams"]["TeamID"].unique()

        # Dictionary to store advanced stats by season and team
        advanced_stats = {}

        # Process each season
        for season in seasons:
            # Get games for this season
            season_games = all_detailed_games[all_detailed_games["Season"] == season]

            # Dictionary to store team stats for this season
            season_stats = {}

            # Initialize stats for each team
            for team_id in all_teams:
                # Initialize with empty stats dictionary
                season_stats[team_id] = self._create_empty_stats_dict()

            # Process each game
            for _, game in season_games.iterrows():
                # Update stats for both teams
                self._update_team_stats(season_stats, game, is_winner=True)
                self._update_team_stats(season_stats, game, is_winner=False)

            # Calculate advanced stats for each team
            for team_id, stats in season_stats.items():
                # Skip teams with no games
                if stats["Games"] == 0:
                    continue

                # Calculate various advanced stats
                self._calculate_team_advanced_stats(stats)

            # Store stats for this season
            advanced_stats[season] = season_stats

        # Store in class instance
        self.advanced_team_stats = advanced_stats

        print(f"Calculated advanced stats for {len(advanced_stats)} seasons")
        return advanced_stats

    def _create_empty_stats_dict(self):
        """Create an empty stats dictionary for a team"""
        return {
            # Basic counting stats
            "Games": 0,
            "Wins": 0,
            "Losses": 0,
            "Points": 0,
            "PointsAllowed": 0,
            # Shooting stats
            "FGM": 0,
            "FGA": 0,
            "FGM3": 0,
            "FGA3": 0,
            "FTM": 0,
            "FTA": 0,
            # Rebounding
            "OR": 0,
            "DR": 0,
            # Other stats
            "Ast": 0,
            "TO": 0,
            "Stl": 0,
            "Blk": 0,
            "PF": 0,
            # Opponent stats
            "OppFGM": 0,
            "OppFGA": 0,
            "OppFGM3": 0,
            "OppFGA3": 0,
            "OppFTM": 0,
            "OppFTA": 0,
            "OppOR": 0,
            "OppDR": 0,
            "OppAst": 0,
            "OppTO": 0,
            "OppStl": 0,
            "OppBlk": 0,
            "OppPF": 0,
        }

    def _update_team_stats(self, season_stats, game, is_winner):
        """Update team stats based on game data"""
        # Get team IDs and prefixes based on winner/loser
        if is_winner:
            team_id = game["WTeamID"]
            team_prefix = "W"
            opp_prefix = "L"
            if not team_id in season_stats:
                season_stats[team_id] = self._create_empty_stats_dict()
            season_stats[team_id]["Wins"] += 1
        else:
            team_id = game["LTeamID"]
            team_prefix = "L"
            opp_prefix = "W"
            if not team_id in season_stats:
                season_stats[team_id] = self._create_empty_stats_dict()
            season_stats[team_id]["Losses"] += 1

        # Skip if team not in our list
        if team_id not in season_stats:
            return

        # Update basic stats
        season_stats[team_id]["Games"] += 1
        season_stats[team_id]["Points"] += game[f"{team_prefix}Score"]
        season_stats[team_id]["PointsAllowed"] += game[f"{opp_prefix}Score"]

        # Update shooting stats
        season_stats[team_id]["FGM"] += game[f"{team_prefix}FGM"]
        season_stats[team_id]["FGA"] += game[f"{team_prefix}FGA"]
        season_stats[team_id]["FGM3"] += game[f"{team_prefix}FGM3"]
        season_stats[team_id]["FGA3"] += game[f"{team_prefix}FGA3"]
        season_stats[team_id]["FTM"] += game[f"{team_prefix}FTM"]
        season_stats[team_id]["FTA"] += game[f"{team_prefix}FTA"]

        # Update rebounding
        season_stats[team_id]["OR"] += game[f"{team_prefix}OR"]
        season_stats[team_id]["DR"] += game[f"{team_prefix}DR"]

        # Update other stats
        season_stats[team_id]["Ast"] += game[f"{team_prefix}Ast"]
        season_stats[team_id]["TO"] += game[f"{team_prefix}TO"]
        season_stats[team_id]["Stl"] += game[f"{team_prefix}Stl"]
        season_stats[team_id]["Blk"] += game[f"{team_prefix}Blk"]
        season_stats[team_id]["PF"] += game[f"{team_prefix}PF"]

        # Update opponent stats
        season_stats[team_id]["OppFGM"] += game[f"{opp_prefix}FGM"]
        season_stats[team_id]["OppFGA"] += game[f"{opp_prefix}FGA"]
        season_stats[team_id]["OppFGM3"] += game[f"{opp_prefix}FGM3"]
        season_stats[team_id]["OppFGA3"] += game[f"{opp_prefix}FGA3"]
        season_stats[team_id]["OppFTM"] += game[f"{opp_prefix}FTM"]
        season_stats[team_id]["OppFTA"] += game[f"{opp_prefix}FTA"]
        season_stats[team_id]["OppOR"] += game[f"{opp_prefix}OR"]
        season_stats[team_id]["OppDR"] += game[f"{opp_prefix}DR"]
        season_stats[team_id]["OppAst"] += game[f"{opp_prefix}Ast"]
        season_stats[team_id]["OppTO"] += game[f"{opp_prefix}TO"]
        season_stats[team_id]["OppStl"] += game[f"{opp_prefix}Stl"]
        season_stats[team_id]["OppBlk"] += game[f"{opp_prefix}Blk"]
        season_stats[team_id]["OppPF"] += game[f"{opp_prefix}PF"]

    def _calculate_team_advanced_stats(self, stats: dict):
        """Calculate advanced stats for a team based on accumulated basic stats"""
        # Calculate shooting percentages
        stats["FG%"] = stats["FGM"] / stats["FGA"] if stats["FGA"] > 0 else 0
        stats["3P%"] = stats["FGM3"] / stats["FGA3"] if stats["FGA3"] > 0 else 0
        stats["FT%"] = stats["FTM"] / stats["FTA"] if stats["FTA"] > 0 else 0

        # Calculate opponent shooting percentages
        stats["OppFG%"] = (
            stats["OppFGM"] / stats["OppFGA"] if stats["OppFGA"] > 0 else 0
        )
        stats["Opp3P%"] = (
            stats["OppFGM3"] / stats["OppFGA3"] if stats["OppFGA3"] > 0 else 0
        )
        stats["OppFT%"] = (
            stats["OppFTM"] / stats["OppFTA"] if stats["OppFTA"] > 0 else 0
        )

        # Calculate effective field goal percentage (eFG%)
        # eFG% = (FGM + 0.5 * FGM3) / FGA
        stats["eFG%"] = (
            (stats["FGM"] + 0.5 * stats["FGM3"]) / stats["FGA"]
            if stats["FGA"] > 0
            else 0
        )
        stats["OppeFG%"] = (
            (stats["OppFGM"] + 0.5 * stats["OppFGM3"]) / stats["OppFGA"]
            if stats["OppFGA"] > 0
            else 0
        )

        # Estimate possessions (Pace)
        # Possessions = FGA - OR + TO + (0.44 * FTA)
        stats["Poss"] = stats["FGA"] - stats["OR"] + stats["TO"] + (0.44 * stats["FTA"])
        stats["OppPoss"] = (
            stats["OppFGA"] - stats["OppOR"] + stats["OppTO"] + (0.44 * stats["OppFTA"])
        )

        # Average possessions per game (Pace)
        stats["Pace"] = (
            (stats["Poss"] + stats["OppPoss"]) / (2 * stats["Games"])
            if stats["Games"] > 0
            else 0
        )

        # Offensive and Defensive Efficiency (points per 100 possessions)
        stats["OffEff"] = (
            100 * stats["Points"] / stats["Poss"] if stats["Poss"] > 0 else 0
        )
        stats["DefEff"] = (
            100 * stats["PointsAllowed"] / stats["OppPoss"]
            if stats["OppPoss"] > 0
            else 0
        )

        # Net Efficiency
        stats["NetEff"] = stats["OffEff"] - stats["DefEff"]

        # Four Factors
        # 1. Shooting - eFG% (already calculated)
        # 2. Turnovers - Turnover Rate
        stats["TOV%"] = stats["TO"] / stats["Poss"] if stats["Poss"] > 0 else 0
        stats["OppTOV%"] = (
            stats["OppTO"] / stats["OppPoss"] if stats["OppPoss"] > 0 else 0
        )

        # 3. Rebounding - Offensive Rebounding Percentage
        total_rebounds = stats["OR"] + stats["OppDR"]
        opp_total_rebounds = stats["OppOR"] + stats["DR"]

        stats["ORB%"] = stats["OR"] / total_rebounds if total_rebounds > 0 else 0
        stats["DRB%"] = (
            stats["DR"] / opp_total_rebounds if opp_total_rebounds > 0 else 0
        )

        # 4. Free Throws - Free Throw Rate (FTA/FGA)
        stats["FTRate"] = stats["FTA"] / stats["FGA"] if stats["FGA"] > 0 else 0
        stats["OppFTRate"] = (
            stats["OppFTA"] / stats["OppFGA"] if stats["OppFGA"] > 0 else 0
        )

        # Additional metrics
        # Assist Rate (percentage of made field goals that are assisted)
        stats["AstRate"] = stats["Ast"] / stats["FGM"] if stats["FGM"] > 0 else 0

        # Block Rate (percentage of opponent 2-point attempts that are blocked)
        opp_2pa = stats["OppFGA"] - stats["OppFGA3"]
        stats["BlkRate"] = stats["Blk"] / opp_2pa if opp_2pa > 0 else 0

        # Steal Rate (percentage of opponent possessions that end in a steal)
        stats["StlRate"] = (
            stats["Stl"] / stats["OppPoss"] if stats["OppPoss"] > 0 else 0
        )

        # Per-game averages
        for key in [
            "Points",
            "PointsAllowed",
            "FGM",
            "FGA",
            "FGM3",
            "FGA3",
            "FTM",
            "FTA",
            "OR",
            "DR",
            "Ast",
            "TO",
            "Stl",
            "Blk",
            "PF",
        ]:
            stats[key + "PerGame"] = (
                stats[key] / stats["Games"] if stats["Games"] > 0 else 0
            )

    def get_team_stats(self, season, team_id):
        """Get advanced stats for a specific team and season"""
        if season not in self.advanced_team_stats:
            return None

        return self.advanced_team_stats[season].get(team_id, None)

    def get_team_stat_rankings(self, season, stat_name):
        """Get rankings of all teams for a specific stat in a season"""
        if season not in self.advanced_team_stats:
            return None

        teams = []
        values = []

        for team_id, stats in self.advanced_team_stats[season].items():
            if stat_name in stats and stats["Games"] > 0:
                teams.append(team_id)
                values.append(stats[stat_name])

        # Create DataFrame with rankings
        df = pd.DataFrame({"TeamID": teams, "Value": values})

        # Sort and rank
        if len(df) > 0:
            df = df.sort_values("Value", ascending=False)
            df["Rank"] = range(1, len(df) + 1)

        return df
