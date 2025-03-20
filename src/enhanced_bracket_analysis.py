#!/usr/bin/env python3
"""
Enhanced March Madness Tournament Analysis

This module extends the basic bracket_analysis.py functionality to provide:
1. Detailed team profiles with advanced stats and ELO history
2. Enhanced matchup analysis with deeper statistical insights
3. Executive summaries for regions and overall tournament

It can be used as a standalone script or imported into a notebook to work with an existing predictor.
"""

import os
import sys
import re
import argparse
import pandas as pd
from datetime import datetime

# Import the original bracket analysis module
from bracket_analysis import (
    SimplePredictor,
    generate_bracket_analysis,
    get_tournament_structure,
)

from data_classes.processing import (
    MarchMadnessDataManager,
    EloRatingSystem,
    TeamStatsCalculator,
    MarchMadnessMLModel,
)

# Constants
SUBMISSION_DIR = "./output"
DATA_DIR = "../data/{year}"
OUTPUT_DIR = "./output"


class EnhancedPredictor(SimplePredictor):
    """Enhanced predictor with more detailed analysis capabilities"""

    def __init__(
        self,
        teams_df,
        seeds_df,
        slots_df,
        predictions_df,
        elo_df=None,
        stats_df=None,
        feature_df=None,
        full_elo_history_df=None,
        current_season=2025,
        ml_model: MarchMadnessMLModel = None,  # Add this parameter
    ):
        super().__init__(
            teams_df,
            seeds_df,
            slots_df,
            predictions_df,
            elo_df,
            stats_df,
            current_season,
        )
        self.feature_df = feature_df
        self.full_elo_history_df = full_elo_history_df
        self.team_profiles = {}  # Cache for generated team profiles
        self.ml_model = ml_model  # Store the ML model for on-demand feature generation

    def generate_team_profile(self, team_id, output_file=None):
        """
        Generate a comprehensive profile of a team with advanced stats and ELO history
        using on-demand feature generation for the most recent data
        """
        # Check if we've already generated this profile
        if team_id in self.team_profiles:
            profile = self.team_profiles[team_id]
            # Also generate the markdown for cached profiles
            markdown_text = self._generate_team_profile_markdown(profile)
            return profile, markdown_text

        # Get basic team info
        team_name = self.data_manager.get_team_name(team_id)
        team_seed = self.data_manager.get_seed(self.current_season, team_id)

        # Get ELO rating and history
        team_elo = self.get_elo_ratings(team_id, team_id)[0]  # Just get the first value

        # Extract team's ELO history if available
        elo_history = None
        elo_trend = None
        if self.full_elo_history_df is not None:
            elo_history = self.full_elo_history_df[
                (self.full_elo_history_df["TeamID"] == team_id)
                & (self.full_elo_history_df["Season"] == self.current_season)
            ].sort_values("DayNum")

            if len(elo_history) > 0:
                # Calculate ELO trend over last 10 games
                recent_elo = elo_history.tail(10)
                if len(recent_elo) >= 5:  # Need at least 5 data points
                    elo_start = recent_elo.iloc[0]["ELO"]
                    elo_end = recent_elo.iloc[-1]["ELO"]
                    elo_trend = elo_end - elo_start

        # Generate fresh features and extract advanced stats using the ML model
        # Use a dummy opponent (against itself) to get team-specific metrics
        day_num = None  # Use the latest day available
        advanced_stats = {}
        performance_metrics = {}

        if self.ml_model is not None:
            # Get on-demand features for the most recent data
            feature_df = self.ml_model.get_matchup_features(
                team_id, team_id, self.current_season, day_num
            )

            if len(feature_df) > 0:
                # Extract advanced stats
                stats = {}
                row = feature_df.iloc[0]

                # Extract prefixed stats
                for col in row.index:
                    if col.startswith("Team1_"):
                        stat_name = col.replace("Team1_", "")
                        stats[stat_name] = row[col]

                # Add efficiency metrics
                if "Team1OffEff" in row:
                    stats["OffEff"] = row["Team1OffEff"]
                if "Team1DefEff" in row:
                    stats["DefEff"] = row["Team1DefEff"]
                if "Team1NetEff" in row:
                    stats["NetEff"] = row["Team1NetEff"]

                # Add standard stats
                for stat in [
                    "Pace",
                    "eFG%",
                    "TOV%",
                    "ORB%",
                    "FTRate",
                    "FG%",
                    "3P%",
                    "FT%",
                ]:
                    stat_col = f"Team1_{stat}"
                    if stat_col in row:
                        stats[stat] = row[stat_col]

                advanced_stats = stats

                # Extract performance metrics
                performance_metrics = {
                    "WinPct": row["Team1WinPct"] if "Team1WinPct" in row else None,
                    "Last10": row["Team1Last10"] if "Team1Last10" in row else None,
                    "RecentMargin": (
                        row["Team1RecentMargin"] if "Team1RecentMargin" in row else None
                    ),
                    "Streak": row["Team1Streak"] if "Team1Streak" in row else None,
                    "AvgRank": row["Team1AvgRank"] if "Team1AvgRank" in row else None,
                    "SOS": row["Team1SOS"] if "Team1SOS" in row else None,
                }

        # Fall back to pre-calculated stats if on-demand generation fails
        if (
            not advanced_stats
            and self.stats_df is not None
            and team_id in self.stats_df
        ):
            advanced_stats = self.stats_df[team_id]

        # Fall back to pre-calculated performance metrics if needed
        if not performance_metrics and self.feature_df is not None:
            team1_games = self.feature_df[self.feature_df["Team1ID"] == team_id]
            if len(team1_games) > 0:
                performance_metrics.update(
                    {
                        "WinPct": (
                            team1_games["Team1WinPct"].iloc[0]
                            if "Team1WinPct" in team1_games.columns
                            else None
                        ),
                        "Last10": (
                            team1_games["Team1Last10"].iloc[0]
                            if "Team1Last10" in team1_games.columns
                            else None
                        ),
                        "RecentMargin": (
                            team1_games["Team1RecentMargin"].iloc[0]
                            if "Team1RecentMargin" in team1_games.columns
                            else None
                        ),
                        "Streak": (
                            team1_games["Team1Streak"].iloc[0]
                            if "Team1Streak" in team1_games.columns
                            else None
                        ),
                        "AvgRank": (
                            team1_games["Team1AvgRank"].iloc[0]
                            if "Team1AvgRank" in team1_games.columns
                            else None
                        ),
                        "SOS": (
                            team1_games["Team1SOS"].iloc[0]
                            if "Team1SOS" in team1_games.columns
                            else None
                        ),
                    }
                )

        # Compile team strengths and weaknesses based on stats
        strengths = []
        weaknesses = []

        if advanced_stats:
            # Check offensive efficiency
            if "OffEff" in advanced_stats and advanced_stats["OffEff"] > 110:
                strengths.append(
                    f"Strong offensive efficiency ({advanced_stats['OffEff']:.1f})"
                )
            elif "OffEff" in advanced_stats and advanced_stats["OffEff"] < 100:
                weaknesses.append(
                    f"Struggles offensively ({advanced_stats['OffEff']:.1f} efficiency)"
                )

            # Check defensive efficiency
            if "DefEff" in advanced_stats and advanced_stats["DefEff"] < 95:
                strengths.append(
                    f"Elite defense ({advanced_stats['DefEff']:.1f} points allowed per 100 possessions)"
                )
            elif "DefEff" in advanced_stats and advanced_stats["DefEff"] > 105:
                weaknesses.append(
                    f"Poor defensive efficiency ({advanced_stats['DefEff']:.1f})"
                )

            # Check shooting
            if "3P%" in advanced_stats and advanced_stats["3P%"] > 0.37:
                strengths.append(
                    f"Excellent 3-point shooting ({advanced_stats['3P%']*100:.1f}%)"
                )
            elif "3P%" in advanced_stats and advanced_stats["3P%"] < 0.32:
                weaknesses.append(
                    f"Poor 3-point shooting ({advanced_stats['3P%']*100:.1f}%)"
                )

            # Check turnovers
            if "TOV%" in advanced_stats and advanced_stats["TOV%"] < 0.16:
                strengths.append(
                    f"Takes care of the ball (only {advanced_stats['TOV%']*100:.1f}% turnover rate)"
                )
            elif "TOV%" in advanced_stats and advanced_stats["TOV%"] > 0.21:
                weaknesses.append(
                    f"Turnover prone ({advanced_stats['TOV%']*100:.1f}% of possessions)"
                )

            # Check rebounding
            if "ORB%" in advanced_stats and advanced_stats["ORB%"] > 0.33:
                strengths.append(
                    f"Strong offensive rebounding ({advanced_stats['ORB%']*100:.1f}%)"
                )
            elif "ORB%" in advanced_stats and advanced_stats["ORB%"] < 0.25:
                weaknesses.append(
                    f"Struggles on the offensive glass ({advanced_stats['ORB%']*100:.1f}%)"
                )

            # Check pace
            if "Pace" in advanced_stats and advanced_stats["Pace"] > 72:
                strengths.append(
                    f"Plays at a fast tempo ({advanced_stats['Pace']:.1f} possessions/game)"
                )
            elif "Pace" in advanced_stats and advanced_stats["Pace"] < 65:
                strengths.append(
                    f"Controls tempo ({advanced_stats['Pace']:.1f} possessions/game)"
                )

        # Add momentum-based strengths/weaknesses
        if (
            "Streak" in performance_metrics
            and performance_metrics["Streak"] is not None
        ):
            if performance_metrics["Streak"] >= 5:
                strengths.append(
                    f"Hot team with a {performance_metrics['Streak']}-game winning streak"
                )
            elif performance_metrics["Streak"] <= -3:
                weaknesses.append(
                    f"Cold team with a {abs(performance_metrics['Streak'])}-game losing streak"
                )

        if elo_trend is not None:
            if elo_trend > 50:
                strengths.append(
                    f"Team is trending up (+{elo_trend:.0f} ELO points in recent games)"
                )
            elif elo_trend < -50:
                weaknesses.append(
                    f"Team is trending down ({elo_trend:.0f} ELO points in recent games)"
                )

        if "SOS" in performance_metrics and performance_metrics["SOS"] is not None:
            if performance_metrics["SOS"] > 5:
                strengths.append(
                    f"Battle-tested with a strong schedule (SOS: {performance_metrics['SOS']:.1f})"
                )
            elif performance_metrics["SOS"] < -5:
                weaknesses.append(
                    f"Played a weak schedule (SOS: {performance_metrics['SOS']:.1f})"
                )

        # Create the team profile dictionary
        profile = {
            "TeamID": team_id,
            "TeamName": team_name,
            "Seed": team_seed,
            "ELO": team_elo,
            "ELOTrend": elo_trend,
            "AdvancedStats": advanced_stats,
            "PerformanceMetrics": performance_metrics,
            "Strengths": strengths,
            "Weaknesses": weaknesses,
        }

        # Cache the profile
        self.team_profiles[team_id] = profile

        # Generate markdown text
        markdown_text = self._generate_team_profile_markdown(profile)

        # Save to file if requested
        if output_file:
            with open(output_file, "w") as f:
                f.write(markdown_text)
            print(f"Team profile saved to {output_file}")

        return profile, markdown_text

    def _generate_team_profile_markdown(self, profile):
        """Generate markdown text for a team profile"""
        markdown = []

        # Header with seed and team name
        seed_display = profile["Seed"] if profile["Seed"] else "UNK"
        markdown.append(f"# {seed_display} {profile['TeamName']} Team Profile\n")

        # Basic info section
        markdown.append("## Team Overview\n")
        markdown.append(f"**Seed:** {profile['Seed']}\n")
        markdown.append(f"**ELO Rating:** {profile['ELO']:.0f}\n")

        if "PerformanceMetrics" in profile and profile["PerformanceMetrics"]:
            metrics = profile["PerformanceMetrics"]
            if "WinPct" in metrics and metrics["WinPct"] is not None:
                markdown.append(f"**Win Percentage:** {metrics['WinPct']:.1%}\n")
            if "AvgRank" in metrics and metrics["AvgRank"] is not None:
                markdown.append(f"**Average Ranking:** {metrics['AvgRank']:.1f}\n")
            if "Last10" in metrics and metrics["Last10"] is not None:
                markdown.append(
                    f"**Last 10 Games:** {metrics['Last10']*10:.0f}-{10-metrics['Last10']*10:.0f}\n"
                )
            if "Streak" in metrics and metrics["Streak"] is not None:
                streak_type = "Win" if metrics["Streak"] > 0 else "Loss"
                streak_val = abs(metrics["Streak"])
                markdown.append(
                    f"**Current Streak:** {streak_val}-game {streak_type} streak\n"
                )
            if "SOS" in metrics and metrics["SOS"] is not None:
                sos_rating = (
                    "Strong"
                    if metrics["SOS"] > 3
                    else "Average" if metrics["SOS"] > -3 else "Weak"
                )
                markdown.append(
                    f"**Strength of Schedule:** {sos_rating} ({metrics['SOS']:.1f})\n"
                )
            if "RecentMargin" in metrics and metrics["RecentMargin"] is not None:
                markdown.append(
                    f"**Recent Point Margin:** {metrics['RecentMargin']:.1f} points\n"
                )

        markdown.append("\n")

        # Advanced metrics section
        if "AdvancedStats" in profile and profile["AdvancedStats"]:
            markdown.append("## Advanced Metrics\n")

            # Create a table for the Four Factors
            markdown.append("### Four Factors\n")
            markdown.append("| Factor | Value | Percentile |\n")
            markdown.append("|--------|-------|------------|\n")

            stats = profile["AdvancedStats"]

            # Estimated percentiles for the Four Factors
            # These are rough estimates based on typical college basketball distributions
            def get_percentile(stat, value):
                if stat == "eFG%":
                    return max(
                        0, min(100, (value - 0.45) * 500)
                    )  # 45% is around average
                elif stat == "TOV%":
                    return max(
                        0, min(100, (0.22 - value) * 500)
                    )  # Lower is better, 22% is average
                elif stat == "ORB%":
                    return max(
                        0, min(100, (value - 0.25) * 500)
                    )  # 25% is around average
                elif stat == "FTRate":
                    return max(
                        0, min(100, (value - 0.25) * 500)
                    )  # 25% is around average
                return 50  # Default to average

            # Add Four Factors
            for factor, label in [
                ("eFG%", "Effective FG%"),
                ("TOV%", "Turnover %"),
                ("ORB%", "Off. Rebounding %"),
                ("FTRate", "FT Rate"),
            ]:
                if factor in stats:
                    value = stats[factor]
                    percentile = get_percentile(factor, value)

                    # Format properly
                    if factor in ["eFG%", "TOV%", "ORB%"]:
                        formatted_value = f"{value*100:.1f}%"
                    else:
                        formatted_value = f"{value:.3f}"

                    # Add a symbol to indicate if this is good or bad
                    rating = (
                        "🔴" if percentile < 30 else "🟡" if percentile < 70 else "🟢"
                    )
                    markdown.append(
                        f"| {label} | {formatted_value} | {percentile:.0f}% {rating} |\n"
                    )

            markdown.append("\n")

            # Offensive and defensive efficiency
            markdown.append("### Efficiency Ratings\n")
            markdown.append("| Metric | Value | Percentile |\n")
            markdown.append("|--------|-------|------------|\n")

            # Add efficiency metrics
            for metric, label in [
                ("OffEff", "Offensive Efficiency"),
                ("DefEff", "Defensive Efficiency"),
                ("NetEff", "Net Efficiency"),
            ]:
                if metric in stats:
                    value = stats[metric]

                    # Estimate percentile (rough approximation)
                    if metric == "OffEff":
                        percentile = max(
                            0, min(100, (value - 95) * 5)
                        )  # 95 is poor, 115 is elite
                        rating = (
                            "🔴"
                            if percentile < 30
                            else "🟡" if percentile < 70 else "🟢"
                        )
                    elif metric == "DefEff":
                        percentile = max(
                            0, min(100, (105 - value) * 5)
                        )  # Lower is better for defense
                        rating = (
                            "🔴"
                            if percentile < 30
                            else "🟡" if percentile < 70 else "🟢"
                        )
                    elif metric == "NetEff":
                        percentile = max(
                            0, min(100, (value + 10) * 5)
                        )  # -10 is poor, +10 is elite
                        rating = (
                            "🔴"
                            if percentile < 30
                            else "🟡" if percentile < 70 else "🟢"
                        )
                    else:
                        percentile = 50
                        rating = "🟡"

                    markdown.append(
                        f"| {label} | {value:.1f} | {percentile:.0f}% {rating} |\n"
                    )

            markdown.append("\n")

            # Additional shooting metrics
            markdown.append("### Shooting Metrics\n")
            markdown.append("| Metric | Value | Percentile |\n")
            markdown.append("|--------|-------|------------|\n")

            for metric, label in [
                ("FG%", "Field Goal %"),
                ("3P%", "3-Point %"),
                ("FT%", "Free Throw %"),
            ]:
                if metric in stats:
                    value = stats[metric]

                    # Estimate percentile
                    if metric == "FG%":
                        percentile = max(0, min(100, (value - 0.4) * 300))
                    elif metric == "3P%":
                        percentile = max(0, min(100, (value - 0.3) * 500))
                    elif metric == "FT%":
                        percentile = max(0, min(100, (value - 0.65) * 300))
                    else:
                        percentile = 50

                    rating = (
                        "🔴" if percentile < 30 else "🟡" if percentile < 70 else "🟢"
                    )
                    markdown.append(
                        f"| {label} | {value*100:.1f}% | {percentile:.0f}% {rating} |\n"
                    )

            markdown.append("\n")

            # Tempo and style metrics if available
            if "Pace" in stats:
                markdown.append("### Style Metrics\n")

                pace = stats["Pace"]
                pace_percentile = max(
                    0, min(100, (pace - 60) * 10)
                )  # 60 to 70 is the typical range
                pace_style = (
                    "Very Slow"
                    if pace < 63
                    else (
                        "Slow"
                        if pace < 67
                        else (
                            "Average"
                            if pace < 71
                            else "Fast" if pace < 75 else "Very Fast"
                        )
                    )
                )
                markdown.append(
                    f"**Tempo:** {pace_style} ({pace:.1f} possessions per game, {pace_percentile:.0f}th percentile)\n\n"
                )

        # Team strengths and weaknesses
        if "Strengths" in profile and profile["Strengths"]:
            markdown.append("## Team Strengths\n")
            for strength in profile["Strengths"]:
                markdown.append(f"* {strength}\n")
            markdown.append("\n")

        if "Weaknesses" in profile and profile["Weaknesses"]:
            markdown.append("## Team Weaknesses\n")
            for weakness in profile["Weaknesses"]:
                markdown.append(f"* {weakness}\n")
            markdown.append("\n")

        # ELO history if available
        if profile["ELOTrend"] is not None:
            trend_direction = "upward" if profile["ELOTrend"] > 0 else "downward"
            trend_magnitude = "strong" if abs(profile["ELOTrend"]) > 50 else "slight"
            markdown.append(f"## ELO Trend\n")
            markdown.append(
                f"Team shows a {trend_magnitude} {trend_direction} trend in recent games ({profile['ELOTrend']:.1f} point change).\n\n"
            )

        return "".join(markdown)

    def generate_enhanced_matchup_analysis(self, team1_id, team2_id, output_file=None):
        """
        Generate an enhanced analysis for a specific matchup

        Parameters:
        team1_id: ID of the first team
        team2_id: ID of the second team
        output_file: Optional file path to save the analysis

        Returns:
        dict: Matchup analysis data
        str: Markdown text of the analysis
        """
        # Get team names and seeds
        team1_name = self.data_manager.get_team_name(team1_id)
        team2_name = self.data_manager.get_team_name(team2_id)
        team1_seed = self.data_manager.get_seed(self.current_season, team1_id)
        team2_seed = self.data_manager.get_seed(self.current_season, team2_id)

        # Get prediction
        win_prob = self.predict_game(team1_id, team2_id)

        # Get ELO ratings
        team1_elo, team2_elo = self.get_elo_ratings(team1_id, team2_id)

        # Calculate betting odds
        american_odds = self.american_odds(win_prob)
        spread = self.win_probability_to_spread(win_prob)
        formatted_spread = self.format_spread(spread)

        # Get team profiles
        team1_profile, _ = self.generate_team_profile(team1_id)
        team2_profile, _ = self.generate_team_profile(team2_id)

        # Generate matchup explanation
        explanation = self.generate_matchup_explanation(team1_id, team2_id)

        # Determine the keys stats that will decide this game
        key_matchup_factors = self._identify_key_matchup_factors(
            team1_profile, team2_profile
        )

        # Create statistical comparison table data
        stat_comparison = self._generate_stat_comparison(team1_profile, team2_profile)

        # Create the matchup analysis dictionary
        analysis = {
            "Team1ID": team1_id,
            "Team2ID": team2_id,
            "Team1Name": team1_name,
            "Team2Name": team2_name,
            "Team1Seed": team1_seed,
            "Team2Seed": team2_seed,
            "Team1ELO": team1_elo,
            "Team2ELO": team2_elo,
            "WinProbability": win_prob,
            "Spread": spread,
            "FormattedSpread": formatted_spread,
            "AmericanOdds": american_odds,
            "Explanation": explanation,
            "KeyMatchupFactors": key_matchup_factors,
            "StatComparison": stat_comparison,
        }

        # Generate markdown text
        markdown_text = self._generate_matchup_analysis_markdown(analysis)

        # Save to file if requested
        if output_file:
            with open(output_file, "w") as f:
                f.write(markdown_text)
            print(f"Matchup analysis saved to {output_file}")

        return analysis, markdown_text

    def _identify_key_matchup_factors(self, team1_profile, team2_profile):
        """Identify the key statistical factors that will determine this matchup"""
        key_factors = []

        # Extract advanced stats for both teams
        team1_stats = team1_profile.get("AdvancedStats", {})
        team2_stats = team2_profile.get("AdvancedStats", {})

        if not team1_stats or not team2_stats:
            return key_factors

        # Compare offensive vs defensive efficiency
        if "OffEff" in team1_stats and "DefEff" in team2_stats:
            off_vs_def_diff = team1_stats["OffEff"] - team2_stats["DefEff"]
            if off_vs_def_diff > 10:
                key_factors.append(
                    {
                        "factor": "Offense vs Defense",
                        "description": f"{team1_profile['TeamName']}'s offense significantly outpaces {team2_profile['TeamName']}'s defense (+{off_vs_def_diff:.1f} efficiency gap).",
                        "advantage": "Team1",
                    }
                )
            elif off_vs_def_diff < -10:
                key_factors.append(
                    {
                        "factor": "Offense vs Defense",
                        "description": f"{team2_profile['TeamName']}'s defense should stifle {team1_profile['TeamName']}'s offense (-{abs(off_vs_def_diff):.1f} efficiency gap).",
                        "advantage": "Team2",
                    }
                )

        # Now check the reverse (team2 offense vs team1 defense)
        if "OffEff" in team2_stats and "DefEff" in team1_stats:
            off_vs_def_diff = team2_stats["OffEff"] - team1_stats["DefEff"]
            if off_vs_def_diff > 10:
                key_factors.append(
                    {
                        "factor": "Offense vs Defense",
                        "description": f"{team2_profile['TeamName']}'s offense significantly outpaces {team1_profile['TeamName']}'s defense (+{off_vs_def_diff:.1f} efficiency gap).",
                        "advantage": "Team2",
                    }
                )
            elif off_vs_def_diff < -10:
                key_factors.append(
                    {
                        "factor": "Offense vs Defense",
                        "description": f"{team1_profile['TeamName']}'s defense should stifle {team2_profile['TeamName']}'s offense (-{abs(off_vs_def_diff):.1f} efficiency gap).",
                        "advantage": "Team1",
                    }
                )

        # Check for 3-point shooting advantages
        if "3P%" in team1_stats and "3P%" in team2_stats:
            three_pt_diff = team1_stats["3P%"] - team2_stats["3P%"]
            if abs(three_pt_diff) > 0.05:  # 5% difference is significant
                better_team = "Team1" if three_pt_diff > 0 else "Team2"
                better_name = (
                    team1_profile["TeamName"]
                    if better_team == "Team1"
                    else team2_profile["TeamName"]
                )
                worse_name = (
                    team2_profile["TeamName"]
                    if better_team == "Team1"
                    else team1_profile["TeamName"]
                )
                pct_diff = abs(three_pt_diff) * 100
                key_factors.append(
                    {
                        "factor": "3-Point Shooting",
                        "description": f"{better_name} has a significant advantage in 3-point shooting (+{pct_diff:.1f}% better than {worse_name}).",
                        "advantage": better_team,
                    }
                )

        # Check for turnover advantages
        if "TOV%" in team1_stats and "TOV%" in team2_stats:
            tov_diff = team2_stats["TOV%"] - team1_stats["TOV%"]  # Lower is better
            if abs(tov_diff) > 0.04:  # 4% difference is significant
                better_team = "Team1" if tov_diff > 0 else "Team2"
                better_name = (
                    team1_profile["TeamName"]
                    if better_team == "Team1"
                    else team2_profile["TeamName"]
                )
                worse_name = (
                    team2_profile["TeamName"]
                    if better_team == "Team1"
                    else team1_profile["TeamName"]
                )
                pct_diff = abs(tov_diff) * 100
                key_factors.append(
                    {
                        "factor": "Ball Security",
                        "description": f"{better_name} takes better care of the ball ({pct_diff:.1f}% lower turnover rate than {worse_name}).",
                        "advantage": better_team,
                    }
                )

        # Check for rebounding advantages
        if "ORB%" in team1_stats and "ORB%" in team2_stats:
            orb_diff = team1_stats["ORB%"] - team2_stats["ORB%"]
            if abs(orb_diff) > 0.06:  # 6% difference is significant
                better_team = "Team1" if orb_diff > 0 else "Team2"
                better_name = (
                    team1_profile["TeamName"]
                    if better_team == "Team1"
                    else team2_profile["TeamName"]
                )
                worse_name = (
                    team2_profile["TeamName"]
                    if better_team == "Team1"
                    else team1_profile["TeamName"]
                )
                pct_diff = abs(orb_diff) * 100
                key_factors.append(
                    {
                        "factor": "Rebounding",
                        "description": f"{better_name} should dominate the offensive glass (+{pct_diff:.1f}% rebounding advantage over {worse_name}).",
                        "advantage": better_team,
                    }
                )

        # Check for tempo mismatches (fast vs slow teams)
        if "Pace" in team1_stats and "Pace" in team2_stats:
            pace_diff = team1_stats["Pace"] - team2_stats["Pace"]
            if abs(pace_diff) > 5:  # 5 possession difference is significant
                faster_team = "Team1" if pace_diff > 0 else "Team2"
                slower_team = "Team2" if pace_diff > 0 else "Team1"
                faster_name = (
                    team1_profile["TeamName"]
                    if faster_team == "Team1"
                    else team2_profile["TeamName"]
                )
                slower_name = (
                    team2_profile["TeamName"]
                    if faster_team == "Team1"
                    else team1_profile["TeamName"]
                )
                key_factors.append(
                    {
                        "factor": "Tempo Control",
                        "description": f"Significant tempo mismatch: {faster_name} wants to play fast ({team1_stats['Pace']:.1f} possessions) while {slower_name} prefers a slower pace ({team2_stats['Pace']:.1f} possessions).",
                        "advantage": "Neutral",  # Tempo itself doesn't give an advantage, just creates a clash
                    }
                )

        # Check for free throw rate advantages
        if "FTRate" in team1_stats and "FTRate" in team2_stats:
            ft_diff = team1_stats["FTRate"] - team2_stats["FTRate"]
            if abs(ft_diff) > 0.1:  # 10% difference is significant
                better_team = "Team1" if ft_diff > 0 else "Team2"
                better_name = (
                    team1_profile["TeamName"]
                    if better_team == "Team1"
                    else team2_profile["TeamName"]
                )
                worse_name = (
                    team2_profile["TeamName"]
                    if better_team == "Team1"
                    else team1_profile["TeamName"]
                )
                key_factors.append(
                    {
                        "factor": "Free Throw Rate",
                        "description": f"{better_name} gets to the line much more frequently than {worse_name}, which could be crucial in a close game.",
                        "advantage": better_team,
                    }
                )

        # Check ELO momentum trends
        if "ELOTrend" in team1_profile and "ELOTrend" in team2_profile:
            if (
                team1_profile["ELOTrend"] is not None
                and team2_profile["ELOTrend"] is not None
            ):
                elo_trend_diff = team1_profile["ELOTrend"] - team2_profile["ELOTrend"]
                if abs(elo_trend_diff) > 50:
                    better_team = "Team1" if elo_trend_diff > 0 else "Team2"
                    better_name = (
                        team1_profile["TeamName"]
                        if better_team == "Team1"
                        else team2_profile["TeamName"]
                    )
                    worse_name = (
                        team2_profile["TeamName"]
                        if better_team == "Team1"
                        else team1_profile["TeamName"]
                    )
                    key_factors.append(
                        {
                            "factor": "Momentum",
                            "description": f"{better_name} has much better momentum coming into the tournament compared to {worse_name}.",
                            "advantage": better_team,
                        }
                    )

        return key_factors

    def _generate_stat_comparison(self, team1_profile, team2_profile):
        """Generate statistical comparison between teams"""
        comparison = {}

        team1_stats = team1_profile.get("AdvancedStats", {})
        team2_stats = team2_profile.get("AdvancedStats", {})

        if not team1_stats or not team2_stats:
            return comparison

        # Four Factors comparison
        four_factors = {}
        for factor in ["eFG%", "TOV%", "ORB%", "FTRate"]:
            if factor in team1_stats and factor in team2_stats:
                team1_val = team1_stats[factor]
                team2_val = team2_stats[factor]

                # For TOV%, lower is better, so invert the advantage
                if factor == "TOV%":
                    advantage = (
                        "Team1"
                        if team1_val < team2_val
                        else "Team2" if team2_val < team1_val else "Even"
                    )
                else:
                    advantage = (
                        "Team1"
                        if team1_val > team2_val
                        else "Team2" if team2_val > team1_val else "Even"
                    )

                four_factors[factor] = {
                    "Team1": team1_val,
                    "Team2": team2_val,
                    "Difference": team1_val - team2_val,
                    "Advantage": advantage,
                }

        comparison["FourFactors"] = four_factors

        # Efficiency metrics comparison
        efficiency = {}
        for metric in ["OffEff", "DefEff", "NetEff"]:
            if metric in team1_stats and metric in team2_stats:
                team1_val = team1_stats[metric]
                team2_val = team2_stats[metric]

                # For DefEff, lower is better, so invert the advantage
                if metric == "DefEff":
                    advantage = (
                        "Team1"
                        if team1_val < team2_val
                        else "Team2" if team2_val < team1_val else "Even"
                    )
                else:
                    advantage = (
                        "Team1"
                        if team1_val > team2_val
                        else "Team2" if team2_val > team1_val else "Even"
                    )

                efficiency[metric] = {
                    "Team1": team1_val,
                    "Team2": team2_val,
                    "Difference": team1_val - team2_val,
                    "Advantage": advantage,
                }

        comparison["Efficiency"] = efficiency

        # Shooting percentages comparison
        shooting = {}
        for pct in ["FG%", "3P%", "FT%"]:
            if pct in team1_stats and pct in team2_stats:
                team1_val = team1_stats[pct]
                team2_val = team2_stats[pct]
                advantage = (
                    "Team1"
                    if team1_val > team2_val
                    else "Team2" if team2_val > team1_val else "Even"
                )

                shooting[pct] = {
                    "Team1": team1_val,
                    "Team2": team2_val,
                    "Difference": team1_val - team2_val,
                    "Advantage": advantage,
                }

        comparison["Shooting"] = shooting

        # Other key metrics comparison
        other_metrics = {}
        for metric in ["Pace"]:
            if metric in team1_stats and metric in team2_stats:
                team1_val = team1_stats[metric]
                team2_val = team2_stats[metric]
                # For pace, there's no clear advantage
                other_metrics[metric] = {
                    "Team1": team1_val,
                    "Team2": team2_val,
                    "Difference": team1_val - team2_val,
                    "Advantage": "Neutral",
                }

        comparison["OtherMetrics"] = other_metrics

        return comparison

    def _generate_matchup_analysis_markdown(self, analysis):
        """Generate markdown text for matchup analysis"""
        markdown = []

        # Header with seed and team names
        team1_seed = analysis["Team1Seed"] if analysis["Team1Seed"] else "UNK"
        team2_seed = analysis["Team2Seed"] if analysis["Team2Seed"] else "UNK"
        markdown.append(
            f"# {team1_seed} {analysis['Team1Name']} vs {team2_seed} {analysis['Team2Name']}\n\n"
        )

        # Prediction and odds section
        markdown.append("## Prediction\n")

        # Format win probability as percentage
        win_pct = analysis["WinProbability"] * 100

        # Determine the favorite
        if win_pct > 50:
            favorite = analysis["Team1Name"]
            underdog = analysis["Team2Name"]
            fav_pct = win_pct
            dog_pct = 100 - win_pct
        else:
            favorite = analysis["Team2Name"]
            underdog = analysis["Team1Name"]
            fav_pct = 100 - win_pct
            dog_pct = win_pct

        # Confidence level based on win probability
        confidence = (
            "Heavy favorite"
            if fav_pct >= 80
            else (
                "Strong favorite"
                if fav_pct >= 65
                else "Moderate favorite" if fav_pct >= 55 else "Slight favorite"
            )
        )

        markdown.append(
            f"**{favorite}** is the {confidence} with a **{fav_pct:.1f}%** win probability.\n"
        )
        markdown.append(
            f"**{underdog}** has a {dog_pct:.1f}% chance of pulling the upset.\n\n"
        )

        # Betting section
        markdown.append("## Betting Information\n")
        if win_pct > 50:
            markdown.append(
                f"**Spread:** {analysis['Team1Name']} {analysis['FormattedSpread']}\n"
            )
            markdown.append(
                f"**Moneyline:** {analysis['Team1Name']} {analysis['AmericanOdds']}, {analysis['Team2Name']} +{abs(analysis['AmericanOdds'])}\n\n"
            )
        else:
            # The spread is inverted for the underdog
            inverted_spread = analysis["Spread"] * -1
            formatted_inverted = (
                "PK"
                if inverted_spread == 0
                else (
                    f"+{inverted_spread}"
                    if inverted_spread > 0
                    else f"{inverted_spread}"
                )
            )
            markdown.append(
                f"**Spread:** {analysis['Team2Name']} {formatted_inverted}\n"
            )
            markdown.append(
                f"**Moneyline:** {analysis['Team2Name']} {analysis['AmericanOdds']}, {analysis['Team1Name']} +{abs(analysis['AmericanOdds'])}\n\n"
            )

        # Statistical comparison section
        if "StatComparison" in analysis and analysis["StatComparison"]:
            markdown.append("## Statistical Comparison\n\n")

            # Team headers with ELO
            markdown.append(
                f"| Metric | {analysis['Team1Name']} | {analysis['Team2Name']} | Advantage |\n"
            )
            markdown.append(
                "|--------|-----------------|-----------------|----------|\n"
            )
            markdown.append(
                f"| **ELO Rating** | {analysis['Team1ELO']:.0f} | {analysis['Team2ELO']:.0f} | {analysis['Team1Name'] if analysis['Team1ELO'] > analysis['Team2ELO'] else analysis['Team2Name']} |\n"
            )

            # Add efficiency metrics
            comparison = analysis["StatComparison"]

            if "Efficiency" in comparison:
                for metric, data in comparison["Efficiency"].items():
                    if metric == "OffEff":
                        label = "Offensive Efficiency"
                    elif metric == "DefEff":
                        label = "Defensive Efficiency"
                    elif metric == "NetEff":
                        label = "Net Efficiency"
                    else:
                        label = metric

                    advantage = (
                        analysis["Team1Name"]
                        if data["Advantage"] == "Team1"
                        else (
                            analysis["Team2Name"]
                            if data["Advantage"] == "Team2"
                            else "Even"
                        )
                    )
                    markdown.append(
                        f"| **{label}** | {data['Team1']:.1f} | {data['Team2']:.1f} | {advantage} |\n"
                    )

            # Add Four Factors
            if "FourFactors" in comparison:
                for factor, data in comparison["FourFactors"].items():
                    if factor == "eFG%":
                        label = "Effective FG%"
                        value1 = f"{data['Team1']*100:.1f}%"
                        value2 = f"{data['Team2']*100:.1f}%"
                    elif factor == "TOV%":
                        label = "Turnover Rate"
                        value1 = f"{data['Team1']*100:.1f}%"
                        value2 = f"{data['Team2']*100:.1f}%"
                    elif factor == "ORB%":
                        label = "Off. Rebounding %"
                        value1 = f"{data['Team1']*100:.1f}%"
                        value2 = f"{data['Team2']*100:.1f}%"
                    elif factor == "FTRate":
                        label = "Free Throw Rate"
                        value1 = f"{data['Team1']:.3f}"
                        value2 = f"{data['Team2']:.3f}"
                    else:
                        label = factor
                        value1 = f"{data['Team1']}"
                        value2 = f"{data['Team2']}"

                    advantage = (
                        analysis["Team1Name"]
                        if data["Advantage"] == "Team1"
                        else (
                            analysis["Team2Name"]
                            if data["Advantage"] == "Team2"
                            else "Even"
                        )
                    )
                    markdown.append(
                        f"| **{label}** | {value1} | {value2} | {advantage} |\n"
                    )

            # Add shooting percentages
            if "Shooting" in comparison:
                for pct, data in comparison["Shooting"].items():
                    if pct == "FG%":
                        label = "Field Goal %"
                    elif pct == "3P%":
                        label = "3-Point %"
                    elif pct == "FT%":
                        label = "Free Throw %"
                    else:
                        label = pct

                    value1 = f"{data['Team1']*100:.1f}%"
                    value2 = f"{data['Team2']*100:.1f}%"
                    advantage = (
                        analysis["Team1Name"]
                        if data["Advantage"] == "Team1"
                        else (
                            analysis["Team2Name"]
                            if data["Advantage"] == "Team2"
                            else "Even"
                        )
                    )
                    markdown.append(
                        f"| **{label}** | {value1} | {value2} | {advantage} |\n"
                    )

            # Add pace
            if "OtherMetrics" in comparison and "Pace" in comparison["OtherMetrics"]:
                data = comparison["OtherMetrics"]["Pace"]
                markdown.append(
                    f"| **Pace (Possessions)** | {data['Team1']:.1f} | {data['Team2']:.1f} | Style mismatch |\n"
                )

            markdown.append("\n")

        # Key matchup factors
        if "KeyMatchupFactors" in analysis and analysis["KeyMatchupFactors"]:
            markdown.append("## Key Matchup Factors\n\n")
            for factor in analysis["KeyMatchupFactors"]:
                markdown.append(f"**{factor['factor']}**: {factor['description']}\n\n")

        # Narrative explanation
        markdown.append("## Analysis\n\n")
        markdown.append(f"{analysis['Explanation']}\n\n")

        # Betting value assessment
        markdown.append("## Betting Value Assessment\n\n")

        # Calculate implied probability from American odds
        if analysis["AmericanOdds"] < 0:
            implied_prob = abs(analysis["AmericanOdds"]) / (
                abs(analysis["AmericanOdds"]) + 100
            )
        else:
            implied_prob = 100 / (analysis["AmericanOdds"] + 100)

        # Compare model probability with implied odds probability
        model_prob = (
            analysis["WinProbability"]
            if win_pct > 50
            else 1 - analysis["WinProbability"]
        )
        edge = model_prob - implied_prob

        if edge > 0.05:
            markdown.append(
                f"**Strong Value**: Our model gives {favorite} a {model_prob:.1%} chance to win, compared to the implied {implied_prob:.1%} from betting odds. This represents significant betting value.\n\n"
            )
        elif edge > 0.02:
            markdown.append(
                f"**Moderate Value**: Our model gives {favorite} a {model_prob:.1%} chance to win, compared to the implied {implied_prob:.1%} from betting odds. This represents some betting value.\n\n"
            )
        elif edge > -0.02:
            markdown.append(
                f"**Fair Odds**: The betting line closely matches our projected win probability. No significant edge either way.\n\n"
            )
        else:
            markdown.append(
                f"**Poor Value**: The betting odds overvalue {favorite}'s chances to win. Our model projects a {model_prob:.1%} win probability compared to the implied {implied_prob:.1%} from the odds.\n\n"
            )

        # Final pick
        markdown.append("## The Pick\n\n")

        if win_pct > 80:
            markdown.append(
                f"**{favorite}** to win easily. This game should not be close.\n\n"
            )
        elif win_pct > 65:
            markdown.append(
                f"**{favorite}** should control this game and win comfortably.\n\n"
            )
        elif win_pct > 55:
            markdown.append(
                f"**{favorite}** has the edge in this matchup, but expect a competitive game.\n\n"
            )
        else:
            markdown.append(
                f"**{favorite}** has a slight edge, but this is essentially a coin flip. Could go either way.\n\n"
            )

        return "".join(markdown)

    def generate_region_summary(self, region, output_file=None):
        """
        Generate a comprehensive summary of a tournament region

        Parameters:
        region: Region identifier ('W', 'X', 'Y', 'Z', etc.)
        output_file: Optional file path to save the summary

        Returns:
        dict: Region summary data
        str: Markdown text of the summary
        """
        # Get teams in this region
        region_seeds = self.data_manager.data["tourney_seeds"][
            (self.data_manager.data["tourney_seeds"]["Season"] == self.current_season)
            & (self.data_manager.data["tourney_seeds"]["Seed"].str.startswith(region))
        ]

        if len(region_seeds) == 0:
            print(f"No teams found in region {region} for season {self.current_season}")
            return (
                None,
                f"No teams found in region {region} for season {self.current_season}",
            )

        # Get team info
        teams = []
        for _, seed_row in region_seeds.iterrows():
            team_id = seed_row["TeamID"]
            seed = seed_row["Seed"]

            # Get team profile
            profile, _ = self.generate_team_profile(team_id)

            # Add seed info
            seed_number = int(re.search(r"\d+", seed).group())

            # Calculate championship odds for this team
            # This would require simulating the tournament - approximate for now
            # based on seed and ELO
            elo_rating = profile.get("ELO", 1500)
            # Simple formula: Higher ELO = better odds, lower seed = better odds
            champ_factor = (elo_rating - 1500) / 100 - (seed_number - 8) / 2
            champ_odds = max(0.1, min(35, 2 ** (champ_factor / 2)))

            # Calculate Sweet 16 odds (simpler estimate)
            sweet16_odds = max(
                5, min(90, 100 - (seed_number**1.5) * 3 + (elo_rating - 1500) / 20)
            )

            teams.append(
                {
                    "TeamID": team_id,
                    "TeamName": profile["TeamName"],
                    "Seed": seed,
                    "SeedNumber": seed_number,
                    "ELO": profile.get("ELO", 1500),
                    "AdvancedStats": profile.get("AdvancedStats", {}),
                    "Strengths": profile.get("Strengths", []),
                    "Weaknesses": profile.get("Weaknesses", []),
                    "ChampionshipOdds": champ_odds,
                    "Sweet16Odds": sweet16_odds,
                }
            )

        # Sort teams by seed number
        teams.sort(key=lambda x: x["SeedNumber"])

        # Generate region matchups for the first round
        first_round_matchups = []

        # Standard first-round matchups: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
        matchup_pairs = [
            (1, 16),
            (8, 9),
            (5, 12),
            (4, 13),
            (6, 11),
            (3, 14),
            (7, 10),
            (2, 15),
        ]

        for seed1, seed2 in matchup_pairs:
            # Find teams with these seeds
            team1 = next((t for t in teams if t["SeedNumber"] == seed1), None)
            team2 = next((t for t in teams if t["SeedNumber"] == seed2), None)

            if team1 and team2:
                # Get prediction
                win_prob = self.predict_game(team1["TeamID"], team2["TeamID"])

                # Get betting odds
                spread = self.win_probability_to_spread(win_prob)
                formatted_spread = self.format_spread(spread)

                first_round_matchups.append(
                    {
                        "Team1": team1,
                        "Team2": team2,
                        "WinProbability": win_prob,
                        "Spread": spread,
                        "FormattedSpread": formatted_spread,
                    }
                )

        # Create the region summary
        summary = {
            "Region": region,
            "Teams": teams,
            "FirstRoundMatchups": first_round_matchups,
        }

        # Generate markdown text
        markdown_text = self._generate_region_summary_markdown(summary)

        # Save to file if requested
        if output_file:
            with open(output_file, "w") as f:
                f.write(markdown_text)
            print(f"Region summary saved to {output_file}")

        return summary, markdown_text

    def _generate_region_summary_markdown(self, summary):
        """Generate markdown text for region summary"""
        markdown = []

        # Header
        markdown.append(f"# {summary['Region']} Region Summary\n\n")

        # Top teams section
        markdown.append("## Top Teams\n\n")

        # Get top 4 seeds
        top_seeds = [t for t in summary["Teams"] if t["SeedNumber"] <= 4]
        for team in top_seeds:
            seed_str = f"#{team['SeedNumber']}"
            markdown.append(f"### {seed_str} {team['TeamName']}\n")

            # Add ELO rating
            markdown.append(f"**ELO Rating:** {team['ELO']:.0f}\n\n")

            # Add top strengths
            if team["Strengths"]:
                markdown.append("**Key Strengths:**\n")
                for strength in team["Strengths"][:3]:  # Top 3 strengths
                    markdown.append(f"* {strength}\n")
                markdown.append("\n")

            # Add championship odds
            markdown.append(f"**Championship Odds:** {team['ChampionshipOdds']:.1f}%\n")
            markdown.append(f"**Sweet 16 Odds:** {team['Sweet16Odds']:.1f}%\n\n")

        # First round matchups
        markdown.append("## First Round Matchups\n\n")

        for matchup in summary["FirstRoundMatchups"]:
            team1 = matchup["Team1"]
            team2 = matchup["Team2"]

            # Format matchup header
            markdown.append(
                f"### #{team1['SeedNumber']} {team1['TeamName']} vs #{team2['SeedNumber']} {team2['TeamName']}\n\n"
            )

            # Add prediction
            win_pct = matchup["WinProbability"] * 100
            if win_pct > 50:
                markdown.append(
                    f"**Prediction:** {team1['TeamName']} has a {win_pct:.1f}% chance to win\n"
                )
                markdown.append(
                    f"**Spread:** {team1['TeamName']} {matchup['FormattedSpread']}\n\n"
                )
            else:
                markdown.append(
                    f"**Prediction:** {team2['TeamName']} has a {(100-win_pct):.1f}% chance to win\n"
                )
                # Flip the spread for team2
                inverted_spread = -matchup["Spread"]
                formatted_inverted = (
                    f"+{inverted_spread}"
                    if inverted_spread > 0
                    else f"{inverted_spread}"
                )
                markdown.append(
                    f"**Spread:** {team2['TeamName']} {formatted_inverted}\n\n"
                )

        # Potential upsets
        markdown.append("## Potential Upsets\n\n")

        upset_candidates = []
        for matchup in summary["FirstRoundMatchups"]:
            team1 = matchup["Team1"]
            team2 = matchup["Team2"]
            win_prob = matchup["WinProbability"]

            # Check if it's a potential upset (lower seed has >35% chance)
            is_upset = (
                team1["SeedNumber"] < team2["SeedNumber"] and win_prob < 0.65
            ) or (team2["SeedNumber"] < team1["SeedNumber"] and win_prob > 0.35)

            seed_diff = abs(team1["SeedNumber"] - team2["SeedNumber"])

            # Only consider matchups with significant seed difference
            if is_upset and seed_diff >= 3:
                upset_candidates.append(
                    {
                        "Matchup": f"#{team1['SeedNumber']} {team1['TeamName']} vs #{team2['SeedNumber']} {team2['TeamName']}",
                        "UpsettingTeam": (
                            team2["TeamName"] if win_prob < 0.5 else team1["TeamName"]
                        ),
                        "UpsettingTeamSeed": (
                            team2["SeedNumber"]
                            if win_prob < 0.5
                            else team1["SeedNumber"]
                        ),
                        "FavoredTeam": (
                            team1["TeamName"] if win_prob >= 0.5 else team2["TeamName"]
                        ),
                        "FavoredTeamSeed": (
                            team1["SeedNumber"]
                            if win_prob >= 0.5
                            else team2["SeedNumber"]
                        ),
                        "UpsetProbability": (
                            1 - win_prob if win_prob >= 0.5 else win_prob
                        ),
                    }
                )

        # Sort by upset probability
        upset_candidates.sort(key=lambda x: x["UpsetProbability"], reverse=True)

        if upset_candidates:
            for upset in upset_candidates:
                markdown.append(
                    f"**{upset['Matchup']}**: #{upset['UpsettingTeamSeed']} {upset['UpsettingTeam']} has a {upset['UpsetProbability']*100:.1f}% chance to upset #{upset['FavoredTeamSeed']} {upset['FavoredTeam']}\n\n"
                )
        else:
            markdown.append(
                "No significant potential upsets identified in the first round.\n\n"
            )

        # Sleeper teams
        markdown.append("## Sleeper Teams to Watch\n\n")

        # Find sleeper candidates (teams seeded 5-12 with good metrics)
        sleeper_candidates = []
        for team in summary["Teams"]:
            if 5 <= team["SeedNumber"] <= 12:
                # Look for teams with good ELO or efficiency metrics
                is_sleeper = False
                reasoning = []

                if team["ELO"] > 1800:
                    is_sleeper = True
                    reasoning.append(f"Strong ELO rating of {team['ELO']:.0f}")

                if "AdvancedStats" in team and team["AdvancedStats"]:
                    stats = team["AdvancedStats"]

                    if "NetEff" in stats and stats["NetEff"] > 15:
                        is_sleeper = True
                        reasoning.append(
                            f"Excellent efficiency metrics (Net: {stats['NetEff']:.1f})"
                        )

                    if "OffEff" in stats and stats["OffEff"] > 115:
                        is_sleeper = True
                        reasoning.append(
                            f"Elite offense ({stats['OffEff']:.1f} efficiency)"
                        )

                    if "DefEff" in stats and stats["DefEff"] < 90:
                        is_sleeper = True
                        reasoning.append(
                            f"Elite defense ({stats['DefEff']:.1f} efficiency)"
                        )

                if team["Sweet16Odds"] > 40:
                    is_sleeper = True
                    reasoning.append(
                        f"Good chance to reach Sweet 16 ({team['Sweet16Odds']:.1f}%)"
                    )

                if is_sleeper:
                    sleeper_candidates.append({"Team": team, "Reasoning": reasoning})

        if sleeper_candidates:
            for sleeper in sleeper_candidates:
                team = sleeper["Team"]
                markdown.append(f"**#{team['SeedNumber']} {team['TeamName']}**\n\n")
                markdown.append(f"**Why they could surprise:**\n")
                for reason in sleeper["Reasoning"]:
                    markdown.append(f"* {reason}\n")
                markdown.append("\n")
        else:
            markdown.append("No clear sleeper teams identified in this region.\n\n")

        # Region winner prediction
        markdown.append("## Region Winner Prediction\n\n")

        # Sort teams by championship odds
        sorted_teams = sorted(
            summary["Teams"], key=lambda x: x["ChampionshipOdds"], reverse=True
        )

        # Top 3 contenders
        markdown.append("**Top Contenders:**\n\n")
        for i, team in enumerate(sorted_teams[:3]):
            markdown.append(
                f"**{i+1}. #{team['SeedNumber']} {team['TeamName']}** - {team['ChampionshipOdds']:.1f}% chance to win the region\n\n"
            )

        return "".join(markdown)

    def generate_executive_summary(self, output_file=None):
        """
        Generate a comprehensive executive summary of the entire tournament

        Parameters:
        output_file: Optional file path to save the summary

        Returns:
        dict: Tournament summary data
        str: Markdown text of the summary
        """
        # Get tournament teams
        seeds_df = self.data_manager.data["tourney_seeds"][
            self.data_manager.data["tourney_seeds"]["Season"] == self.current_season
        ]

        if len(seeds_df) == 0:
            print(f"No teams found for season {self.current_season}")
            return None, f"No teams found for season {self.current_season}"

        # Get unique regions
        regions = []
        for seed in seeds_df["Seed"]:
            region = seed[0]  # First character of seed is region
            if region not in regions:
                regions.append(region)

        # Generate region summaries
        region_summaries = []
        for region in regions:
            summary, _ = self.generate_region_summary(region)
            if summary:
                region_summaries.append(summary)

        # Compile tournament teams data
        teams = []
        for summary in region_summaries:
            teams.extend(summary["Teams"])

        # Find the favorites
        teams_by_odds = sorted(teams, key=lambda x: x["ChampionshipOdds"], reverse=True)
        favorites = teams_by_odds[:5]  # Top 5 teams

        # Find best first-round upset picks
        upset_picks = []
        for summary in region_summaries:
            for matchup in summary["FirstRoundMatchups"]:
                team1 = matchup["Team1"]
                team2 = matchup["Team2"]
                win_prob = matchup["WinProbability"]

                # Consider upsets where underdog has >25% chance
                seed_diff = abs(team1["SeedNumber"] - team2["SeedNumber"])

                if seed_diff >= 4:
                    if team1["SeedNumber"] < team2["SeedNumber"] and win_prob < 0.75:
                        upset_picks.append(
                            {
                                "UpsettingTeam": team2,
                                "FavoredTeam": team1,
                                "Region": summary["Region"],
                                "UpsetProbability": 1 - win_prob,
                                "SeedDifference": seed_diff,
                            }
                        )
                    elif team2["SeedNumber"] < team1["SeedNumber"] and win_prob > 0.25:
                        upset_picks.append(
                            {
                                "UpsettingTeam": team1,
                                "FavoredTeam": team2,
                                "Region": summary["Region"],
                                "UpsetProbability": win_prob,
                                "SeedDifference": seed_diff,
                            }
                        )

        # Sort by upset probability
        upset_picks.sort(key=lambda x: x["UpsetProbability"], reverse=True)
        best_upset_picks = upset_picks[:5]  # Top 5 upset picks

        # Final Four predictions - simple heuristic based on championship odds
        final_four = []
        for region in regions:
            region_teams = [t for t in teams if t["Seed"].startswith(region)]
            if region_teams:
                # Take team with highest championship odds from each region
                region_teams.sort(key=lambda x: x["ChampionshipOdds"], reverse=True)
                final_four.append(region_teams[0])

        # Generate champion prediction
        champion = None
        if final_four:
            final_four.sort(key=lambda x: x["ChampionshipOdds"], reverse=True)
            champion = final_four[0]

        # Create the tournament summary
        summary = {
            "Season": self.current_season,
            "Regions": region_summaries,
            "Favorites": favorites,
            "BestUpsetPicks": best_upset_picks,
            "FinalFour": final_four,
            "Champion": champion,
        }

        # Generate markdown text
        markdown_text = self._generate_executive_summary_markdown(summary)

        # Save to file if requested
        if output_file:
            with open(output_file, "w") as f:
                f.write(markdown_text)
            print(f"Tournament executive summary saved to {output_file}")

        return summary, markdown_text

    def _generate_executive_summary_markdown(self, summary):
        """Generate markdown text for tournament executive summary"""
        markdown = []

        # Header
        current_date = datetime.now().strftime("%B %d, %Y")
        markdown.append(
            f"# {summary['Season']} March Madness Tournament Executive Summary\n\n"
        )
        markdown.append(f"*Generated on {current_date}*\n\n")

        # Introduction
        markdown.append("## Tournament Overview\n\n")
        markdown.append(
            f"This executive summary provides a comprehensive analysis of the {summary['Season']} NCAA Men's Basketball Tournament. "
        )
        markdown.append(
            "Our predictions are based on advanced analytics, including ELO ratings, efficiency metrics, and statistical modeling.\n\n"
        )

        # Top title contenders
        markdown.append("## Top Title Contenders\n\n")

        for i, team in enumerate(summary["Favorites"]):
            markdown.append(
                f"### {i+1}. #{team['SeedNumber']} {team['TeamName']} ({team['ChampionshipOdds']:.1f}%)\n\n"
            )

            # Add ELO rating
            markdown.append(f"**ELO Rating:** {team['ELO']:.0f}\n\n")

            # Add advanced stats
            if "AdvancedStats" in team and team["AdvancedStats"]:
                stats = team["AdvancedStats"]

                if "OffEff" in stats and "DefEff" in stats:
                    markdown.append(
                        f"**Offensive Efficiency:** {stats['OffEff']:.1f} (Points per 100 possessions)\n"
                    )
                    markdown.append(
                        f"**Defensive Efficiency:** {stats['DefEff']:.1f} (Points allowed per 100 possessions)\n"
                    )

                if "NetEff" in stats:
                    markdown.append(f"**Net Efficiency:** {stats['NetEff']:.1f}\n\n")

            # Add key strengths
            if "Strengths" in team and team["Strengths"]:
                markdown.append("**Key Strengths:**\n")
                for strength in team["Strengths"][:3]:  # Top 3 strengths
                    markdown.append(f"* {strength}\n")
                markdown.append("\n")

        # Best first-round upset picks
        markdown.append("## Best First-Round Upset Picks\n\n")

        for i, pick in enumerate(summary["BestUpsetPicks"]):
            upset_team = pick["UpsettingTeam"]
            favored_team = pick["FavoredTeam"]

            markdown.append(
                f"### {i+1}. #{upset_team['SeedNumber']} {upset_team['TeamName']} over #{favored_team['SeedNumber']} {favored_team['TeamName']} ({pick['Region']} Region)\n\n"
            )
            markdown.append(
                f"**Upset Probability:** {pick['UpsetProbability']*100:.1f}%\n\n"
            )

            # Add reasons for the upset potential
            markdown.append("**Why This Upset Could Happen:**\n")

            # Check if upsetting team has strong metrics
            if "AdvancedStats" in upset_team and upset_team["AdvancedStats"]:
                stats = upset_team["AdvancedStats"]

                if "NetEff" in stats and stats["NetEff"] > 10:
                    markdown.append(
                        f"* {upset_team['TeamName']} has excellent efficiency metrics (Net: {stats['NetEff']:.1f})\n"
                    )

                if "OffEff" in stats and stats["OffEff"] > 110:
                    markdown.append(
                        f"* {upset_team['TeamName']} has a potent offense ({stats['OffEff']:.1f} efficiency)\n"
                    )

                if "DefEff" in stats and stats["DefEff"] < 95:
                    markdown.append(
                        f"* {upset_team['TeamName']} plays elite defense ({stats['DefEff']:.1f} efficiency)\n"
                    )

            # Check for style mismatches
            if (
                "AdvancedStats" in upset_team
                and "AdvancedStats" in favored_team
                and upset_team["AdvancedStats"]
                and favored_team["AdvancedStats"]
            ):

                upset_stats = upset_team["AdvancedStats"]
                favored_stats = favored_team["AdvancedStats"]

                # Tempo mismatch
                if "Pace" in upset_stats and "Pace" in favored_stats:
                    pace_diff = upset_stats["Pace"] - favored_stats["Pace"]
                    if abs(pace_diff) > 5:
                        faster = (
                            upset_team["TeamName"]
                            if pace_diff > 0
                            else favored_team["TeamName"]
                        )
                        slower = (
                            favored_team["TeamName"]
                            if pace_diff > 0
                            else upset_team["TeamName"]
                        )
                        markdown.append(
                            f"* Significant tempo mismatch: {faster} wants to play fast while {slower} prefers a slower pace\n"
                        )

                # 3-point shooting advantage
                if "3P%" in upset_stats and "3P%" in favored_stats:
                    three_diff = upset_stats["3P%"] - favored_stats["3P%"]
                    if three_diff > 0.05:
                        markdown.append(
                            f"* {upset_team['TeamName']} has a significant 3-point shooting advantage ({upset_stats['3P%']*100:.1f}% vs {favored_stats['3P%']*100:.1f}%)\n"
                        )

            # Add team strengths if available
            if "Strengths" in upset_team and upset_team["Strengths"]:
                for strength in upset_team["Strengths"][:2]:  # Top 2 strengths
                    markdown.append(f"* {strength}\n")

            markdown.append("\n")

        # Final Four predictions
        markdown.append("## Final Four Predictions\n\n")

        if summary["FinalFour"]:
            for team in summary["FinalFour"]:
                region = team["Seed"][0]
                markdown.append(
                    f"**{region} Region:** #{team['SeedNumber']} {team['TeamName']}\n\n"
                )
                markdown.append(f"* ELO Rating: {team['ELO']:.0f}\n")
                markdown.append(
                    f"* Championship Odds: {team['ChampionshipOdds']:.1f}%\n\n"
                )

        # Championship prediction
        markdown.append("## Championship Prediction\n\n")

        if summary["Champion"]:
            champion = summary["Champion"]
            markdown.append(f"**#{champion['SeedNumber']} {champion['TeamName']}**\n\n")

            # Add reasoning
            markdown.append("**Why They'll Win:**\n")

            # Add ELO and advanced stats reasoning
            markdown.append(f"* Elite ELO rating of {champion['ELO']:.0f}\n")

            if "AdvancedStats" in champion and champion["AdvancedStats"]:
                stats = champion["AdvancedStats"]

                if "NetEff" in stats:
                    markdown.append(
                        f"* Impressive {stats['NetEff']:.1f} net efficiency rating\n"
                    )

                if "OffEff" in stats and "DefEff" in stats:
                    if stats["OffEff"] > 110 and stats["DefEff"] < 95:
                        markdown.append(
                            f"* Elite on both ends of the court (Offense: {stats['OffEff']:.1f}, Defense: {stats['DefEff']:.1f})\n"
                        )
                    elif stats["OffEff"] > 115:
                        markdown.append(
                            f"* Dominant offense ({stats['OffEff']:.1f} efficiency) that can score against anyone\n"
                        )
                    elif stats["DefEff"] < 90:
                        markdown.append(
                            f"* Lockdown defense ({stats['DefEff']:.1f} efficiency) that will carry them through tough matchups\n"
                        )

            # Add key strengths
            if "Strengths" in champion and champion["Strengths"]:
                for strength in champion["Strengths"][:3]:  # Top 3 strengths
                    markdown.append(f"* {strength}\n")

            markdown.append("\n")

        # Region-by-region breakdown
        markdown.append("## Region-by-Region Breakdown\n\n")

        for region_summary in summary["Regions"]:
            region = region_summary["Region"]
            markdown.append(f"### {region} Region\n\n")

            # Top seed
            top_seed = next(
                (t for t in region_summary["Teams"] if t["SeedNumber"] == 1), None
            )
            if top_seed:
                markdown.append(f"**1 Seed:** {top_seed['TeamName']}\n\n")

            # Most likely Sweet 16 teams
            sweet16_teams = sorted(
                region_summary["Teams"], key=lambda x: x["Sweet16Odds"], reverse=True
            )[:4]

            markdown.append("**Most Likely Sweet 16 Teams:**\n")
            for team in sweet16_teams:
                markdown.append(
                    f"* #{team['SeedNumber']} {team['TeamName']} ({team['Sweet16Odds']:.1f}%)\n"
                )
            markdown.append("\n")

            # Upset to watch
            upset_candidates = []
            for matchup in region_summary["FirstRoundMatchups"]:
                team1 = matchup["Team1"]
                team2 = matchup["Team2"]
                win_prob = matchup["WinProbability"]

                # Check if it's a potential upset (lower seed has >35% chance)
                seed_diff = abs(team1["SeedNumber"] - team2["SeedNumber"])

                if seed_diff >= 4:
                    if team1["SeedNumber"] < team2["SeedNumber"] and win_prob < 0.7:
                        upset_candidates.append(
                            {
                                "Matchup": f"#{team2['SeedNumber']} {team2['TeamName']} over #{team1['SeedNumber']} {team1['TeamName']}",
                                "Probability": 1 - win_prob,
                            }
                        )
                    elif team2["SeedNumber"] < team1["SeedNumber"] and win_prob > 0.3:
                        upset_candidates.append(
                            {
                                "Matchup": f"#{team1['SeedNumber']} {team1['TeamName']} over #{team2['SeedNumber']} {team2['TeamName']}",
                                "Probability": win_prob,
                            }
                        )

            # Sort by probability
            upset_candidates.sort(key=lambda x: x["Probability"], reverse=True)

            if upset_candidates:
                top_upset = upset_candidates[0]
                markdown.append(
                    f"**Best Upset Pick:** {top_upset['Matchup']} ({top_upset['Probability']*100:.1f}%)\n\n"
                )

            # Region winner prediction
            teams_by_odds = sorted(
                region_summary["Teams"],
                key=lambda x: x["ChampionshipOdds"],
                reverse=True,
            )

            if teams_by_odds:
                winner = teams_by_odds[0]
                markdown.append(
                    f"**Predicted Winner:** #{winner['SeedNumber']} {winner['TeamName']} ({winner['ChampionshipOdds']:.1f}%)\n\n"
                )

        # Strategy tips
        markdown.append("## Bracket Strategy Tips\n\n")

        markdown.append("**1. Don't pick too many upsets in the first round.**\n")
        markdown.append(
            "   While upsets always happen, most 1-4 seeds advance. Focus on the handful of upsets with strong analytical backing.\n\n"
        )

        markdown.append("**2. Look for sleeper teams in the 5-7 seed range.**\n")
        markdown.append(
            "   These teams often have similar statistical profiles to top seeds but with lower expectations.\n\n"
        )

        markdown.append("**3. Pay attention to style matchups.**\n")
        markdown.append(
            "   Teams that control tempo, shoot well from outside, or have dominant defenses often exceed their seed expectations.\n\n"
        )

        markdown.append(
            "**4. Consider picking at least one non-1 seed in your Final Four.**\n"
        )
        markdown.append(
            "   Since 2000, it's extremely rare for all four 1 seeds to make the Final Four (happened only once in 2008).\n\n"
        )

        # Methodology note
        markdown.append("## Methodology\n\n")

        markdown.append("These predictions combine several analytical approaches:\n\n")
        markdown.append(
            "* **ELO Ratings:** A measure of team strength that updates after every game\n"
        )
        markdown.append(
            "* **Efficiency Metrics:** Points scored and allowed per 100 possessions, adjusted for opponent strength\n"
        )
        markdown.append(
            "* **Four Factors:** Field goal percentage, turnover rate, offensive rebounding, and free throw rate\n"
        )
        markdown.append(
            "* **Recency Factors:** Team momentum, recent performance, and tournament readiness\n\n"
        )

        return "".join(markdown)


def load_data(gender_code, year):
    """Load tournament data files"""
    # Path to data files
    teams_path = os.path.join(DATA_DIR.format(year=year), f"{gender_code}Teams.csv")
    seeds_path = os.path.join(
        DATA_DIR.format(year=year), f"{gender_code}NCAATourneySeeds.csv"
    )
    slots_path = os.path.join(
        DATA_DIR.format(year=year), f"{gender_code}NCAATourneySlots.csv"
    )

    # Load data
    teams_df = pd.read_csv(teams_path)
    seeds_df = pd.read_csv(seeds_path)
    slots_df = pd.read_csv(slots_path)

    # Filter for current season
    seeds_df = seeds_df[seeds_df["Season"] == year]
    slots_df = slots_df[slots_df["Season"] == year]

    return teams_df, seeds_df, slots_df


def load_elo_data(gender_code="M", year=2025):
    """Load full ELO history data from CSV file"""
    file_path = os.path.join(OUTPUT_DIR, f"{gender_code}_elo_ratings.csv")

    if os.path.exists(file_path):
        try:
            elo_df = pd.read_csv(file_path)
            return elo_df
        except:
            print(f"Error loading ELO data from {file_path}")

    return None


def load_elo_dict(elo_df, season=2025, day_num=None):
    """Extract current ELO ratings from full ELO history"""
    if elo_df is None:
        return None

    # Get latest ELO for each team up to day_num or the max available day
    latest_elos = {}

    # If day_num is None, get the max day number for the season
    if day_num is None:
        season_data = elo_df[elo_df["Season"] == season]
        if len(season_data) > 0:
            day_num = season_data["DayNum"].max()
            print(f"Using max day number for season {season}: {day_num}")
        else:
            day_num = 150  # Higher default to ensure we get all data
            print(f"No data found for season {season}, using default day: {day_num}")

    # Group by TeamID and find the latest day before day_num
    for team_id in elo_df["TeamID"].unique():
        team_data = elo_df[
            (elo_df["TeamID"] == team_id)
            & (elo_df["Season"] == season)
            & (elo_df["DayNum"] <= day_num)
        ]

        if len(team_data) > 0:
            # Sort by day and get the latest
            latest = team_data.sort_values("DayNum", ascending=False).iloc[0]
            latest_elos[team_id] = latest["ELO"]

    return latest_elos


def load_feature_data(gender_code="M"):
    """Load feature dataset"""
    file_path = os.path.join(OUTPUT_DIR, f"{gender_code}_feature_dataset.csv")

    if os.path.exists(file_path):
        try:
            feature_df = pd.read_csv(file_path)
            return feature_df
        except:
            print(f"Error loading feature data from {file_path}")

    return None


def load_advanced_stats(feature_df, season=2025):
    """Extract advanced stats from feature dataset"""
    if feature_df is None:
        return None

    stats_dict = {}

    # Filter to season
    season_df = feature_df[feature_df["Season"] == season]

    # Process unique Team1 entries
    for _, row in season_df.drop_duplicates("Team1ID").iterrows():
        team_id = row["Team1ID"]

        # Extract advanced stats
        stats = {}
        for col in row.index:
            # Get all Team1_ prefixed columns
            if col.startswith("Team1_"):
                stat_name = col.replace("Team1_", "")
                stats[stat_name] = row[col]

        # Also add non-prefixed stats
        if "Team1OffEff" in row.index:
            stats["OffEff"] = row["Team1OffEff"]
        if "Team1DefEff" in row.index:
            stats["DefEff"] = row["Team1DefEff"]
        if "Team1NetEff" in row.index:
            stats["NetEff"] = row["Team1NetEff"]

        # Add standard stats
        for stat in ["Pace", "eFG%", "TOV%", "ORB%", "FTRate", "FG%", "3P%", "FT%"]:
            stat_col = f"Team1_{stat}"
            if stat_col in row.index:
                stats[stat] = row[stat_col]

        stats_dict[team_id] = stats

    return stats_dict


def load_predictions(gender_code="M", year=2025):
    """Load prediction data"""
    file_path = os.path.join(OUTPUT_DIR, f"submission_{year}_{gender_code}.csv")

    if os.path.exists(file_path):
        try:
            predictions_df = pd.read_csv(file_path)
            return predictions_df
        except:
            print(f"Error loading predictions from {file_path}")

    return None


def generate_enhanced_analysis(
    gender_code="M",
    year=2025,
    output_dir="./output/analysis",
    analysis_type="all",
    team_id=None,
    matchup_ids=None,
    region=None,
):
    """
    Generate enhanced bracket analysis

    Parameters:
    gender_code: Gender code ('M' for men, 'W' for women)
    year: Tournament year
    output_dir: Directory to save output files
    analysis_type: Type of analysis to generate ('team', 'matchup', 'region', 'executive', 'all')
    team_id: Team ID for team profile (required if analysis_type='team')
    matchup_ids: Tuple of (team1_id, team2_id) for matchup analysis (required if analysis_type='matchup')
    region: Region identifier for region analysis (required if analysis_type='region')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    teams_df, seeds_df, slots_df = load_data(gender_code, year)
    elo_df = load_elo_data(gender_code, year)
    current_elo_dict = load_elo_dict(elo_df, year, None)  # Use None to get max day
    predictions_df = load_predictions(gender_code, year)

    # Set up components to generate fresh feature data
    print("Setting up ML components to generate fresh feature data...")

    data_manager = MarchMadnessDataManager(
        data_dir=DATA_DIR.format(year=year), gender=gender_code, current_season=year
    )
    data_manager.load_data()

    # Create ELO system
    elo_system = EloRatingSystem(data_manager)
    if elo_df is not None:
        # Use pre-calculated ELO ratings if available
        team_elo_ratings = {}
        for _, row in elo_df.iterrows():
            key = (row["Season"], row["TeamID"], row["DayNum"])
            team_elo_ratings[key] = row["ELO"]
        elo_system.team_elo_ratings = team_elo_ratings
    else:
        # Calculate ELO ratings if not available
        elo_system.calculate_elo_ratings(start_year=2003)

    # Create TeamStatsCalculator
    stats_calculator = TeamStatsCalculator(data_manager)
    stats_calculator.calculate_advanced_team_stats()

    # Create MLModel for on-demand feature generation
    ml_model = MarchMadnessMLModel(data_manager, elo_system, stats_calculator)

    # We'll still load the feature dataset for fallback
    feature_df = load_feature_data(gender_code)

    # Extract advanced stats for fallback
    advanced_stats = None
    if feature_df is not None:
        advanced_stats = load_advanced_stats(feature_df, year)

    # Create enhanced predictor with ml_model
    predictor = EnhancedPredictor(
        teams_df=teams_df,
        seeds_df=seeds_df,
        slots_df=slots_df,
        predictions_df=predictions_df,
        elo_df=current_elo_dict,
        stats_df=advanced_stats,
        feature_df=feature_df,
        full_elo_history_df=elo_df,
        current_season=year,
        ml_model=ml_model,  # Pass the ML model
    )

    # Generate requested analysis
    if analysis_type == "team" or analysis_type == "all":
        if analysis_type == "team" and team_id is None:
            print("Error: team_id is required for team profile analysis")
            return

        if team_id is not None:
            # Generate single team profile
            output_file = os.path.join(output_dir, f"team_profile_{team_id}.md")
            predictor.generate_team_profile(team_id, output_file)
        else:
            # Generate profiles for all tournament teams
            for _, row in seeds_df.iterrows():
                team_id = row["TeamID"]
                output_file = os.path.join(output_dir, f"team_profile_{team_id}.md")
                predictor.generate_team_profile(team_id, output_file)

    if analysis_type == "matchup" or analysis_type == "all":
        if analysis_type == "matchup" and matchup_ids is None:
            print("Error: matchup_ids is required for matchup analysis")
            return

        if matchup_ids is not None:
            # Generate single matchup analysis
            team1_id, team2_id = matchup_ids
            output_file = os.path.join(
                output_dir, f"matchup_{team1_id}_vs_{team2_id}.md"
            )
            predictor.generate_enhanced_matchup_analysis(
                team1_id, team2_id, output_file
            )
        else:
            # Generate first round matchups
            rounds, slot_team_map = get_tournament_structure(slots_df, seeds_df, year)

            # Process first round matchups
            for strong_seed, weak_seed, _ in rounds["R1"]:
                team1_id = predictor.data_manager.get_team_id_from_seed(
                    year, strong_seed
                )
                team2_id = predictor.data_manager.get_team_id_from_seed(year, weak_seed)

                if team1_id is None or team2_id is None:
                    continue  # Skip if teams aren't set yet

                output_file = os.path.join(
                    output_dir, f"matchup_{team1_id}_vs_{team2_id}.md"
                )
                predictor.generate_enhanced_matchup_analysis(
                    team1_id, team2_id, output_file
                )

    if analysis_type == "region" or analysis_type == "all":
        if analysis_type == "region" and region is None:
            print("Error: region is required for region analysis")
            return

        # Get unique regions
        regions = []
        if region is not None:
            regions = [region]
        else:
            for seed in seeds_df["Seed"]:
                region_code = seed[0]  # First character of seed is region
                if region_code not in regions:
                    regions.append(region_code)

        # Generate region summaries
        for region_code in regions:
            output_file = os.path.join(output_dir, f"region_{region_code}_summary.md")
            predictor.generate_region_summary(region_code, output_file)

    if analysis_type == "executive" or analysis_type == "all":
        # Generate executive summary
        output_file = os.path.join(output_dir, f"tournament_executive_summary.md")
        predictor.generate_executive_summary(output_file)

    print(f"Analysis generation complete. Output saved to {output_dir}")

    # Also generate the traditional bracket analysis for comparison
    if analysis_type == "all":
        original_output = os.path.join(
            output_dir, f"original_matchup_analysis_{gender_code}_{year}.md"
        )
        generate_bracket_analysis(gender_code, "elo_enhanced", year, original_output)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate enhanced March Madness tournament analysis"
    )
    parser.add_argument(
        "-g",
        "--gender",
        choices=["M", "W"],
        default="M",
        help="Gender: M for men's tournament, W for women's",
    )
    parser.add_argument("-y", "--year", type=int, default=2025, help="Tournament year")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./output/analysis",
        help="Output directory for analysis files",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=["team", "matchup", "region", "executive", "all"],
        default="all",
        help="Type of analysis to generate",
    )
    parser.add_argument(
        "--team",
        type=int,
        default=None,
        help="Team ID for team profile (required if type=team)",
    )
    parser.add_argument(
        "--matchup",
        type=str,
        default=None,
        help="Comma-separated team IDs for matchup analysis (required if type=matchup)",
    )
    parser.add_argument(
        "-r",
        "--region",
        type=str,
        default=None,
        help="Region identifier for region analysis (required if type=region)",
    )

    args = parser.parse_args()

    # Process matchup IDs if provided
    matchup_ids = None
    if args.matchup:
        try:
            team1_id, team2_id = map(int, args.matchup.split(","))
            matchup_ids = (team1_id, team2_id)
        except:
            print(
                "Error: --matchup should be two comma-separated team IDs (e.g., 1234,5678)"
            )
            sys.exit(1)

    # Generate the analysis
    generate_enhanced_analysis(
        gender_code=args.gender,
        year=args.year,
        output_dir=args.output,
        analysis_type=args.type,
        team_id=args.team,
        matchup_ids=matchup_ids,
        region=args.region,
    )
