#!/usr/bin/env python3
"""
March Madness Tournament Matchup Analysis

This script generates a comprehensive round-by-round analysis of all tournament matchups,
including team info, ELO ratings, win probabilities, implied betting odds, and explanations.

It can be used in two ways:
1. As a standalone script from the command line
2. Imported into a Jupyter notebook to work with an existing predictor object

Command line usage:
    python bracket_analysis.py [-g GENDER] [-m METHOD] [-y YEAR] [-o OUTPUT]

Arguments:
    -g, --gender    Gender: "M" for men's tournament, "W" for women's (default: M)
    -m, --method    Prediction method: "elo" or "elo_enhanced" (default: elo_enhanced)
    -y, --year      Tournament year (default: 2025)
    -o, --output    Output file path (default: matchup_analysis_{gender}_{year}.md)

Jupyter notebook usage:
    from bracket_analysis import generate_analysis_from_predictor

    # Using existing predictor
    output_file = generate_analysis_from_predictor(
        predictor=my_predictor,
        method="elo_enhanced",
        output_file="my_analysis.md"
    )
"""

import os
import re
import math
import argparse
import pandas as pd

# Constants
SUBMISSION_DIR = "./output"
DATA_DIR = "../data/{year}"


# Create a simple DataManager class with the minimal required methods
class SimpleDataManager:
    def __init__(self, teams_df, seeds_df, slots_df):
        self.data = {
            "teams": teams_df,
            "tourney_seeds": seeds_df,
            "tourney_slots": slots_df,
        }

        # Create seed lookup dictionary for convenience
        self.seed_lookup = {}
        for _, row in seeds_df.iterrows():
            self.seed_lookup[(row["Season"], row["TeamID"])] = row["Seed"]

    def get_team_name(self, team_id):
        """Get team name from team ID"""
        team_row = self.data["teams"][self.data["teams"]["TeamID"] == team_id]
        if len(team_row) > 0:
            return team_row["TeamName"].iloc[0]
        return f"Team {team_id}"

    def get_seed(self, season, team_id):
        """Get team seed for a specific season"""
        return self.seed_lookup.get((season, team_id), None)

    def get_team_id_from_seed(self, season, seed):
        """Get team ID from a seed"""
        seed_row = self.data["tourney_seeds"][
            (self.data["tourney_seeds"]["Season"] == season)
            & (self.data["tourney_seeds"]["Seed"] == seed)
        ]
        if len(seed_row) > 0:
            return seed_row["TeamID"].iloc[0]
        return None


class SimplePredictor:
    def __init__(
        self,
        teams_df,
        seeds_df,
        slots_df,
        predictions_df,
        elo_df=None,
        stats_df=None,
        current_season=2025,
    ):
        self.current_season = current_season
        self.data_manager = SimpleDataManager(teams_df, seeds_df, slots_df)
        self.predictions_df = predictions_df
        self.elo_df = elo_df
        self.stats_df = stats_df

    def predict_game(self, team1_id, team2_id, day_num=None, season=None, method=None):
        """Custom prediction function using pre-generated predictions"""
        if season is None:
            season = self.current_season

        matchup_id = f"{season}_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"
        match = self.predictions_df[self.predictions_df["ID"] == matchup_id]

        if len(match) == 0:
            # No prediction found, return default
            return 0.5

        pred = match["Pred"].iloc[0]

        # Adjust if team order is reversed in the matchup ID
        if team1_id > team2_id:
            pred = 1.0 - pred

        return pred

    def get_elo_ratings(self, team1_id, team2_id):
        """Get ELO ratings for two teams if available"""
        if self.elo_df is not None:
            team1_elo = self.elo_df.get(team1_id, 1500)
            team2_elo = self.elo_df.get(team2_id, 1500)
            return team1_elo, team2_elo
        return 1500, 1500

    def generate_matchup_explanation(self, team1_id, team2_id, day_num=None):
        """Generate a narrative explanation for why one team is favored over the other"""
        # Get prediction and teams info
        prediction = self.predict_game(team1_id, team2_id, day_num)

        # Team names and seeds
        team1_name = self.data_manager.get_team_name(team1_id)
        team2_name = self.data_manager.get_team_name(team2_id)
        team1_seed = self.data_manager.get_seed(self.current_season, team1_id)
        team2_seed = self.data_manager.get_seed(self.current_season, team2_id)

        # Get ELO ratings if available
        team1_elo, team2_elo = self.get_elo_ratings(team1_id, team2_id)

        # Determine favorite and underdog
        if prediction > 0.5:
            favorite = {
                "id": team1_id,
                "name": team1_name,
                "seed": team1_seed,
                "elo": team1_elo,
            }
            underdog = {
                "id": team2_id,
                "name": team2_name,
                "seed": team2_seed,
                "elo": team2_elo,
            }
            win_prob = prediction
        else:
            favorite = {
                "id": team2_id,
                "name": team2_name,
                "seed": team2_seed,
                "elo": team2_elo,
            }
            underdog = {
                "id": team1_id,
                "name": team1_name,
                "seed": team1_seed,
                "elo": team1_elo,
            }
            win_prob = 1 - prediction

        # Extract seed numbers
        try:
            favorite_seed_num = (
                int(re.search(r"\d+", favorite["seed"]).group())
                if favorite["seed"]
                else 16
            )
            underdog_seed_num = (
                int(re.search(r"\d+", underdog["seed"]).group())
                if underdog["seed"]
                else 16
            )
        except:
            favorite_seed_num = 8
            underdog_seed_num = 9

        seed_diff = abs(favorite_seed_num - underdog_seed_num)
        elo_diff = favorite["elo"] - underdog["elo"]

        # Generate explanation components
        explanation_parts = []

        # Seed-based component
        if seed_diff > 0:
            if favorite_seed_num < underdog_seed_num:
                # Higher seed is favored (as expected)
                if seed_diff >= 10:
                    seed_text = f"{favorite['name']} ({favorite['seed']}) is a major favorite as a much higher seed than {underdog['name']} ({underdog['seed']})."
                elif seed_diff >= 5:
                    seed_text = f"As a #{favorite_seed_num} seed, {favorite['name']} has a significant seeding advantage over #{underdog_seed_num} seed {underdog['name']}."
                else:
                    seed_text = f"{favorite['name']} has a slight edge as a #{favorite_seed_num} seed versus #{underdog_seed_num} seed {underdog['name']}."
            else:
                # Lower seed is favored (upset potential)
                seed_text = f"Despite being a lower #{favorite_seed_num} seed, {favorite['name']} is favored over #{underdog_seed_num} seed {underdog['name']}."

            explanation_parts.append(seed_text)
        else:
            # Equal seeds
            explanation_parts.append(
                f"In this #{favorite_seed_num} vs #{underdog_seed_num} matchup, {favorite['name']} has the edge over {underdog['name']}."
            )

        # ELO ratings component if significant difference
        if abs(elo_diff) > 50:
            if elo_diff > 200:
                elo_text = f"{favorite['name']} has a substantially higher ELO rating ({favorite['elo']:.0f} vs {underdog['elo']:.0f}), indicating significantly better season-long performance."
            else:
                elo_text = f"{favorite['name']} has a higher ELO rating ({favorite['elo']:.0f} vs {underdog['elo']:.0f}), indicating better season-long performance."
            explanation_parts.append(elo_text)

        # Historical seed matchup data
        if seed_diff > 0 and favorite_seed_num < underdog_seed_num:
            historical_matchups = {
                (1, 16): 98.7,
                (2, 15): 93.8,
                (3, 14): 85.2,
                (4, 13): 79.6,
                (5, 12): 64.9,
                (6, 11): 62.3,
                (7, 10): 60.9,
                (8, 9): 51.4,
            }

            matchup_key = (favorite_seed_num, underdog_seed_num)
            if matchup_key in historical_matchups:
                hist_pct = historical_matchups[matchup_key]
                explanation_parts.append(
                    f"Historically, #{favorite_seed_num} seeds have won {hist_pct:.2f}% of games against #{underdog_seed_num} seeds in the tournament."
                )

        # Generate statistics comparison if available
        if self.stats_df is not None:
            # If we have team stats, add advanced metrics comparisons
            favorite_stats = self.stats_df.get(favorite["id"], {})
            underdog_stats = self.stats_df.get(underdog["id"], {})

            if favorite_stats and underdog_stats:
                # Offensive efficiency comparison
                if "OffEff" in favorite_stats and "OffEff" in underdog_stats:
                    off_diff = favorite_stats["OffEff"] - underdog_stats["OffEff"]
                    if abs(off_diff) > 5:
                        if off_diff > 0:
                            explanation_parts.append(
                                f"{favorite['name']} has a more efficient offense ({favorite_stats['OffEff']:.1f} vs {underdog_stats['OffEff']:.1f})."
                            )
                        else:
                            explanation_parts.append(
                                f"While {underdog['name']} actually has a more efficient offense ({underdog_stats['OffEff']:.1f} vs {favorite_stats['OffEff']:.1f}), other factors favor {favorite['name']}."
                            )

                # Defensive efficiency comparison
                if "DefEff" in favorite_stats and "DefEff" in underdog_stats:
                    def_diff = (
                        underdog_stats["DefEff"] - favorite_stats["DefEff"]
                    )  # Lower is better for defense
                    if abs(def_diff) > 5:
                        if def_diff > 0:
                            explanation_parts.append(
                                f"{favorite['name']} has a stronger defense, allowing {favorite_stats['DefEff']:.1f} points per 100 possessions compared to {underdog_stats['DefEff']:.1f} for {underdog['name']}."
                            )
                        else:
                            explanation_parts.append(
                                f"Although {underdog['name']} has a better defensive efficiency ({underdog_stats['DefEff']:.1f} vs {favorite_stats['DefEff']:.1f}), other factors favor {favorite['name']}."
                            )

                # 3-point shooting
                if "3P%" in favorite_stats and "3P%" in underdog_stats:
                    three_diff = favorite_stats["3P%"] - underdog_stats["3P%"]
                    if (
                        abs(three_diff) > 0.03
                    ):  # 3% difference in 3PT shooting is significant
                        if three_diff > 0:
                            explanation_parts.append(
                                f"{favorite['name']} has a significant advantage in 3-point shooting ({favorite_stats['3P%']*100:.2f}% vs {underdog_stats['3P%']*100:.2f}%)."
                            )
                        else:
                            explanation_parts.append(
                                f"{underdog['name']} is the better 3-point shooting team ({underdog_stats['3P%']*100:.2f}% vs {favorite_stats['3P%']*100:.2f}%)."
                            )

        # Win probability phrasing based on confidence
        prob_pct = win_prob * 100
        if prob_pct >= 90:
            prob_text = f"{favorite['name']} is strongly favored with a {prob_pct:.2f}% win probability."
        elif prob_pct >= 70:
            prob_text = f"{favorite['name']} is the clear favorite with a {prob_pct:.2f}% chance to win."
        elif prob_pct >= 60:
            prob_text = f"{favorite['name']} has the advantage with a {prob_pct:.2f}% win probability."
        else:
            prob_text = f"This is expected to be a close matchup, with {favorite['name']} having a slight edge ({prob_pct:.2f}% win probability)."

        explanation_parts.append(prob_text)

        # Combine all explanation parts
        explanation = " ".join(explanation_parts)
        return explanation

    def american_odds(self, probability):
        """Convert win probability to American betting odds"""
        if probability > 0.5:
            # Favorite: how much you need to bet to win $100
            return -round(probability / (1 - probability) * 100)
        else:
            # Underdog: how much you win if you bet $100
            return round((1 - probability) / probability * 100)

    def win_probability_to_spread(
        self, win_probability, std_dev=11.0, calibration=1.8, tournament_mode=True
    ):
        """
        Convert a win probability to an implied point spread for basketball.

        Parameters:
        win_probability (float): Probability that Team 1 will win (between 0 and 1)
        std_dev (float): Standard deviation of scoring margin (default 11 for college basketball)
        calibration (float): Calibration factor for the conversion (lower = larger spreads)
        tournament_mode (bool): Enable special handling for tournament mismatches

        Returns:
        float: Implied point spread (negative means Team 1 is favored)
        """
        # Ensure win probability is within valid range
        win_probability = min(max(win_probability, 0.01), 0.99)

        # Convert win probability to a spread using the logit function
        logit = math.log(win_probability / (1 - win_probability))

        # Base calculation
        point_spread = -logit * std_dev / calibration

        # Special handling for tournament mode extreme mismatches
        if tournament_mode and win_probability > 0.85:
            # Apply additional scaling for very high probabilities
            # This helps match the observed spreads in tournament games
            extra_factor = (win_probability - 0.85) * 2.5
            point_spread = point_spread * (1 + extra_factor)

        return point_spread

    def format_spread(self, point_spread):
        """Format a point spread with proper betting notation"""
        # Round to nearest half point (common in betting markets)
        rounded_spread = round(point_spread * 2) / 2

        if rounded_spread < 0:
            # Team is favored
            return f"{rounded_spread}"
        elif rounded_spread > 0:
            # Team is underdog
            return f"+{rounded_spread}"
        else:
            # Pick'em (even odds)
            return "PK"


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


def load_predictions(gender_code, method, year=2025):
    """Load pre-generated predictions from CSV file"""
    if method == "elo":
        file_path = os.path.join(
            SUBMISSION_DIR, f"submission_{year}_{gender_code}_ELO.csv"
        )
    else:
        file_path = os.path.join(SUBMISSION_DIR, f"submission_{year}_{gender_code}.csv")

    if not os.path.exists(file_path):
        print(
            f"Warning: Prediction file not found: {file_path}. Falling back to main predictions file."
        )
        file_path = os.path.join(SUBMISSION_DIR, f"submission_{year}.csv")

    predictions_df = pd.read_csv(file_path)
    return predictions_df


def load_elo_data(gender_code, year=2025):
    """Load ELO data if available"""
    try:
        file_path = os.path.join(SUBMISSION_DIR, f"team_elo_{gender_code}_{year}.csv")
        if os.path.exists(file_path):
            elo_df = pd.read_csv(file_path)
            # Convert to dictionary for easier lookup
            elo_dict = dict(zip(elo_df["TeamID"], elo_df["ELO"]))
            return elo_dict
    except:
        pass
    return None


def get_tournament_structure(slots_df, seeds_df, year):
    """Build tournament structure from slots data"""
    # Get the tournament regions
    regions = []
    for seed in seeds_df["Seed"]:
        region = seed[0]  # First character of seed is region
        if region not in regions:
            regions.append(region)

    # Build rounds structure
    rounds = {
        "R1": [],  # First Round
        "R2": [],  # Second Round
        "R3": [],  # Sweet 16
        "R4": [],  # Elite 8
        "R5": [],  # Final Four
        "R6": [],  # Championship
    }

    # Process slots to get matchups for each round
    slot_team_map = {}

    # First fill in the initial slots from seeds
    for _, row in seeds_df.iterrows():
        slot_team_map[row["Seed"]] = row["TeamID"]

    # Now process the slots from first round to championship
    for r in range(1, 7):
        round_key = f"R{r}"

        # Get slots for this round
        round_slots = slots_df[slots_df["Slot"].str.startswith(round_key)]

        # Add each matchup to the rounds structure
        for _, slot in round_slots.iterrows():
            strong_seed = slot["StrongSeed"]
            weak_seed = slot["WeakSeed"]

            rounds[round_key].append((strong_seed, weak_seed, slot["Slot"]))

            # Update the slot_team_map for future rounds
            # In the real bracket we would compute the winner, but for analysis we track both possibilities
            slot_team_map[slot["Slot"]] = (
                slot_team_map.get(strong_seed, strong_seed),
                slot_team_map.get(weak_seed, weak_seed),
            )

    return rounds, slot_team_map


def generate_bracket_analysis(
    gender_code="M", method="elo_enhanced", year=2025, output_file=None
):
    """Generate a comprehensive round-by-round analysis of all tournament matchups"""
    # Create output file path if not provided
    if output_file is None:
        output_file = f"matchup_analysis_{gender_code}_{year}.md"

    # Load data
    teams_df, seeds_df, slots_df = load_data(gender_code, year)
    predictions_df = load_predictions(gender_code, method, year)
    elo_dict = load_elo_data(gender_code, year)

    # Create predictor
    predictor = SimplePredictor(
        teams_df=teams_df,
        seeds_df=seeds_df,
        slots_df=slots_df,
        predictions_df=predictions_df,
        elo_df=elo_dict,
        current_season=year,
    )

    # Generate tournament structure
    rounds, slot_team_map = get_tournament_structure(slots_df, seeds_df, year)

    # Generate the analysis markdown
    with open(output_file, "w") as f:
        # Write the header
        f.write(f"# {year} March Madness Tournament Analysis\n\n")
        f.write(f"**Gender:** {'Men' if gender_code == 'M' else 'Women'}\n")
        f.write(f"**Prediction Method:** {method}\n\n")

        # Process each round
        round_names = {
            "R1": "First Round",
            "R2": "Second Round (Round of 32)",
            "R3": "Sweet 16",
            "R4": "Elite 8",
            "R5": "Final Four",
            "R6": "Championship",
        }

        # Process rounds in order
        for round_key in ["R1", "R2", "R3", "R4", "R5", "R6"]:
            round_name = round_names[round_key]
            f.write(f"## {round_name}\n\n")

            # Process each matchup in this round
            for strong_seed, weak_seed, slot in rounds[round_key]:
                # For Round 1, we have the actual seeds
                if round_key == "R1":
                    team1_id = predictor.data_manager.get_team_id_from_seed(
                        year, strong_seed
                    )
                    team2_id = predictor.data_manager.get_team_id_from_seed(
                        year, weak_seed
                    )

                    if team1_id is None or team2_id is None:
                        continue  # Skip if teams aren't set yet

                    team1_name = predictor.data_manager.get_team_name(team1_id)
                    team2_name = predictor.data_manager.get_team_name(team2_id)

                    # Write matchup header
                    f.write(
                        f"### {strong_seed} {team1_name} vs {weak_seed} {team2_name}\n\n"
                    )

                    # Get prediction and odds
                    win_prob = predictor.predict_game(team1_id, team2_id)
                    team1_elo, team2_elo = predictor.get_elo_ratings(team1_id, team2_id)

                    american_odds = predictor.american_odds(win_prob)
                    spread = predictor.win_probability_to_spread(win_prob)
                    formatted_spread = predictor.format_spread(spread)

                    # Write prediction info with formatted percentages
                    f.write(
                        f"**Prediction:** {team1_name} has a {win_prob:.2%} chance to win\n\n"
                    )
                    f.write(
                        f"**ELO Ratings:** {team1_name}: {team1_elo:.0f}, {team2_name}: {team2_elo:.0f}\n\n"
                    )
                    f.write(
                        f"**Betting Odds:** {team1_name} {formatted_spread}, Moneyline: {american_odds}\n\n"
                    )

                    # Generate explanation
                    explanation = predictor.generate_matchup_explanation(
                        team1_id, team2_id
                    )
                    f.write(f"**Analysis:** {explanation}\n\n")

                    # Add a separator
                    f.write("---\n\n")
                else:
                    # For later rounds, we don't have fixed teams yet
                    # We can list the possible matchups from slot_team_map
                    if slot in slot_team_map:
                        possible_team1_ids, possible_team2_ids = slot_team_map[slot]

                        if isinstance(possible_team1_ids, tuple):
                            team1_options = possible_team1_ids
                        else:
                            team1_options = [possible_team1_ids]

                        if isinstance(possible_team2_ids, tuple):
                            team2_options = possible_team2_ids
                        else:
                            team2_options = [possible_team2_ids]

                        # Write potential matchup header
                        f.write(f"### {slot} Matchup: Potential Teams\n\n")

                        # List potential teams for slot 1
                        f.write(f"**{strong_seed} Slot Possibilities:**\n\n")
                        for team_id in team1_options:
                            if isinstance(team_id, int):
                                team_name = predictor.data_manager.get_team_name(
                                    team_id
                                )
                                team_seed = predictor.data_manager.get_seed(
                                    year, team_id
                                )
                                if team_seed:
                                    f.write(f"- {team_seed} {team_name}\n")
                                else:
                                    f.write(f"- {team_name}\n")
                        f.write("\n")

                        # List potential teams for slot 2
                        f.write(f"**{weak_seed} Slot Possibilities:**\n\n")
                        for team_id in team2_options:
                            if isinstance(team_id, int):
                                team_name = predictor.data_manager.get_team_name(
                                    team_id
                                )
                                team_seed = predictor.data_manager.get_seed(
                                    year, team_id
                                )
                                if team_seed:
                                    f.write(f"- {team_seed} {team_name}\n")
                                else:
                                    f.write(f"- {team_name}\n")
                        f.write("\n")

                        # Add a separator
                        f.write("---\n\n")

    print(f"Analysis written to {output_file}")
    return output_file


# New function for use with a predictor in Jupyter notebooks
def generate_analysis_from_predictor(
    predictor, method="elo_enhanced", output_file=None, year=None
):
    """
    Generate tournament analysis using an existing MarchMadnessPredictor object

    Parameters:
    predictor: Existing MarchMadnessPredictor object with data already loaded
    method: Prediction method to use ('elo' or 'elo_enhanced')
    output_file: Output file path (default: matchup_analysis_{gender}_{year}.md)
    year: Tournament year (default: predictor.current_season)

    Returns:
    str: Path to the output file
    """
    # Use predictor's current_season if year not provided
    if year is None:
        year = predictor.current_season

    # Use male/female code from predictor if possible
    gender_code = getattr(predictor, "gender", "M")

    # Create output file path if not provided
    if output_file is None:
        output_file = f"matchup_analysis_{gender_code}_{year}.md"

    # Extract data from the predictor
    teams_df = predictor.data_manager.data["teams"]
    seeds_df = predictor.data_manager.data["tourney_seeds"]
    slots_df = predictor.data_manager.data["tourney_slots"]

    # Filter for current season
    season_seeds_df = seeds_df[seeds_df["Season"] == year].copy()
    season_slots_df = slots_df[slots_df["Season"] == year].copy()

    # Create or use prediction data
    if hasattr(predictor, "predictions_df") and predictor.predictions_df is not None:
        # Use existing predictions if available
        predictions_df = predictor.predictions_df
    else:
        # Generate fresh predictions for all tournament teams
        predictions_df = predictor.generate_predictions(
            method=method, tournament_teams_only=True
        )

    # Create ELO dictionary from predictor's ELO system
    elo_dict = {}
    for team_id in teams_df["TeamID"].unique():
        elo_dict[team_id] = predictor.elo_system.get_team_elo(year, team_id)

    # Create stats dictionary from predictor's stats calculator if available
    stats_dict = None
    if (
        hasattr(predictor, "stats_calculator")
        and hasattr(predictor.stats_calculator, "advanced_team_stats")
        and predictor.stats_calculator.advanced_team_stats
    ):
        # Extract stats for the current season
        if year in predictor.stats_calculator.advanced_team_stats:
            stats_dict = predictor.stats_calculator.advanced_team_stats[year]

    # Create a wrapper predictor that uses our predict function but accesses the real predictor
    class WrapperPredictor(SimplePredictor):
        def predict_game(
            self, team1_id, team2_id, day_num=None, season=None, method_override=None
        ):
            # Use the original predictor's predict_game method
            return predictor.predict_game(
                team1_id,
                team2_id,
                day_num=day_num or 134,
                season=season or year,
                method=method_override or method,
            )

        def get_elo_ratings(self, team1_id, team2_id):
            # Get ELO ratings directly from predictor's ELO system
            team1_elo = predictor.elo_system.get_team_elo(year, team1_id)
            team2_elo = predictor.elo_system.get_team_elo(year, team2_id)
            return team1_elo, team2_elo

    # Create the wrapper predictor
    wrapper_predictor = WrapperPredictor(
        teams_df=teams_df,
        seeds_df=season_seeds_df,
        slots_df=season_slots_df,
        predictions_df=predictions_df,
        elo_df=elo_dict,
        stats_df=stats_dict,
        current_season=year,
    )

    # Generate tournament structure
    rounds, slot_team_map = get_tournament_structure(
        season_slots_df, season_seeds_df, year
    )

    # Generate the analysis markdown
    with open(output_file, "w") as f:
        # Write the header
        f.write(f"# {year} March Madness Tournament Analysis\n\n")
        f.write(f"**Gender:** {'Men' if gender_code == 'M' else 'Women'}\n")
        f.write(f"**Prediction Method:** {method}\n\n")

        # Process each round
        round_names = {
            "R1": "First Round",
            "R2": "Second Round (Round of 32)",
            "R3": "Sweet 16",
            "R4": "Elite 8",
            "R5": "Final Four",
            "R6": "Championship",
        }

        # Process rounds in order
        for round_key in ["R1", "R2", "R3", "R4", "R5", "R6"]:
            round_name = round_names[round_key]
            f.write(f"## {round_name}\n\n")

            # Process each matchup in this round
            for strong_seed, weak_seed, slot in rounds[round_key]:
                # For Round 1, we have the actual seeds
                if round_key == "R1":
                    team1_id = wrapper_predictor.data_manager.get_team_id_from_seed(
                        year, strong_seed
                    )
                    team2_id = wrapper_predictor.data_manager.get_team_id_from_seed(
                        year, weak_seed
                    )

                    if team1_id is None or team2_id is None:
                        continue  # Skip if teams aren't set yet

                    team1_name = wrapper_predictor.data_manager.get_team_name(team1_id)
                    team2_name = wrapper_predictor.data_manager.get_team_name(team2_id)

                    # Write matchup header
                    f.write(
                        f"### {strong_seed} {team1_name} vs {weak_seed} {team2_name}\n\n"
                    )

                    # Get prediction and odds
                    win_prob = wrapper_predictor.predict_game(team1_id, team2_id)
                    team1_elo, team2_elo = wrapper_predictor.get_elo_ratings(
                        team1_id, team2_id
                    )

                    american_odds = wrapper_predictor.american_odds(win_prob)
                    spread = wrapper_predictor.win_probability_to_spread(win_prob)
                    formatted_spread = wrapper_predictor.format_spread(spread)

                    # Write prediction info with formatted percentages
                    f.write(
                        f"**Prediction:** {team1_name} has a {win_prob:.2%} chance to win\n\n"
                    )
                    f.write(
                        f"**ELO Ratings:** {team1_name}: {team1_elo:.0f}, {team2_name}: {team2_elo:.0f}\n\n"
                    )
                    f.write(
                        f"**Betting Odds:** {team1_name} {formatted_spread}, Moneyline: {american_odds}\n\n"
                    )

                    # Get advanced stats comparison if available
                    if stats_dict is not None:
                        # Get offensive and defensive efficiency if available
                        team1_stats = stats_dict.get(team1_id, {})
                        team2_stats = stats_dict.get(team2_id, {})

                        if team1_stats and team2_stats:
                            f.write("**Advanced Stats Comparison:**\n\n")

                            # Create stats table
                            f.write(
                                "| Metric | {} | {} |\n".format(team1_name, team2_name)
                            )
                            f.write(
                                "|--------|{}|{}|\n".format(
                                    "-" * len(team1_name), "-" * len(team2_name)
                                )
                            )

                            # Add key stats
                            for stat, label, is_percent in [
                                ("OffEff", "Offensive Efficiency", False),
                                ("DefEff", "Defensive Efficiency", False),
                                ("NetEff", "Net Efficiency", False),
                                ("eFG%", "Effective FG%", True),
                                ("TOV%", "Turnover %", True),
                                ("ORB%", "Offensive Rebound %", True),
                                ("FTRate", "FT Rate", False),
                                ("3P%", "3-Point %", True),
                                ("FG%", "Field Goal %", True),
                                ("FT%", "Free Throw %", True),
                            ]:
                                if stat in team1_stats and stat in team2_stats:
                                    t1_val = team1_stats[stat]
                                    t2_val = team2_stats[stat]

                                    # Format numbers as percentages if needed
                                    if is_percent:
                                        f.write(
                                            f"| {label} | {t1_val*100:.2f}% | {t2_val*100:.2f}% |\n"
                                        )
                                    else:
                                        f.write(
                                            f"| {label} | {t1_val:.1f} | {t2_val:.1f} |\n"
                                        )

                            f.write("\n")

                    # Generate explanation
                    explanation = wrapper_predictor.generate_matchup_explanation(
                        team1_id, team2_id
                    )
                    f.write(f"**Analysis:** {explanation}\n\n")

                    # Add a separator
                    f.write("---\n\n")
                else:
                    # For later rounds, we don't have fixed teams yet
                    # We can list the possible matchups from slot_team_map
                    if slot in slot_team_map:
                        possible_team1_ids, possible_team2_ids = slot_team_map[slot]

                        if isinstance(possible_team1_ids, tuple):
                            team1_options = possible_team1_ids
                        else:
                            team1_options = [possible_team1_ids]

                        if isinstance(possible_team2_ids, tuple):
                            team2_options = possible_team2_ids
                        else:
                            team2_options = [possible_team2_ids]

                        # Write potential matchup header
                        f.write(f"### {slot} Matchup: Potential Teams\n\n")

                        # List potential teams for slot 1
                        f.write(f"**{strong_seed} Slot Possibilities:**\n\n")
                        for team_id in team1_options:
                            if isinstance(team_id, int):
                                team_name = (
                                    wrapper_predictor.data_manager.get_team_name(
                                        team_id
                                    )
                                )
                                team_seed = wrapper_predictor.data_manager.get_seed(
                                    year, team_id
                                )
                                if team_seed:
                                    f.write(f"- {team_seed} {team_name}\n")
                                else:
                                    f.write(f"- {team_name}\n")
                        f.write("\n")

                        # List potential teams for slot 2
                        f.write(f"**{weak_seed} Slot Possibilities:**\n\n")
                        for team_id in team2_options:
                            if isinstance(team_id, int):
                                team_name = (
                                    wrapper_predictor.data_manager.get_team_name(
                                        team_id
                                    )
                                )
                                team_seed = wrapper_predictor.data_manager.get_seed(
                                    year, team_id
                                )
                                if team_seed:
                                    f.write(f"- {team_seed} {team_name}\n")
                                else:
                                    f.write(f"- {team_name}\n")
                        f.write("\n")

                        # Add a separator
                        f.write("---\n\n")

    print(f"Analysis written to {output_file}")
    return output_file


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate March Madness tournament matchup analysis"
    )
    parser.add_argument(
        "-g",
        "--gender",
        choices=["M", "W"],
        default="M",
        help="Gender: M for men's tournament, W for women's",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=["elo", "elo_enhanced"],
        default="elo_enhanced",
        help="Prediction method",
    )
    parser.add_argument("-y", "--year", type=int, default=2025, help="Tournament year")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output file path"
    )

    args = parser.parse_args()

    # Generate the analysis
    output_file = generate_bracket_analysis(
        gender_code=args.gender,
        method=args.method,
        year=args.year,
        output_file=args.output,
    )

    print(f"Analysis complete. Results written to {output_file}")
