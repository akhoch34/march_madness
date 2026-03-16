"""
Shared lightweight predictor/data-manager used by bracket_analysis.py and streamlit_app.py.

SimpleDataManager  — minimal data access (teams, seeds, slots)
SimplePredictor    — wraps a pre-generated submission CSV for bracket simulation / analysis
"""

import math
import re

import pandas as pd


class SimpleDataManager:
    def __init__(self, teams_df, seeds_df, slots_df, results_df=None):
        self.data = {
            "teams": teams_df,
            "tourney_seeds": seeds_df,
            "tourney_slots": slots_df,
        }
        if results_df is not None:
            self.data["tourney_results"] = results_df

        self.seed_lookup = {}
        for _, row in seeds_df.iterrows():
            self.seed_lookup[(row["Season"], row["TeamID"])] = row["Seed"]

    def get_team_name(self, team_id):
        team_row = self.data["teams"][self.data["teams"]["TeamID"] == team_id]
        if len(team_row) > 0:
            return team_row["TeamName"].iloc[0]
        return f"Team {team_id}"

    def get_seed(self, season, team_id):
        return self.seed_lookup.get((season, team_id), None)

    def get_team_id_from_seed(self, season, seed):
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
        results_df=None,
        current_season=2026,
    ):
        self.current_season = current_season
        self.data_manager = SimpleDataManager(teams_df, seeds_df, slots_df, results_df)
        self.predictions_df = predictions_df
        self.elo_df = elo_df
        self.stats_df = stats_df

    def predict_game(self, team1_id, team2_id, day_num=None, season=None, method=None):
        if season is None:
            season = self.current_season
        matchup_id = f"{season}_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"
        match = self.predictions_df[self.predictions_df["ID"] == matchup_id]
        if len(match) == 0:
            return 0.5
        pred = match["Pred"].iloc[0]
        if team1_id > team2_id:
            pred = 1.0 - pred
        return pred

    def get_elo_ratings(self, team1_id, team2_id):
        if self.elo_df is not None:
            team1_elo = self.elo_df.get(team1_id, 1500)
            team2_elo = self.elo_df.get(team2_id, 1500)
            return team1_elo, team2_elo
        return 1500, 1500

    def generate_matchup_explanation(self, team1_id, team2_id, day_num=None):
        prediction = self.predict_game(team1_id, team2_id, day_num)
        team1_name = self.data_manager.get_team_name(team1_id)
        team2_name = self.data_manager.get_team_name(team2_id)
        team1_seed = self.data_manager.get_seed(self.current_season, team1_id)
        team2_seed = self.data_manager.get_seed(self.current_season, team2_id)
        team1_elo, team2_elo = self.get_elo_ratings(team1_id, team2_id)

        if prediction > 0.5:
            favorite = {"id": team1_id, "name": team1_name, "seed": team1_seed, "elo": team1_elo}
            underdog = {"id": team2_id, "name": team2_name, "seed": team2_seed, "elo": team2_elo}
            win_prob = prediction
        else:
            favorite = {"id": team2_id, "name": team2_name, "seed": team2_seed, "elo": team2_elo}
            underdog = {"id": team1_id, "name": team1_name, "seed": team1_seed, "elo": team1_elo}
            win_prob = 1 - prediction

        try:
            favorite_seed_num = int(re.search(r"\d+", favorite["seed"]).group()) if favorite["seed"] else 16
            underdog_seed_num = int(re.search(r"\d+", underdog["seed"]).group()) if underdog["seed"] else 16
        except Exception:
            favorite_seed_num = 8
            underdog_seed_num = 9

        seed_diff = abs(favorite_seed_num - underdog_seed_num)
        elo_diff = favorite["elo"] - underdog["elo"]
        explanation_parts = []

        if seed_diff > 0:
            if favorite_seed_num < underdog_seed_num:
                if seed_diff >= 10:
                    seed_text = f"{favorite['name']} ({favorite['seed']}) is a major favorite as a much higher seed than {underdog['name']} ({underdog['seed']})."
                elif seed_diff >= 5:
                    seed_text = f"As a #{favorite_seed_num} seed, {favorite['name']} has a significant seeding advantage over #{underdog_seed_num} seed {underdog['name']}."
                else:
                    seed_text = f"{favorite['name']} has a slight edge as a #{favorite_seed_num} seed versus #{underdog_seed_num} seed {underdog['name']}."
            else:
                seed_text = f"Despite being a lower #{favorite_seed_num} seed, {favorite['name']} is favored over #{underdog_seed_num} seed {underdog['name']}."
            explanation_parts.append(seed_text)
        else:
            explanation_parts.append(
                f"In this #{favorite_seed_num} vs #{underdog_seed_num} matchup, {favorite['name']} has the edge over {underdog['name']}."
            )

        if abs(elo_diff) > 50:
            if elo_diff > 200:
                explanation_parts.append(
                    f"{favorite['name']} has a substantially higher ELO rating ({favorite['elo']:.0f} vs {underdog['elo']:.0f}), indicating significantly better season-long performance."
                )
            else:
                explanation_parts.append(
                    f"{favorite['name']} has a higher ELO rating ({favorite['elo']:.0f} vs {underdog['elo']:.0f}), indicating better season-long performance."
                )

        historical_matchups = {
            (1, 16): 98.7, (2, 15): 93.8, (3, 14): 85.2, (4, 13): 79.6,
            (5, 12): 64.9, (6, 11): 62.3, (7, 10): 60.9, (8, 9): 51.4,
        }
        if seed_diff > 0 and favorite_seed_num < underdog_seed_num:
            matchup_key = (favorite_seed_num, underdog_seed_num)
            if matchup_key in historical_matchups:
                hist_pct = historical_matchups[matchup_key]
                explanation_parts.append(
                    f"Historically, #{favorite_seed_num} seeds have won {hist_pct:.2f}% of games against #{underdog_seed_num} seeds in the tournament."
                )

        if self.stats_df is not None:
            favorite_stats = self.stats_df.get(favorite["id"], {})
            underdog_stats = self.stats_df.get(underdog["id"], {})
            if favorite_stats and underdog_stats:
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

        return " ".join(explanation_parts)

    def american_odds(self, probability):
        if probability > 0.5:
            return -round(probability / (1 - probability) * 100)
        else:
            return round((1 - probability) / probability * 100)

    def win_probability_to_spread(self, win_probability, std_dev=11.0, calibration=1.8, tournament_mode=True):
        win_probability = min(max(win_probability, 0.01), 0.99)
        logit = math.log(win_probability / (1 - win_probability))
        point_spread = -logit * std_dev / calibration
        if tournament_mode and win_probability > 0.85:
            extra_factor = (win_probability - 0.85) * 2.5
            point_spread = point_spread * (1 + extra_factor)
        return point_spread

    def format_spread(self, point_spread):
        rounded_spread = round(point_spread * 2) / 2
        if rounded_spread < 0:
            return f"{rounded_spread}"
        elif rounded_spread > 0:
            return f"+{rounded_spread}"
        else:
            return "PK"
