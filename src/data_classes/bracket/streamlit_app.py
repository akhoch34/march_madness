import streamlit as st
from src.data_classes.bracket.BracketGenerator import BracketSimulator
from io import BytesIO
import pandas as pd
import os
import numpy as np
import re

SUBMISSION_DIR = "/Users/conor/Documents/march_madness/src"
DATA_DIR = "/Users/conor/Documents/march_madness/data/{year}"

# Create a simple class with the minimal required methods for BracketSimulator
class SimplePredictor:
    def __init__(self, teams_df, seeds_df, slots_df, predictions_df, stats_df=None, current_season=2025):
        self.current_season = current_season
        self.data_manager = SimpleDataManager(teams_df, seeds_df, slots_df)
        self.predictions_df = predictions_df
        self.stats_df = stats_df
    
    def predict_game(self, team1_id, team2_id, day_num=None, season=None, method=None):
        """Custom prediction function using pre-generated predictions"""
        if season is None:
            season = self.current_season
            
        matchup_id = f"{season}_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"
        match = self.predictions_df[self.predictions_df['ID'] == matchup_id]
        
        if len(match) == 0:
            # No prediction found, return default
            return 0.5
        
        pred = match['Pred'].iloc[0]
        
        # Adjust if team order is reversed in the matchup ID
        if team1_id > team2_id:
            pred = 1.0 - pred
            
        return pred
        
    def generate_matchup_explanation(self, team1_id, team2_id, day_num=None):
        """Generate a narrative explanation for why one team is favored over the other"""
        # Get prediction and teams info
        prediction = self.predict_game(team1_id, team2_id, day_num)
        
        # Team names and seeds
        team1_name = self.data_manager.get_team_name(team1_id)
        team2_name = self.data_manager.get_team_name(team2_id)
        team1_seed = self.data_manager.get_seed(self.current_season, team1_id)
        team2_seed = self.data_manager.get_seed(self.current_season, team2_id)
        
        # Determine favorite and underdog
        if prediction > 0.5:
            favorite = {"id": team1_id, "name": team1_name, "seed": team1_seed}
            underdog = {"id": team2_id, "name": team2_name, "seed": team2_seed}
            win_prob = prediction
        else:
            favorite = {"id": team2_id, "name": team2_name, "seed": team2_seed}
            underdog = {"id": team1_id, "name": team1_name, "seed": team1_seed}
            win_prob = 1 - prediction
        
        # Extract seed numbers
        try:
            favorite_seed_num = int(re.search(r'\d+', favorite["seed"]).group()) if favorite["seed"] else 16
            underdog_seed_num = int(re.search(r'\d+', underdog["seed"]).group()) if underdog["seed"] else 16
        except:
            favorite_seed_num = 8
            underdog_seed_num = 9
        
        seed_diff = abs(favorite_seed_num - underdog_seed_num)
        
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
            explanation_parts.append(f"In this #{favorite_seed_num} vs #{underdog_seed_num} matchup, {favorite['name']} has the edge over {underdog['name']}.")
        
        # Historical seed matchup data
        if seed_diff > 0 and favorite_seed_num < underdog_seed_num:
            historical_matchups = {
                (1,16): 98.7, (2,15): 93.8, (3,14): 85.2, (4,13): 79.6,
                (5,12): 64.9, (6,11): 62.3, (7,10): 60.9, (8,9): 51.4
            }
            
            matchup_key = (favorite_seed_num, underdog_seed_num)
            if matchup_key in historical_matchups:
                hist_pct = historical_matchups[matchup_key]
                explanation_parts.append(f"Historically, #{favorite_seed_num} seeds have won {hist_pct}% of games against #{underdog_seed_num} seeds in the tournament.")
        
        # Generate statistics comparison if available
        if self.stats_df is not None:
            # Implementation would compare team statistics from self.stats_df
            pass
        
        # Win probability phrasing based on confidence
        prob_pct = win_prob * 100
        if prob_pct >= 90:
            prob_text = f"{favorite['name']} is strongly favored with a {prob_pct:.1f}% win probability."
        elif prob_pct >= 70:
            prob_text = f"{favorite['name']} is the clear favorite with a {prob_pct:.1f}% chance to win."
        elif prob_pct >= 60:
            prob_text = f"{favorite['name']} has the advantage with a {prob_pct:.1f}% win probability."
        else:
            prob_text = f"This is expected to be a close matchup, with {favorite['name']} having a slight edge ({prob_pct:.1f}% win probability)."
        
        explanation_parts.append(prob_text)
        
        # Combine all explanation parts
        explanation = " ".join(explanation_parts)
        return explanation

class SimpleDataManager:
    def __init__(self, teams_df, seeds_df, slots_df):
        self.data = {
            'teams': teams_df,
            'tourney_seeds': seeds_df,
            'tourney_slots': slots_df
        }
        
        # Create seed lookup dictionary for convenience
        self.seed_lookup = {}
        for _, row in seeds_df.iterrows():
            self.seed_lookup[(row['Season'], row['TeamID'])] = row['Seed']
            
    def get_team_name(self, team_id):
        """Get team name from team ID"""
        team_row = self.data['teams'][self.data['teams']['TeamID'] == team_id]
        if len(team_row) > 0:
            return team_row['TeamName'].iloc[0]
        return f"Team {team_id}"
    
    def get_seed(self, season, team_id):
        """Get team seed for a specific season"""
        return self.seed_lookup.get((season, team_id), None)

def load_data(gender_code, year):
    """Load tournament data files"""
    # Path to data files
    teams_path = os.path.join(DATA_DIR.format(year=year), "MTeams.csv" if gender_code == "M" else "WTeams.csv")
    seeds_path = os.path.join(DATA_DIR.format(year=year), "MNCAATourneySeeds.csv" if gender_code == "M" else "WNCAATourneySeeds.csv")
    slots_path = os.path.join(DATA_DIR.format(year=year), "MNCAATourneySlots.csv" if gender_code == "M" else "WNCAATourneySlots.csv")
    
    # Load data
    teams_df = pd.read_csv(teams_path)
    seeds_df = pd.read_csv(seeds_path)
    slots_df = pd.read_csv(slots_path)
    
    # Filter for current season
    seeds_df = seeds_df[seeds_df['Season'] == year]
    slots_df = slots_df[slots_df['Season'] == year]
    
    return teams_df, seeds_df, slots_df

def create_streamlit_app():
    """Create a Streamlit app for bracket visualization"""
    
    st.title("March Madness Bracket Simulator")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Prediction Method",
        ["elo", "elo_enhanced"],
        index=1
    )
    
    # Gender selection
    gender = st.sidebar.selectbox(
        "Gender",
        ["Men's", "Women's"],
        index=0
    )
    
    gender_code = "M" if gender == "Men's" else "W"
    
    # Year selection (if we have multiple years)
    available_years = [2025]
    year = st.sidebar.selectbox(
        "Tournament Year",
        available_years,
        index=0
    )
    
    @st.cache_data
    def load_predictions(gender_code, method):
        """Load pre-generated predictions from CSV file"""
        if method == "elo":
            file_path = os.path.join(SUBMISSION_DIR, f"submission_2025_{gender_code}_ELO.csv")
        else:
            file_path = os.path.join(SUBMISSION_DIR, f"submission_2025_{gender_code}.csv")
            
        if not os.path.exists(file_path):
            st.warning(f"Prediction file not found: {file_path}. Falling back to main predictions file.")
            file_path = os.path.join(SUBMISSION_DIR, "submission_2025.csv")
            
        predictions_df = pd.read_csv(file_path)
        return predictions_df
    
    # Generate button
    if st.sidebar.button("Generate Bracket"):
        with st.spinner("Loading data and simulating bracket..."):
            try:
                # Load tournament data
                teams_df, seeds_df, slots_df = load_data(gender_code, year)
                
                # Load predictions
                predictions_df = load_predictions(gender_code, method)
                
                # Create a simple predictor
                predictor = SimplePredictor(teams_df, seeds_df, slots_df, predictions_df, current_season=year)
                
                # Create and configure the simulator
                simulator = BracketSimulator(predictor)
                simulator.teams_df = teams_df
                simulator.seeds_df = seeds_df
                simulator.slots_df = slots_df
                simulator.current_season = year
                
                # Build the bracket tree
                simulator.build_bracket_tree()
                
                # Store in session state for team explorer
                st.session_state.bracket_simulated = True
                st.session_state.simulator = simulator
                st.session_state.predictor = predictor
                st.session_state.method = method
                st.session_state.year = year
                st.session_state.gender_code = gender_code
                st.session_state.teams_df = teams_df
                st.session_state.seeds_df = seeds_df
                
                # Visualize bracket
                fig = simulator.visualize_bracket(method=method, show_plot=False)
                st.pyplot(fig)
                
                # Add download button
                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="Download Bracket",
                    data=buf,
                    file_name=f"bracket_{year}_{gender_code}_{method}.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error generating bracket: {str(e)}")
                st.exception(e)
    
    # Team explorer section
    st.header("Team Explorer")
    
    # Check if bracket has been simulated
    if "bracket_simulated" not in st.session_state:
        st.session_state.bracket_simulated = False
    
    # Only show team explorer if bracket has been simulated
    if st.session_state.bracket_simulated:
        # Use stored variables from session state
        simulator = st.session_state.simulator
        teams_df = st.session_state.teams_df
        seeds_df = st.session_state.seeds_df
        year = st.session_state.year
        
        # Get tournament teams
        tournament_teams = seeds_df.merge(teams_df, on='TeamID')
        
        # Team selection
        selected_team = st.selectbox(
            "Select Team",
            tournament_teams['TeamName'].tolist()
        )
        
        # Display team details and path
        team_id = tournament_teams[tournament_teams['TeamName'] == selected_team]['TeamID'].iloc[0]
        seed = tournament_teams[tournament_teams['TeamID'] == team_id]['Seed'].iloc[0]
        
        st.write(f"Exploring {gender} tournament path for {seed[1:]} seed {selected_team} (ID: {team_id})")
        
        try:
            # Get the team's path
            path = simulator.get_team_path(team_id)
            
            if path:
                st.subheader(f"{selected_team} Tournament Path")
                
                path_df = pd.DataFrame([
                    {
                        "Round": p["round"],
                        "Opponent": p["opponent_name"],
                        "Opponent Seed": p["opponent_seed"][1:] if p["opponent_seed"] else "N/A",
                        "Win Probability": f"{p['win_probability']:.1%}" if p['win_probability'] else "N/A"
                    }
                    for p in path
                ])
                
                st.table(path_df)
                
                # Generate narrative for the matchup
                st.subheader("Matchup Analysis")
                
                for p in path:
                    if p["opponent_id"] is not None:
                        explanation = predictor.generate_matchup_explanation(team_id, p["opponent_id"])
                        st.write(f"**{p['round']}:** {explanation}")
        except Exception as e:
            st.error(f"Error getting team path: {str(e)}")
            st.info("This functionality will work once the tournament bracket is released.")
    else:
        st.info("Generate a bracket to explore team details.")

if __name__ == "__main__":
    create_streamlit_app()