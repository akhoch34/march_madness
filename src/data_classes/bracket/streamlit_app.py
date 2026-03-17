import streamlit as st
from src.data_classes.bracket.BracketGenerator import BracketSimulator
from src.data_classes.simple_predictor import SimpleDataManager, SimplePredictor
from io import BytesIO
import pandas as pd
import os
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBMISSION_DIR = os.path.join(_REPO_ROOT, "src")
DATA_DIR = os.path.join(_REPO_ROOT, "data", "{year}")

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