import streamlit as st
from src.data_classes.bracket.BracketGenerator import BracketSimulator
from src.data_classes.processing.Predictor import MarchMadnessPredictor
from io import BytesIO
import pandas as pd

DATA_DIR = "../../../data/{year}"
GENDER = "M"
CURRENT_SEASON = 2025

def create_streamlit_app():
    """Create a Streamlit app for bracket visualization"""
    
    st.title("March Madness Bracket Simulator")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    # Method selection
    method = st.sidebar.selectbox(
        "Prediction Method",
        ["elo", "ml", "ensemble"],
        index=2
    )
    
    # Year selection (if we have multiple years)
    available_years = [2021, 2022, 2023, 2024, 2025]
    year = st.sidebar.selectbox(
        "Tournament Year",
        available_years,
        index=len(available_years)-1
    )
    
    # Create simulator and predictor
    @st.cache_resource
    def get_predictor():
        predictor = MarchMadnessPredictor(
            data_dir=DATA_DIR.format(year=year), 
            gender=GENDER, 
            current_season=year
        )
        predictor.initialize_models(train_ml=True)
        return predictor
    
    # Generate button
    if st.sidebar.button("Generate Bracket"):
        with st.spinner("Simulating bracket..."):
            predictor = get_predictor()
            simulator = BracketSimulator(predictor)
            
            try:
                simulator.use_predictor_data(year)
                simulator.build_bracket_tree()
                
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
                    file_name=f"bracket_{year}_{method}.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error generating bracket: {str(e)}")
    
    # Team explorer section
    st.header("Team Explorer")
    
    predictor = get_predictor()
    if "bracket_simulated" not in st.session_state:
        st.session_state.bracket_simulated = False
    
    # Only show team explorer if bracket has been simulated
    if st.session_state.bracket_simulated:
        # Get tournament teams
        tournament_teams = predictor.data_manager.data['tourney_seeds'][
            predictor.data_manager.data['tourney_seeds']['Season'] == year
        ].merge(
            predictor.data_manager.data['teams'], 
            left_on='TeamID', 
            right_on='TeamID'
        )
        
        # Team selection
        selected_team = st.selectbox(
            "Select Team",
            tournament_teams['TeamName'].tolist()
        )
        
        # Display team details and path
        team_id = tournament_teams[tournament_teams['TeamName'] == selected_team]['TeamID'].iloc[0]
        
        simulator.use_predictor_data(year)
        simulator.build_bracket_tree()
        simulator.simulate_bracket(method=method)
        
        try:
            # Get the team's path
            path = simulator.get_team_path(team_id)
            
            if path:
                st.subheader(f"{selected_team} Tournament Path")
                
                path_df = pd.DataFrame([
                    {
                        "Round": p["round"],
                        "Opponent": p["opponent_name"],
                        "Opponent Seed": p["opponent_seed"],
                        "Win Probability": f"{p['win_probability']:.1%}"
                    }
                    for p in path
                ])
                
                st.table(path_df)
        except:
            st.info("Team path information not available.")
    else:
        st.info("Generate a bracket to explore team details.")

if __name__ == "__main__":
    create_streamlit_app()
