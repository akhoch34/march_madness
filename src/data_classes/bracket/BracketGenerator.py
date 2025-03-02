
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
from binarytree import Node
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

from src.data_classes.processing import MarchMadnessPredictor
from src.data_classes.bracket.seed_slots import SLOTS


class BracketNode(Node):
    """Extended Node class with parent reference and team information"""
    def __init__(self, value, left=None, right=None, parent=None):
        super().__init__(value, left=left, right=right)
        self.parent: BracketNode = parent
        self.team_id: int = None
        self.seed: int = None
        self.team_name: str = None
        self.win_prob: float = None
        
    def __setattr__(self, name, value):
        """Override to maintain parent references when children are added"""
        if name in ['left', 'right'] and value is not None and isinstance(value, BracketNode):
            value.parent = self
        super().__setattr__(name, value)


class BracketSimulator:
    """Class for simulating and visualizing tournament brackets"""
    
    def __init__(self, predictor: MarchMadnessPredictor=None):
        """
        Initialize the bracket simulator
        
        Parameters:
        predictor: MarchMadnessPredictor instance (optional)
        """
        self.predictor = predictor
        self.teams_df: pd.DataFrame = None
        self.seeds_df: pd.DataFrame = None
        self.slots_df: pd.DataFrame = None
        self.current_season: int = None
        self.bracket_tree: BracketNode = None
        self.seed_slot_map: dict[int, Tuple[int, int]] = None
        
        # Default slot coordinates (you can replace with your existing dictionary)
        self.slot_coordinates = SLOTS
    
    def load_data(self, teams_path, seeds_path, slots_path):
        """Load tournament data from CSV files"""
        self.teams_df = pd.read_csv(teams_path)
        self.seeds_df = pd.read_csv(seeds_path)
        self.slots_df = pd.read_csv(slots_path)
        
        # Convert column names to lowercase
        self.teams_df.columns = [col.lower() for col in self.teams_df.columns]
        self.seeds_df.columns = [col.lower() for col in self.seeds_df.columns]
        self.slots_df.columns = [col.lower() for col in self.slots_df.columns]
    
    def use_predictor_data(self, season=None):
        """Use data from the attached predictor"""
        if self.predictor is None:
            raise ValueError("No predictor attached. Call set_predictor() first.")
        
        # Get the current season if not specified
        if season is None:
            season = self.predictor.current_season
        
        self.current_season = season
        
        # Get data from the predictor
        self.teams_df = self.predictor.data_manager.data['teams'].copy()
        self.seeds_df = self.predictor.data_manager.data['tourney_seeds'].copy()
        
        # Check if slots data is available (might need to be loaded separately)
        if 'tourney_slots' in self.predictor.data_manager.data:
            self.slots_df = self.predictor.data_manager.data['tourney_slots'].copy()
        else:
            raise ValueError("Tournament slots data not found in predictor. Please load separately.")
    
    def set_predictor(self, predictor):
        """Set or update the predictor instance"""
        self.predictor = predictor
    
    def build_bracket_tree(self, season=None):
        """
        Build the bracket tree structure from tournament slot data
        
        Parameters:
        season: Tournament season to build (default: current_season)
        
        Returns:
        root_node: Root node of the bracket tree
        seed_slot_map: Dictionary mapping slot numbers to seed strings
        """
        if season is None:
            season = self.current_season
        
        if self.slots_df is None:
            raise ValueError("Slot data not loaded. Call load_data() or use_predictor_data() first.")
        
        print(self.slots_df.columns)
        # Filter slots for the specified season
        s = self.slots_df[self.slots_df['Season'] == season]
        
        if len(s) == 0:
            raise ValueError(f"No slot data found for season {season}")
        
        # Create the bracket tree
        seed_slot_map = {0: 'R6CH'}  # Root is the championship
        root = BracketNode(0)
        
        counter = 1
        current_nodes = [root]
        
        # Build the tree by processing the slots
        while current_nodes:
            next_nodes = []
            
            for node in current_nodes:
                # Check if this node has child slots
                slots = s[s['Slot'] == seed_slot_map[node.value]]
                
                if len(slots) > 0:
                    # Create left and right children
                    node.left = BracketNode(counter)
                    node.right = BracketNode(counter + 1)
                    
                    # Map children to their slots
                    seed_slot_map[counter] = slots.iloc[0]['StrongSeed']
                    seed_slot_map[counter + 1] = slots.iloc[0]['WeakSeed']
                    
                    # Add children to next level processing
                    next_nodes.append(node.left)
                    next_nodes.append(node.right)
                    
                    counter += 2
            
            current_nodes = next_nodes
        
        # Store the results
        self.bracket_tree = root
        self.seed_slot_map = seed_slot_map
        
        return root, seed_slot_map
    
    def simulate_bracket(self, method='ensemble'):
        """
        Simulate the tournament bracket using the predictor
        
        Parameters:
        method: Prediction method to use ('elo', 'ml', or 'ensemble')
        
        Returns:
        slot_data: List of (coordinates, text) tuples for visualization
        """
        if self.predictor is None:
            raise ValueError("No predictor attached. Call set_predictor() first.")
        
        if self.bracket_tree is None or self.seed_slot_map is None:
            self.build_bracket_tree()
        
        # Get the tournament teams and seeds
        teams_df = self.teams_df
        seeds_df = self.seeds_df[self.seeds_df['Season'] == self.current_season]
        
        # Map seeds to team IDs
        seed_team_map = dict(zip(seeds_df['Seed'], seeds_df['TeamID']))
        
        # Helper function to find team ID for a seed
        def get_team_id(seed_slot):
            seed = self.seed_slot_map[seed_slot]
            return seed_team_map.get(seed)
        
        # Helper function to get prediction for a matchup
        def predict_matchup(team1_id, team2_id):
            if team1_id is None or team2_id is None:
                return 0.5  # Default if we don't have both teams
            
            # Ensure team1_id < team2_id for consistency
            if team1_id > team2_id:
                team1_id, team2_id = team2_id, team1_id
                is_reversed = True
            else:
                is_reversed = False
            
            # Get prediction from predictor
            pred = self.predictor.predict_game(
                team1_id, team2_id, 
                day_num=134,  # First round of tournament
                season=self.current_season,
                method=method
            )
            
            # Adjust if we reversed the teams
            return 1 - pred if is_reversed else pred
        
        # Solve the bracket by traversing the tree from bottom up
        # We'll fill in the leaf nodes first, then work our way up
        levels = list(reversed(self.bracket_tree.levels))
        
        for level in levels:
            # Process pairs of nodes at this level
            for i in range(0, len(level), 2):
                if i + 1 >= len(level):  # Skip if no pair
                    continue
                
                left_node = level[i]
                right_node = level[i + 1]
                
                # If this is a leaf node, get the teams
                if left_node.left is None:
                    left_node.team_id = get_team_id(left_node.value)
                    left_node.seed = self.seed_slot_map[left_node.value]
                    if left_node.team_id:
                        left_node.team_name = teams_df[teams_df['TeamID'] == left_node.team_id]['TeamName'].iloc[0]
                
                if right_node.left is None:
                    right_node.team_id = get_team_id(right_node.value)
                    right_node.seed = self.seed_slot_map[right_node.value]
                    if right_node.team_id:
                        right_node.team_name = teams_df[teams_df['TeamID'] == right_node.team_id]['TeamName'].iloc[0]
                
                # If both teams are known, predict the winner
                if left_node.team_id is not None and right_node.team_id is not None:
                    # Predict the game
                    win_prob = predict_matchup(left_node.team_id, right_node.team_id)
                    
                    # Store the probability
                    left_node.win_prob = win_prob
                    right_node.win_prob = 1 - win_prob
                    
                    # Determine the winner
                    if win_prob > 0.5:
                        winner = left_node
                    else:
                        winner = right_node
                    
                    # Advance the winner to the parent node
                    if left_node.parent is not None:  # Should be the same as right_node.parent
                        parent = left_node.parent
                        parent.team_id = winner.team_id
                        parent.seed = winner.seed
                        parent.team_name = winner.team_name
        
        # Generate the slot data for visualization
        slot_data = []
        
        # Flatten the tree
        all_nodes = [node for level in self.bracket_tree.levels for node in level]
        
        # Generate text for each node
        for node in all_nodes:
            # Get the coordinates for this slot
            slot_num = len(self.slot_coordinates) - node.value
            coords = self.slot_coordinates.get(slot_num, (0, 0))
            
            # Generate text
            if node.team_name:
                # Show seed, team name, and probability if it's a non-root node
                if node.parent is not None and node.win_prob is not None:
                    prob_text = f" {node.win_prob:.1%}"
                else:
                    prob_text = ""
                
                text = f"{node.seed[1:]} {node.team_name}{prob_text}"
            else:
                text = ""
            
            slot_data.append((coords, text))
        
        return slot_data
    
    def visualize_bracket(self, method='ensemble', output_path=None, show_plot=True, betting_odds=False):
        """
        Visualize the tournament bracket
        
        Parameters:
        method: Prediction method to use
        output_path: Path to save the image (optional)
        show_plot: Whether to display the plot (only works in interactive environments)
        betting_odds: Whether to show betting odds instead of probabilities
        
        Returns:
        fig: Matplotlib figure object
        """
        # Simulate the bracket
        slot_data = self.simulate_bracket(method=method)
        
        # Load the empty bracket template
        empty_bracket_path = "./empty.jpg"  # Update with your path
        
        try:
            img = Image.open(empty_bracket_path)
        except:
            # Create a blank image if template not found
            img = Image.new('RGB', (940, 700), color='white')
            print(f"Warning: Empty bracket template not found at {empty_bracket_path}. Using blank image.")
        
        # Draw on the image
        draw = ImageDraw.Draw(img)
        
        # Try to use a font if available
        try:
            font = ImageFont.truetype("Helvetica", 10)
        except:
            font = None
        
        # Draw each team and prediction
        for coords, text in slot_data:
            draw.text(coords, text, fill=(0, 0, 0), font=font)
        
        # Convert to numpy array for matplotlib
        img_array = np.array(img)
        
        # Create a matplotlib figure
        dpi = 30
        height, width, _ = img_array.shape
        figsize = (width / dpi, height / dpi)
        
        # Use plt.ioff() to avoid showing the figure if not requested
        import matplotlib.pyplot as plt
        # plt.ioff()  # Turn off interactive mode
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(img_array)
        ax.axis('off')
        
        # Set title
        title = f"{self.current_season} March Madness Bracket Prediction ({method})"
        if betting_odds:
            title += " - Betting Odds"
        plt.title(title)
        
        # Save if requested
        if output_path:
            # Check if output_path is a file path or a BytesIO object
            if isinstance(output_path, str):
                plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
                print(f"Saved bracket to {output_path}")
            else:
                # Assume it's a BytesIO object
                plt.savefig(output_path, format='png', bbox_inches='tight', dpi=dpi)
        
        # Show if requested and in an interactive environment
        if show_plot:
            try:
                plt.tight_layout()
                plt.ion()  # Turn on interactive mode
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display plot: {str(e)}")
        
        plt.close(fig)  # Close the figure to free memory
        
        return fig
    
    def get_team_path(self, team_id):
        """
        Get a team's path through the tournament
        
        Parameters:
        team_id: ID of the team to track
        
        Returns:
        path: List of dictionaries with round, opponent, and win probability
        """
        if self.bracket_tree is None:
            raise ValueError("Bracket not simulated yet. Call simulate_bracket() first.")
        
        # Find the leaf node for this team
        all_nodes = [node for level in self.bracket_tree.levels for node in level]
        team_node = None
        
        for node in all_nodes:
            if node.team_id == team_id and node.left is None:  # Leaf node
                team_node = node
                break
        
        if team_node is None:
            raise ValueError(f"Team ID {team_id} not found in the bracket")
        
        # Track the path up through the bracket
        path = []
        current = team_node
        
        while current.parent is not None:
            # Find the opponent (sibling node)
            parent = current.parent
            opponent = parent.left if current == parent.right else parent.right
            
            # Determine the round name
            level_index = self.bracket_tree.levels.index(level for level in self.bracket_tree.levels if current in level)
            round_name = self._get_round_name(level_index)
            
            # Add to path
            path.append({
                "round": round_name,
                "opponent_id": opponent.team_id,
                "opponent_name": opponent.team_name,
                "opponent_seed": opponent.seed,
                "win_probability": current.win_prob
            })
            
            # Move up to the next round
            current = parent
        
        return path
    
    def _get_round_name(self, level_index):
        """Convert a level index to a round name"""
        # Adjust based on your tournament structure
        round_names = {
            0: "Championship",
            1: "Final Four",
            2: "Elite Eight",
            3: "Sweet Sixteen",
            4: "Round of 32",
            5: "First Round",
            6: "Play-In"
        }
        return round_names.get(level_index, f"Round {level_index}")
