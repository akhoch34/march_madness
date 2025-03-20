import math
import os
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
from binarytree import Node
import pandas as pd
import numpy as np

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
        self.odds: float = None
        self.next_round_style = None  # Add this new property

    def __setattr__(self, name, value):
        """Override to maintain parent references when children are added"""
        if (
            name in ["left", "right"]
            and value is not None
            and isinstance(value, BracketNode)
        ):
            value.parent = self
        super().__setattr__(name, value)


class BracketSimulator:
    """Class for simulating and visualizing tournament brackets"""

    def __init__(self, predictor: MarchMadnessPredictor = None):
        """
        Initialize the bracket simulator

        Parameters:
        predictor: MarchMadnessPredictor instance (optional)
        bracket_template_path: Path to the empty bracket template file (optional)
        """
        self.predictor = predictor
        self.teams_df: pd.DataFrame = None
        self.seeds_df: pd.DataFrame = None
        self.slots_df: pd.DataFrame = None
        self.current_season: int = None
        self.bracket_tree: BracketNode = None
        self.seed_slot_map: dict[int, Tuple[int, int]] = None

        # Get the directory where this class is defined
        self.class_dir = os.path.dirname(os.path.abspath(__file__))

        self.bracket_template_path = os.path.join(self.class_dir, "empty.jpg")

        # Default slot coordinates
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
        self.teams_df = self.predictor.data_manager.data["teams"].copy()
        self.seeds_df = self.predictor.data_manager.data["tourney_seeds"].copy()

        # Check if slots data is available (might need to be loaded separately)
        if "tourney_slots" in self.predictor.data_manager.data:
            self.slots_df = self.predictor.data_manager.data["tourney_slots"].copy()
        else:
            raise ValueError(
                "Tournament slots data not found in predictor. Please load separately."
            )

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
            raise ValueError(
                "Slot data not loaded. Call load_data() or use_predictor_data() first."
            )

        # Filter slots for the specified season
        s = self.slots_df[self.slots_df["Season"] == season]

        if len(s) == 0:
            raise ValueError(f"No slot data found for season {season}")

        # Create the bracket tree
        seed_slot_map = {0: "R6CH"}  # Root is the championship
        root = BracketNode(0)

        counter = 1
        current_nodes = [root]

        # Build the tree by processing the slots
        while current_nodes:
            next_nodes = []

            for node in current_nodes:
                # Check if this node has child slots
                slots = s[s["Slot"] == seed_slot_map[node.value]]

                if len(slots) > 0:
                    # Create left and right children
                    node.left = BracketNode(counter)
                    node.right = BracketNode(counter + 1)

                    # Map children to their slots
                    seed_slot_map[counter] = slots.iloc[0]["StrongSeed"]
                    seed_slot_map[counter + 1] = slots.iloc[0]["WeakSeed"]

                    # Add children to next level processing
                    next_nodes.append(node.left)
                    next_nodes.append(node.right)

                    counter += 2

            current_nodes = next_nodes

        # Store the results
        self.bracket_tree = root
        self.seed_slot_map = seed_slot_map

        return root, seed_slot_map

    def simulate_bracket(self, method="ensemble", betting_odds=False):
        """
        Simulate the tournament bracket using the predictor

        Parameters:
        method: Prediction method to use ('elo', 'ml', or 'ensemble')
        betting_odds: Whether to show betting odds instead of probabilities

        Returns:
        slot_data: List of (coordinates, text) tuples for visualization
        """
        if self.predictor is None:
            raise ValueError("No predictor attached. Call set_predictor() first.")

        if self.bracket_tree is None or self.seed_slot_map is None:
            self.build_bracket_tree()

        # Get the tournament teams and seeds
        teams_df = self.teams_df
        seeds_df = self.seeds_df[self.seeds_df["Season"] == self.current_season]

        # Map seeds to team IDs
        seed_team_map = dict(zip(seeds_df["Seed"], seeds_df["TeamID"]))

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
                team1_id,
                team2_id,
                day_num=134,  # First round of tournament
                season=self.current_season,
                method=method,
            )

            # Adjust if we reversed the teams
            return 1 - pred if is_reversed else pred

        def win_probability_to_spread(
            win_probability, std_dev=11.0, calibration=1.8, tournament_mode=True
        ):
            """
            Convert a win probability to an implied point spread for basketball
            with special handling for extreme mismatches in tournament games.
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

        def format_spread(point_spread):
            """Format a point spread with standard betting notation."""
            # Round to nearest half point (common in betting markets)
            rounded_spread = round(point_spread * 2) / 2

            if rounded_spread < 0:
                # Team is favored
                return f" ({rounded_spread})"
            elif rounded_spread > 0:
                # Team is underdog
                return f" (+{rounded_spread})"
            else:
                # Pick'em (even odds)
                return f" (PK)"

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
                        left_node.team_name = teams_df[
                            teams_df["TeamID"] == left_node.team_id
                        ]["TeamName"].iloc[0]

                if right_node.left is None:
                    right_node.team_id = get_team_id(right_node.value)
                    right_node.seed = self.seed_slot_map[right_node.value]
                    if right_node.team_id:
                        right_node.team_name = teams_df[
                            teams_df["TeamID"] == right_node.team_id
                        ]["TeamName"].iloc[0]

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
                        if betting_odds:
                            winner.odds = win_probability_to_spread(winner.win_prob)
                            right_node.odds = None
                    else:
                        winner = right_node
                        if betting_odds:
                            winner.odds = win_probability_to_spread(winner.win_prob)
                            left_node.odds = None

                    # Advance the winner to the parent node
                    if (
                        left_node.parent is not None
                    ):  # Should be the same as right_node.parent
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
                    if betting_odds:
                        if node.odds is not None:
                            prob_text = format_spread(node.odds)
                        else:
                            prob_text = ""
                    else:
                        prob_text = f" {node.win_prob:.1%}"
                else:
                    prob_text = ""

                text = f"{node.seed[1:]} {node.team_name}{prob_text}"
            else:
                text = ""

            slot_data.append((coords, text))

        return slot_data

    def simulate_historical_bracket(self, season, method="elo_enhanced"):
        """
        Simulate a historical tournament bracket comparing predicted vs actual results

        Parameters:
        season: Historical season to simulate
        method: Prediction method to use

        Returns:
        tuple: (slot_data for visualization, accuracy metrics)
        """
        if self.predictor is None:
            raise ValueError("No predictor attached. Call set_predictor() first.")

        # Build the bracket tree
        if (
            self.bracket_tree is None
            or self.seed_slot_map is None
            or self.current_season != season
        ):
            self.current_season = season
            self.build_bracket_tree(season)

        # Get the tournament teams and seeds
        teams_df = self.teams_df
        seeds_df = self.seeds_df[self.seeds_df["Season"] == season]

        # Get actual tournament results
        tourney_results = self.predictor.data_manager.data["tourney_results"]
        season_results = tourney_results[tourney_results["Season"] == season]

        # Create a lookup of actual game results
        actual_results = {}
        for _, game in season_results.iterrows():
            winner_id = game["WTeamID"]
            loser_id = game["LTeamID"]
            # Create a key with both teams (order doesn't matter when checking)
            game_key1 = (winner_id, loser_id)
            game_key2 = (loser_id, winner_id)
            actual_results[game_key1] = winner_id
            actual_results[game_key2] = winner_id  # Same winner, different key order

        # Map seeds to team IDs
        seed_team_map = dict(zip(seeds_df["Seed"], seeds_df["TeamID"]))

        # Helper function to find team ID for a seed
        def get_team_id(seed_slot):
            seed = self.seed_slot_map[seed_slot]
            return seed_team_map.get(seed)

        # Helper function to get prediction for a matchup
        def predict_matchup(team1_id, team2_id):
            if team1_id is None or team2_id is None:
                return 0.5  # Default if we don't have both teams

            # Get prediction from predictor
            pred = self.predictor.predict_game(
                team1_id,
                team2_id,
                day_num=134,  # First round of tournament
                season=season,
                method=method,
            )

            return pred

        # Solve the bracket using actual results when available
        levels = list(reversed(self.bracket_tree.levels))
        correct_predictions = 0
        total_predictions = 0

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
                        left_node.team_name = teams_df[
                            teams_df["TeamID"] == left_node.team_id
                        ]["TeamName"].iloc[0]

                if right_node.left is None:
                    right_node.team_id = get_team_id(right_node.value)
                    right_node.seed = self.seed_slot_map[right_node.value]
                    if right_node.team_id:
                        right_node.team_name = teams_df[
                            teams_df["TeamID"] == right_node.team_id
                        ]["TeamName"].iloc[0]

                # If both teams are known, predict the winner
                if left_node.team_id is not None and right_node.team_id is not None:
                    # Predict the game
                    win_prob = predict_matchup(left_node.team_id, right_node.team_id)

                    # Store the probability
                    left_node.win_prob = win_prob
                    right_node.win_prob = 1 - win_prob

                    # Determine predicted winner
                    predicted_winner = left_node if win_prob > 0.5 else right_node
                    predicted_loser = right_node if win_prob > 0.5 else left_node

                    # Determine actual winner if this game has been played
                    game_key = (left_node.team_id, right_node.team_id)
                    if game_key in actual_results:
                        actual_winner_id = actual_results[game_key]
                        actual_winner = (
                            left_node
                            if actual_winner_id == left_node.team_id
                            else right_node
                        )

                        # Check if prediction was correct
                        prediction_correct = predicted_winner == actual_winner

                        # Store style information for the parent node (where the winner advances to)
                        if left_node.parent is not None:
                            parent = left_node.parent
                            if prediction_correct:
                                parent.next_round_style = {
                                    "color": "green",
                                    "strikethrough": False,
                                    "actual_winner": None,
                                }
                            else:
                                # Show predicted winner with strikethrough, then actual winner
                                parent.next_round_style = {
                                    "color": "red",
                                    "strikethrough": True,
                                    "predicted_team": f"{predicted_winner.seed[1:]} {predicted_winner.team_name}",
                                    "actual_winner": f"{actual_winner.seed[1:]} {actual_winner.team_name}",
                                }

                        # Track accuracy
                        correct_predictions += 1 if prediction_correct else 0
                        total_predictions += 1

                        # Advance the ACTUAL winner to the parent node
                        if left_node.parent is not None:
                            parent = left_node.parent
                            parent.team_id = actual_winner.team_id
                            parent.seed = actual_winner.seed
                            parent.team_name = actual_winner.team_name

        # Generate the slot data for visualization
        slot_data = []
        all_nodes = [node for level in self.bracket_tree.levels for node in level]

        # Generate text for each node
        for node in all_nodes:
            slot_num = len(self.slot_coordinates) - node.value
            coords = self.slot_coordinates.get(slot_num, (0, 0))

            if node.team_name:
                # Base text always includes seed and team name
                if hasattr(node, "next_round_style") and node.next_round_style:
                    if node.next_round_style["strikethrough"]:
                        # For incorrect predictions, use the predicted team name
                        base_text = node.next_round_style["predicted_team"]
                    else:
                        # For correct predictions or non-styled nodes, use actual team name
                        base_text = f"{node.seed[1:]} {node.team_name}"
                else:
                    base_text = f"{node.seed[1:]} {node.team_name}"

                # Add probability if available
                if node.parent is not None and node.win_prob is not None:
                    base_text += f" {node.win_prob:.1%}"

                # Get styling information
                if hasattr(node, "next_round_style") and node.next_round_style:
                    color = node.next_round_style["color"]
                    strikethrough = node.next_round_style["strikethrough"]
                    actual_winner = node.next_round_style.get("actual_winner")
                else:
                    color = "black"
                    strikethrough = False
                    actual_winner = None

                slot_data.append(
                    (coords, base_text, color, strikethrough, actual_winner)
                )
            else:
                slot_data.append((coords, "", "black", False, None))

        # Calculate accuracy
        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )

        return slot_data, {
            "accuracy": accuracy,
            "correct": correct_predictions,
            "total": total_predictions,
        }

    # Shared helper methods
    def _prepare_bracket_image(self):
        """Prepare the base bracket image for drawing"""
        try:
            img = Image.open(self.bracket_template_path)
        except:
            # Create a blank image if template not found
            img = Image.new("RGB", (940, 700), color="white")
            print(
                f"Warning: Empty bracket template not found at {self.bracket_template_path}. Using blank image."
            )

        # Create drawing object
        draw = ImageDraw.Draw(img)

        # Try to use a font if available
        try:
            font = ImageFont.truetype("Helvetica", 10)
        except:
            font = None

        return img, draw, font

    def _draw_bracket_data(self, draw, slot_data, font, is_historical=False):
        """Draw team information on the bracket image"""
        if not is_historical:
            # Handle regular bracket visualization
            for coords, text in slot_data:
                draw.text(coords, text, fill=(0, 0, 0), font=font)
            return

        def get_text_dimensions(text, font):
            """Helper function to get text dimensions that works with all PIL versions"""
            if hasattr(font, "getbbox"):  # Newer PIL versions
                bbox = font.getbbox(text)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            elif hasattr(font, "getsize"):  # Older PIL versions
                return font.getsize(text)
            else:  # Fallback if no font or missing methods
                return len(text) * 6, 10

        # Handle historical bracket visualization
        for coords, text, color, strikethrough, actual_winner in slot_data:
            if not text:  # Skip empty slots
                continue

            # Convert color names to RGB
            color_map = {"black": (0, 0, 0), "red": (255, 0, 0), "green": (0, 128, 0)}
            rgb_color = color_map.get(color, (0, 0, 0))

            # Get text dimensions for strikethrough
            if font:
                text_width, text_height = get_text_dimensions(text, font)
            else:
                text_width = len(text) * 6  # Approximate width
                text_height = 10  # Approximate height

            # Draw the text
            draw.text(coords, text, fill=rgb_color, font=font)

            # Draw strikethrough if needed
            if strikethrough:
                x, y = coords
                # Draw line through the middle of the text
                line_y = y + (text_height // 2)
                draw.line((x, line_y, x + text_width, line_y), fill=rgb_color, width=1)

                # Add actual winner after the strikethrough text
                if actual_winner:
                    # Draw arrow and actual winner
                    actual_text = f" → {actual_winner}"
                    actual_x = x + text_width + 5
                    draw.text((actual_x, y), actual_text, fill=rgb_color, font=font)

    def _create_and_save_figure(self, img, title, output_path=None, show_plot=True):
        """Create matplotlib figure, save and display if requested"""
        # Convert to numpy array for matplotlib
        img_array = np.array(img)

        # Create a matplotlib figure
        dpi = 30
        height, width, _ = img_array.shape
        figsize = (width / dpi, height / dpi)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(img_array)
        ax.axis("off")

        # Set title
        plt.title(title)

        # Save if requested
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
            print(f"Saved bracket to {output_path}")

        # Show if requested
        if show_plot:
            try:
                plt.tight_layout()
                plt.ion()  # Turn on interactive mode
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display plot: {str(e)}")

        return fig

    def _visualize_bracket_common(
        self, slot_data, title, output_path=None, show_plot=True, is_historical=False
    ):
        """Shared visualization method for regular and historical brackets"""
        # Prepare the image and drawing objects
        img, draw, font = self._prepare_bracket_image()

        # Draw bracket data
        self._draw_bracket_data(draw, slot_data, font, is_historical)

        # Create, save and show the figure
        return self._create_and_save_figure(img, title, output_path, show_plot)

    def visualize_bracket(
        self, method="ensemble", output_path=None, show_plot=True, betting_odds=False
    ):
        """
        Visualize the tournament bracket

        Parameters:
        method: Prediction method to use
        output_path: Path to save the image (optional)
        show_plot: Whether to display the plot
        betting_odds: Whether to show betting odds instead of probabilities

        Returns:
        fig: Matplotlib figure object
        """
        # Simulate the bracket
        slot_data = self.simulate_bracket(method=method, betting_odds=betting_odds)

        # Set title
        title = f"{self.current_season} March Madness Bracket Prediction ({method})"
        if betting_odds:
            title += " - Betting Odds"

        # Use shared visualization method
        return self._visualize_bracket_common(slot_data, title, output_path, show_plot)

    def visualize_historical_bracket(
        self, season, method="elo_enhanced", output_path=None, show_plot=True
    ):
        """
        Visualize a historical tournament bracket with predictions vs actual results

        Parameters:
        season: Historical season to visualize
        method: Prediction method to use
        output_path: Path to save the image (optional)
        show_plot: Whether to display the plot

        Returns:
        fig: Matplotlib figure object
        """
        # Simulate the historical bracket
        slot_data, metrics = self.simulate_historical_bracket(season, method=method)

        # Set title with accuracy metrics
        accuracy_pct = metrics["accuracy"] * 100
        title = f"{season} Tournament: Predicted vs Actual ({method})\n"
        title += f"Accuracy: {accuracy_pct:.1f}% ({metrics['correct']}/{metrics['total']} games)"

        # Use shared visualization method
        return self._visualize_bracket_common(
            slot_data, title, output_path, show_plot, is_historical=True
        )

    def get_team_path(self, team_id):
        """
        Get a team's path through the tournament

        Parameters:
        team_id: ID of the team to track

        Returns:
        path: List of dictionaries with round, opponent, and win probability
        """
        if self.bracket_tree is None:
            raise ValueError(
                "Bracket not simulated yet. Call simulate_bracket() first."
            )

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
            # Find the level that contains the current node and get its index
            for i, level in enumerate(self.bracket_tree.levels):
                if current in level:
                    level_index = i
                    break

            round_name = self._get_round_name(level_index)

            # Add to path
            path.append(
                {
                    "round": round_name,
                    "opponent_id": opponent.team_id,
                    "opponent_name": opponent.team_name,
                    "opponent_seed": opponent.seed,
                    "win_probability": current.win_prob,
                }
            )

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
            6: "Play-In",
        }
        return round_names.get(level_index, f"Round {level_index}")
