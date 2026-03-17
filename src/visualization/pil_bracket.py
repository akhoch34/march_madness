"""
PIL-based bracket renderer using empty.jpg template.

Draws team names directly onto an empty bracket template JPEG at hardcoded
slot coordinates from seed_slots.py.  No matplotlib dependency.
"""

import os
import re

from PIL import Image, ImageDraw
from binarytree import Node

from src.data_classes.bracket.seed_slots import SLOTS


_TEMPLATE_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "data_classes", "bracket", "empty.jpg",
    )
)

# Resampling filter — compatible with both Pillow < 9 and >= 9
try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE = Image.LANCZOS


class _ExtNode(Node):
    """Binary tree node extended with a parent reference.

    Required so we can propagate the winner upward after each game without
    storing a separate parent lookup dict.
    """

    def __init__(self, value, left=None, right=None):
        super().__init__(value, left=left, right=right)
        self.parent = None
        self.team_id = None
        self.seed_str = None   # raw seed string, e.g. "W01", "X16b"
        self.team_name = None
        self.win_prob = None
        self.correct = None    # True / False / None (None = no game result)

    def __setattr__(self, name, value):
        # Auto-assign parent when a child is attached
        if name in ("left", "right") and isinstance(value, _ExtNode):
            value.parent = self
        super().__setattr__(name, value)


def _build_tree(slots_df, season):
    """Build an _ExtNode binary tree from tourney_slots data."""
    s = slots_df[slots_df["Season"] == season]
    if len(s) == 0:
        raise ValueError(f"No tourney_slots data for season {season}")

    seed_slot_map = {0: "R6CH"}  # node.value → slot-name string
    root = _ExtNode(0)
    counter = 1
    queue = [root]

    while queue:
        next_q = []
        for node in queue:
            rows = s[s["Slot"] == seed_slot_map[node.value]]
            if len(rows) > 0:
                row = rows.iloc[0]
                node.left = _ExtNode(counter)
                node.right = _ExtNode(counter + 1)
                seed_slot_map[counter] = row["StrongSeed"]
                seed_slot_map[counter + 1] = row["WeakSeed"]
                next_q.extend([node.left, node.right])
                counter += 2
        queue = next_q

    return root, seed_slot_map


def render_pil_bracket(
    predictor,
    season: int,
    output_path: str,
    gender: str = "M",
    logo_dir: str = None,
    show_win_probs: bool = True,
    historical: bool = False,
) -> None:
    """Render a bracket PNG onto the empty.jpg template using PIL.

    Args:
        predictor: SimplePredictor (or any duck-type equivalent) with
                   data_manager.data["teams" / "tourney_seeds" / "tourney_slots"]
                   and predictions_df with columns ID + Pred.
        season:    Tournament year.
        output_path: Destination .png path.
        gender:    "M" or "W" (used only for caller convenience; data already
                   scoped by the predictor).
        logo_dir:  Directory containing {team_id}.png logos.  Pass None to skip.
        show_win_probs: Include win-probability % in team labels.
        historical: If True, advance the *actual* winner and colour slots
                    green (correct) / red (wrong).  Requires tourney_results.
    """
    dm = predictor.data_manager
    teams_df = dm.data["teams"]
    seeds_df = dm.data["tourney_seeds"]
    slots_df = dm.data["tourney_slots"]
    results_df = dm.data.get("tourney_results")

    # ------------------------------------------------------------------ tree
    root, seed_slot_map = _build_tree(slots_df, season)

    # ---------------------------------------------------------------- lookups
    season_seeds = seeds_df[seeds_df["Season"] == season]
    seed_to_team = dict(zip(season_seeds["Seed"], season_seeds["TeamID"]))
    team_name_map = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

    # Normalise prediction columns to lowercase
    pred_df = predictor.predictions_df.copy()
    pred_df.columns = [c.lower() for c in pred_df.columns]
    pred_lookup = dict(zip(pred_df["id"], pred_df["pred"]))

    # Actual result lookup: (min_id, max_id) → winner_id
    actual_winners = {}
    if historical and results_df is not None:
        for _, row in results_df[results_df["Season"] == season].iterrows():
            w, l = int(row["WTeamID"]), int(row["LTeamID"])
            actual_winners[(min(w, l), max(w, l))] = w

    # ------------------------------------------------------------- simulation
    for level in reversed(root.levels):
        for i in range(0, len(level) - 1, 2):
            left, right = level[i], level[i + 1]

            # Populate leaf nodes with team data
            for node in (left, right):
                if node.left is None and node.team_id is None:
                    s = seed_slot_map.get(node.value, "")
                    tid = seed_to_team.get(s)
                    node.seed_str = s
                    node.team_id = tid
                    node.team_name = team_name_map.get(tid) if tid else None

            if not left.team_id or not right.team_id:
                continue

            # Prediction: pred_lookup stores P(lower_team_id wins)
            t1 = min(left.team_id, right.team_id)
            t2 = max(left.team_id, right.team_id)
            game_id = f"{season}_{t1}_{t2}"
            raw = pred_lookup.get(game_id, 0.5)
            p_left = raw if left.team_id == t1 else 1.0 - raw

            left.win_prob = p_left
            right.win_prob = 1.0 - p_left

            predicted_winner = left if p_left > 0.5 else right

            if left.parent is None:
                continue
            parent = left.parent

            if historical:
                actual_id = actual_winners.get((t1, t2))
                if actual_id is not None:
                    actual_winner = left if actual_id == left.team_id else right
                    parent.team_id = actual_winner.team_id
                    parent.seed_str = actual_winner.seed_str
                    parent.team_name = actual_winner.team_name
                    parent.correct = (predicted_winner.team_id == actual_winner.team_id)
                else:
                    # Game not yet played — advance predicted winner, no colour
                    parent.team_id = predicted_winner.team_id
                    parent.seed_str = predicted_winner.seed_str
                    parent.team_name = predicted_winner.team_name
                    parent.correct = None
            else:
                parent.team_id = predicted_winner.team_id
                parent.seed_str = predicted_winner.seed_str
                parent.team_name = predicted_winner.team_name

    # ----------------------------------------------------------- slot entries
    n_slots = len(SLOTS)
    slotdata = []  # list of (xy, label, team_id, correct)

    for node in (n for lvl in root.levels for n in lvl):
        slot_num = n_slots - node.value
        xy = SLOTS.get(slot_num)
        if xy is None or not node.team_name:
            continue

        # Extract numeric seed (strip region prefix e.g. "W" from "W01")
        raw_seed = node.seed_str or ""
        tail = raw_seed[1:] if len(raw_seed) > 1 else raw_seed
        m = re.search(r"\d+", tail)
        seed_num = m.group().zfill(2) if m else "??"

        # Truncate name to prevent overflow into adjacent column
        name = node.team_name[:12]

        prob = ""
        if show_win_probs and node.win_prob is not None and node.parent is not None:
            prob = f" {node.win_prob * 100:.1f}%"

        label = f"{seed_num} {name}{prob}"
        slotdata.append((xy, label, node.team_id, node.correct))

    # ---------------------------------------------------------------- drawing
    try:
        img = Image.open(_TEMPLATE_PATH).convert("RGB")
    except Exception:
        img = Image.new("RGB", (940, 700), color="white")

    draw = ImageDraw.Draw(img)

    for xy, label, team_id, correct in slotdata:
        x, y = xy
        text_x = x

        # Optionally paste team logo to the left of the text
        if logo_dir and team_id:
            logo_path = os.path.join(logo_dir, f"{team_id}.png")
            if os.path.exists(logo_path):
                try:
                    logo = Image.open(logo_path).convert("RGBA").resize(
                        (14, 14), _RESAMPLE
                    )
                    img.paste(logo, (x, y - 5), logo)
                    text_x = x + 17
                except Exception:
                    pass

        # Colour: green = correct, red = wrong, black = unknown/future
        if historical and correct is True:
            color = (0, 128, 0)
        elif historical and correct is False:
            color = (180, 0, 0)
        else:
            color = (0, 0, 0)

        draw.text((text_x, y), label, fill=color)

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    img.save(output_path)
