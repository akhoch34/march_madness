"""
Modern matplotlib NCAA bracket visualization.

Draws a structured bracket directly from `BracketSimulator.get_bracket_layout()`
with dedicated region panels, a separate First Four area, and short connectors.
"""

import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

matplotlib.use("Agg")

REGION_NAMES = {"W": "East", "X": "West", "Y": "South", "Z": "Midwest"}
ROUND_LABELS = {1: "Round of 64", 2: "Round of 32", 3: "Sweet 16", 4: "Elite 8"}
BACKGROUND = "#f6f1e8"
CANVAS = "#fcfaf6"
PANEL = "#efe5d5"
LINE = "#b59d7c"
TEXT = "#17212b"
SUBTEXT = "#60707c"
CARD_EDGE = "#d7c3a4"
CARD_EMPTY = "#ece6db"
CMAP_NAME = "RdYlGn"


def _abbreviate(name: str, max_len: int = 18) -> str:
    if name is None:
        return "TBD"
    if len(name) <= max_len:
        return name
    abbrevs = {
        "State": "St.",
        "University": "U.",
        "North": "N.",
        "South": "S.",
        "Eastern": "E.",
        "Western": "W.",
        "Northern": "N.",
        "Southern": "S.",
        "Central": "Cen.",
        "International": "Intl.",
        "Connecticut": "UConn",
    }
    for word, short in abbrevs.items():
        name = name.replace(word, short)
        if len(name) <= max_len:
            return name
    return name[: max_len - 1] + "…"


def _card_face(prob: Optional[float]):
    if prob is None:
        return CARD_EMPTY
    r, g, b, _ = plt.get_cmap(CMAP_NAME)(prob)
    return (r, g, b, 0.92)


def _card_text_color(face) -> str:
    if isinstance(face, str):
        return TEXT
    r, g, b = face[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return TEXT if luminance > 0.62 else "#ffffff"


def _team_prob(team_info) -> Optional[float]:
    if not team_info:
        return None
    prob = team_info.get("win_prob")
    if prob is None:
        return None
    return float(prob)


def _draw_team_card(ax, x, y, width, height, team_info, align="left"):
    face = _card_face(_team_prob(team_info))
    text_color = _card_text_color(face)
    card = FancyBboxPatch(
        (x, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.35,rounding_size=5",
        linewidth=1.0,
        edgecolor=CARD_EDGE,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(card)

    if not team_info:
        return

    seed = team_info.get("seed")
    seed_label = f"{int(seed):02d}" if seed is not None else "--"
    name = _abbreviate(team_info.get("team_name"), max_len=18)
    prob = team_info.get("win_prob")
    prob_text = "" if prob is None else f"{prob * 100:.1f}%"

    if align == "left":
        name_x = x + 7
        prob_x = x + width - 7
        name_ha = "left"
        prob_ha = "right"
    else:
        name_x = x + width - 7
        prob_x = x + 7
        name_ha = "right"
        prob_ha = "left"

    ax.text(
        name_x,
        y,
        f"{seed_label} {name}",
        ha=name_ha,
        va="center",
        fontsize=7.2,
        color=text_color,
        family="DejaVu Sans",
        zorder=4,
    )
    if prob_text:
        ax.text(
            prob_x,
            y,
            prob_text,
            ha=prob_ha,
            va="center",
            fontsize=6.5,
            color=text_color,
            alpha=0.95,
            family="DejaVu Sans",
            zorder=4,
        )


def _draw_region(ax, games, region_code, side, x0, y0, width, height):
    panel = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.0,rounding_size=14",
        linewidth=0,
        facecolor=PANEL,
        alpha=0.52,
        zorder=0,
    )
    ax.add_patch(panel)

    ax.text(
        x0 + width / 2,
        y0 + 18,
        REGION_NAMES.get(region_code, region_code),
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
        family="DejaVu Sans",
    )

    round_groups = {rnd: [] for rnd in range(1, 5)}
    for game in games:
        round_groups[game["round"]].append(game)

    col_x = np.linspace(x0 + 22, x0 + width - 132, 4)
    for idx, rnd in enumerate(range(1, 5)):
        ax.text(
            col_x[idx] + 44,
            y0 + 36,
            ROUND_LABELS[rnd],
            ha="center",
            va="center",
            fontsize=8,
            color=SUBTEXT,
            family="DejaVu Sans",
        )

    leaf_y = np.linspace(y0 + 70, y0 + height - 24, 16)
    card_w = 108
    card_h = 15

    centers = {
        1: np.array([(leaf_y[i * 2] + leaf_y[i * 2 + 1]) / 2 for i in range(8)]),
    }
    centers[2] = np.array([(centers[1][i * 2] + centers[1][i * 2 + 1]) / 2 for i in range(4)])
    centers[3] = np.array([(centers[2][i * 2] + centers[2][i * 2 + 1]) / 2 for i in range(2)])
    centers[4] = np.array([(centers[3][0] + centers[3][1]) / 2])

    align = "left" if side == "left" else "right"

    for idx, game in enumerate(round_groups[1]):
        y_top = leaf_y[idx * 2]
        y_bot = leaf_y[idx * 2 + 1]
        x = col_x[0]
        _draw_team_card(ax, x, y_top, card_w, card_h, game["top"], align=align)
        _draw_team_card(ax, x, y_bot, card_w, card_h, game["bot"], align=align)
        center_y = centers[1][idx]
        edge_x = x + card_w if side == "left" else x
        target_x = col_x[1] - 10 if side == "left" else col_x[1] + card_w + 10
        join_x = (edge_x + target_x) / 2
        ax.plot([edge_x, join_x], [y_top, y_top], color=LINE, lw=1.3, zorder=1)
        ax.plot([edge_x, join_x], [y_bot, y_bot], color=LINE, lw=1.3, zorder=1)
        ax.plot([join_x, join_x], [y_top, y_bot], color=LINE, lw=1.3, zorder=1)
        ax.plot([join_x, target_x], [center_y, center_y], color=LINE, lw=1.3, zorder=1)

    for rnd in (2, 3, 4):
        for idx, game in enumerate(round_groups[rnd]):
            x = col_x[rnd - 1]
            y_top = centers[rnd - 1][idx * 2]
            y_bot = centers[rnd - 1][idx * 2 + 1]
            _draw_team_card(ax, x, y_top, card_w, card_h, game["top"], align=align)
            _draw_team_card(ax, x, y_bot, card_w, card_h, game["bot"], align=align)
            if rnd == 4:
                continue
            center_y = centers[rnd][idx]
            edge_x = x + card_w if side == "left" else x
            target_x = col_x[rnd] - 10 if side == "left" else col_x[rnd] + card_w + 10
            join_x = (edge_x + target_x) / 2
            ax.plot([edge_x, join_x], [y_top, y_top], color=LINE, lw=1.3, zorder=1)
            ax.plot([edge_x, join_x], [y_bot, y_bot], color=LINE, lw=1.3, zorder=1)
            ax.plot([join_x, join_x], [y_top, y_bot], color=LINE, lw=1.3, zorder=1)
            ax.plot([join_x, target_x], [center_y, center_y], color=LINE, lw=1.3, zorder=1)

    return centers[4][0]


def _draw_play_in(ax, play_in_games, x0, y0, width, height):
    if not play_in_games:
        return

    panel = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.0,rounding_size=14",
        linewidth=0,
        facecolor="#e6ddd0",
        alpha=0.6,
        zorder=0,
    )
    ax.add_patch(panel)
    ax.text(
        x0 + width / 2,
        y0 + 16,
        "First Four",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
        family="DejaVu Sans",
    )

    rows = np.linspace(y0 + 42, y0 + height - 16, len(play_in_games))
    for row_y, game in zip(rows, play_in_games):
        ax.text(
            x0 + 10,
            row_y - 10,
            REGION_NAMES.get(game.get("region"), game.get("region", "")),
            ha="left",
            va="center",
            fontsize=7,
            color=SUBTEXT,
            family="DejaVu Sans",
        )
        _draw_team_card(ax, x0 + 8, row_y - 5, 102, 14, game["top"], align="left")
        _draw_team_card(ax, x0 + 8, row_y + 11, 102, 14, game["bot"], align="left")


def _draw_center(ax, layout, left_top_y, left_bottom_y, right_top_y, right_bottom_y):
    semi_x_left = 404
    semi_x_right = 520
    final_x = 462
    card_w = 112
    card_h = 16

    final_four = layout.get("final_four", [])
    if len(final_four) >= 2:
        top_semi = final_four[0]
        bot_semi = final_four[1]
    else:
        top_semi = bot_semi = None

    semi_y_top = (left_top_y + right_top_y) / 2
    semi_y_bot = (left_bottom_y + right_bottom_y) / 2
    final_y = (semi_y_top + semi_y_bot) / 2

    for source_y, semi_y, x_from, x_card in [
        (left_top_y, semi_y_top, 344, semi_x_left),
        (right_top_y, semi_y_top, 704, semi_x_right + card_w),
        (left_bottom_y, semi_y_bot, 344, semi_x_left),
        (right_bottom_y, semi_y_bot, 704, semi_x_right + card_w),
    ]:
        mid_x = (x_from + x_card) / 2
        ax.plot([x_from, mid_x], [source_y, source_y], color=LINE, lw=1.5, zorder=1)
        ax.plot([mid_x, mid_x], [source_y, semi_y], color=LINE, lw=1.5, zorder=1)
        ax.plot([mid_x, x_card], [semi_y, semi_y], color=LINE, lw=1.5, zorder=1)

    if top_semi:
        _draw_team_card(ax, semi_x_left, left_top_y, card_w, card_h, top_semi["top"], align="left")
        _draw_team_card(ax, semi_x_right, right_top_y, card_w, card_h, top_semi["bot"], align="right")
    if bot_semi:
        _draw_team_card(ax, semi_x_left, left_bottom_y, card_w, card_h, bot_semi["top"], align="left")
        _draw_team_card(ax, semi_x_right, right_bottom_y, card_w, card_h, bot_semi["bot"], align="right")

    ax.text(470, 130, "Final Four", ha="center", va="center", fontsize=10, color=SUBTEXT, family="DejaVu Sans")
    ax.text(470, 352, "Championship", ha="center", va="center", fontsize=10, color=SUBTEXT, family="DejaVu Sans")

    top_mid_x = final_x + card_w / 2
    for semi_y in (semi_y_top, semi_y_bot):
        ax.plot([semi_x_left + card_w, top_mid_x], [semi_y, semi_y], color=LINE, lw=1.6, zorder=1)
        ax.plot([semi_x_right, top_mid_x], [semi_y, semi_y], color=LINE, lw=1.6, zorder=1)
    ax.plot([top_mid_x, top_mid_x], [semi_y_top, semi_y_bot], color=LINE, lw=1.6, zorder=1)

    championship = layout.get("championship")
    if championship:
        _draw_team_card(ax, final_x, semi_y_top, card_w, card_h, championship["top"], align="left")
        _draw_team_card(ax, final_x, semi_y_bot, card_w, card_h, championship["bot"], align="left")
        _draw_team_card(ax, final_x, final_y, card_w, card_h, championship["winner"], align="left")
        ax.text(
            final_x + card_w / 2,
            final_y + 24,
            "Champion",
            ha="center",
            va="center",
            fontsize=8,
            color=SUBTEXT,
            family="DejaVu Sans",
        )


def _draw_layout_bracket(sim, title: str, output_path: Optional[str], method: str, layout=None) -> plt.Figure:
    if layout is None:
        layout = sim.get_bracket_layout(method=method, season=sim.current_season)
    fig, ax = plt.subplots(figsize=(18, 12), dpi=180)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(CANVAS)
    ax.set_xlim(0, 940)
    ax.set_ylim(700, 0)
    ax.axis("off")

    ax.text(
        470,
        24,
        title,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        family="DejaVu Sans",
    )
    ax.text(
        470,
        44,
        "Structured bracket view with round-by-round advancement probabilities.",
        ha="center",
        va="center",
        fontsize=9,
        color=SUBTEXT,
        family="DejaVu Sans",
    )

    _draw_play_in(ax, layout.get("play_in", []), 355, 58, 230, 112)

    left_top_y = _draw_region(ax, layout["W"], "W", "left", 18, 86, 338, 272)
    right_top_y = _draw_region(ax, layout["X"], "X", "right", 584, 86, 338, 272)
    left_bottom_y = _draw_region(ax, layout["Y"], "Y", "left", 18, 394, 338, 272)
    right_bottom_y = _draw_region(ax, layout["Z"], "Z", "right", 584, 394, 338, 272)
    _draw_center(ax, layout, left_top_y, left_bottom_y, right_top_y, right_bottom_y)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved bracket to {output_path}")

    return fig


def _draw_bracket_matplotlib(sim, slot_data, title: str, output_path: Optional[str] = None, method: str = "ensemble") -> plt.Figure:
    return _draw_layout_bracket(sim=sim, title=title, output_path=output_path, method=method)


def visualize_bracket(
    predictor,
    season: int,
    method: str = "elo_enhanced",
    output_path: Optional[str] = None,
    colormap: str = "RdYlGn",
    show_odds: bool = False,
    show_logos: bool = False,
) -> plt.Figure:
    from src.data_classes.bracket.BracketGenerator import BracketSimulator

    sim = BracketSimulator(predictor=predictor)
    sim.use_predictor_data(season=season)
    sim.build_bracket_tree(season=season)
    title = f"{season} NCAA Tournament | {method}"
    if show_odds:
        title += " | betting odds unavailable in modern view"
    return _draw_layout_bracket(sim=sim, title=title, output_path=output_path, method=method)


def visualize_historical_bracket(
    predictor,
    season: int,
    method: str = "elo_enhanced",
    output_path: Optional[str] = None,
):
    from src.data_classes.bracket.BracketGenerator import BracketSimulator

    sim = BracketSimulator(predictor=predictor)
    sim.use_predictor_data(season=season)
    sim.build_bracket_tree(season=season)
    _, metrics = sim.simulate_historical_bracket(season=season, method=method)
    title = (
        f"{season} Tournament | {method} | "
        f"accuracy {metrics['accuracy'] * 100:.1f}% "
        f"({metrics['correct']}/{metrics['total']})"
    )
    layout = sim.get_bracket_layout(method=method, season=season, simulate=False)
    fig = _draw_layout_bracket(
        sim=sim,
        title=title,
        output_path=output_path,
        method=method,
        layout=layout,
    )
    return fig, metrics


def visualize_bracket_html(
    predictor,
    season: int,
    method: str = "elo_enhanced",
    output_path: Optional[str] = None,
) -> str:
    from src.data_classes.bracket.BracketGenerator import BracketSimulator

    sim = BracketSimulator(predictor=predictor)
    sim.use_predictor_data(season=season)
    sim.build_bracket_tree(season=season)
    slot_data = sim.simulate_bracket(method=method, betting_odds=False)

    scale = 0.98
    canvas_w = 940 * scale
    canvas_h = 700 * scale
    box_w = 118
    box_h = 18
    center_x = canvas_w / 2

    all_nodes = [node for level in sim.bracket_tree.levels for node in level]
    slot_lookup = {
        node.value: sim.slot_coordinates.get(len(sim.slot_coordinates) - node.value, (0, 0))
        for node in all_nodes
    }

    def esc(text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def display_label(text: str) -> str:
        if not text:
            return ""
        pieces = text.rsplit(" ", 1)
        if len(pieces) == 2 and pieces[1].endswith("%"):
            text = pieces[0]
        return _abbreviate(text, 20)

    def display_prob(text: str) -> str:
        if not text:
            return ""
        pieces = text.rsplit(" ", 1)
        if len(pieces) == 2 and pieces[1].endswith("%"):
            return pieces[1]
        return ""

    cards_svg = []
    connectors_svg = []

    for node in all_nodes:
        if node.parent is None:
            continue
        x1_raw, y1_raw = slot_lookup.get(node.value, (0, 0))
        x2_raw, y2_raw = slot_lookup.get(node.parent.value, (0, 0))
        x1 = x1_raw * scale
        y1 = y1_raw * scale
        x2 = x2_raw * scale
        y2 = y2_raw * scale
        child_on_left = x1 < center_x
        child_edge = x1 + box_w if child_on_left else x1
        parent_edge = x2 if child_on_left else x2 + box_w
        mid_x = (child_edge + parent_edge) / 2
        connectors_svg.append(
            f'<path d="M {child_edge:.1f} {y1:.1f} L {mid_x:.1f} {y1:.1f} '
            f'L {mid_x:.1f} {y2:.1f} L {parent_edge:.1f} {y2:.1f}" '
            f'class="connector" />'
        )

    for node, entry in zip(all_nodes, slot_data):
        coords, text = entry[0], entry[1]
        if not text:
            continue
        x_raw, y_raw = coords
        x = x_raw * scale
        y = y_raw * scale
        tooltip = esc(text)
        label = esc(display_label(text))
        prob = esc(display_prob(text))
        cards_svg.append(
            f'<g class="team-card" data-tip="{tooltip}">'
            f'<rect x="{x:.1f}" y="{y - box_h / 2:.1f}" width="{box_w}" height="{box_h}" rx="0" ry="0" class="team-box" />'
            f'<text x="{x + 6:.1f}" y="{y + 3:.1f}" class="team-text">{label}</text>'
            f'<text x="{x + box_w - 6:.1f}" y="{y + 3:.1f}" class="prob-text" text-anchor="end">{prob}</text>'
            f"</g>"
        )

    round_labels = [
        ("First Round", 88 * scale, 28 * scale),
        ("Second Round", 196 * scale, 28 * scale),
        ("Sweet 16", 318 * scale, 28 * scale),
        ("Elite Eight", 438 * scale, 28 * scale),
        ("Final Four", 560 * scale, 28 * scale),
        ("Championship", center_x, 28 * scale),
        ("Final Four", canvas_w - 560 * scale, 28 * scale),
        ("Elite Eight", canvas_w - 438 * scale, 28 * scale),
        ("Sweet 16", canvas_w - 318 * scale, 28 * scale),
        ("Second Round", canvas_w - 196 * scale, 28 * scale),
        ("First Round", canvas_w - 88 * scale, 28 * scale),
    ]
    round_svg = "".join(
        f'<text x="{x:.1f}" y="{y:.1f}" class="round-label" text-anchor="middle">{esc(label)}</text>'
        for label, x, y in round_labels
    )

    title = f"{season} NCAA Tournament Bracket"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      background: #ffffff;
      font-family: "Helvetica Neue", Arial, sans-serif;
      color: #111111;
    }}
    .wrap {{
      padding: 16px;
      overflow-x: auto;
    }}
    svg {{
      display: block;
      margin: 0 auto;
      background: #ffffff;
    }}
    .connector {{
      fill: none;
      stroke: #111111;
      stroke-width: 1.3;
      stroke-linecap: square;
      stroke-linejoin: miter;
    }}
    .team-box {{
      fill: #ffffff;
      stroke: #111111;
      stroke-width: 1.2;
    }}
    .team-text {{
      font-size: 8.2px;
      font-weight: 600;
      fill: #111111;
      dominant-baseline: middle;
    }}
    .prob-text {{
      font-size: 7.2px;
      font-weight: 700;
      fill: #444444;
      dominant-baseline: middle;
    }}
    .round-label {{
      font-size: 10px;
      font-weight: 700;
      fill: #111111;
    }}
    .region-label {{
      font-size: 13px;
      font-weight: 700;
      fill: #111111;
    }}
    .title-small {{
      font-size: 15px;
      font-weight: 800;
      letter-spacing: 0.08em;
      fill: #111111;
    }}
    .title-big {{
      font-size: 22px;
      font-weight: 900;
      letter-spacing: 0.03em;
      fill: #111111;
    }}
    .year-text {{
      font-size: 16px;
      font-weight: 800;
      fill: #111111;
    }}
    .champ-box {{
      fill: #ffffff;
      stroke: #111111;
      stroke-width: 1.8;
    }}
    .champ-label {{
      font-size: 12px;
      font-weight: 700;
      fill: #111111;
      text-anchor: middle;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <svg viewBox="0 0 {canvas_w:.0f} {canvas_h:.0f}" width="{canvas_w:.0f}" height="{canvas_h:.0f}" role="img" aria-label="{esc(title)}">
      <line x1="30" y1="14" x2="{canvas_w - 30:.0f}" y2="14" stroke="#111111" stroke-width="2" />
      {round_svg}
      {''.join(connectors_svg)}
      {''.join(cards_svg)}
      <text x="{center_x:.1f}" y="{132 * scale:.1f}" class="title-small" text-anchor="middle">MEN’S</text>
      <text x="{center_x:.1f}" y="{160 * scale:.1f}" class="title-big" text-anchor="middle">NCAA TOURNAMENT</text>
      <text x="{center_x:.1f}" y="{184 * scale:.1f}" class="title-big" text-anchor="middle">BRACKET</text>
      <line x1="{center_x - 78:.1f}" y1="{198 * scale:.1f}" x2="{center_x - 30:.1f}" y2="{198 * scale:.1f}" stroke="#111111" stroke-width="3" />
      <line x1="{center_x + 30:.1f}" y1="{198 * scale:.1f}" x2="{center_x + 78:.1f}" y2="{198 * scale:.1f}" stroke="#111111" stroke-width="3" />
      <text x="{center_x:.1f}" y="{206 * scale:.1f}" class="year-text" text-anchor="middle">{season}</text>
      <text x="{340 * scale:.1f}" y="{318 * scale:.1f}" class="region-label" text-anchor="middle">South</text>
      <text x="{340 * scale:.1f}" y="{574 * scale:.1f}" class="region-label" text-anchor="middle">West</text>
      <text x="{600 * scale:.1f}" y="{318 * scale:.1f}" class="region-label" text-anchor="middle">East</text>
      <text x="{600 * scale:.1f}" y="{574 * scale:.1f}" class="region-label" text-anchor="middle">Midwest</text>
      <rect x="{center_x - 50:.1f}" y="{420 * scale:.1f}" width="100" height="24" class="champ-box" />
      <text x="{center_x:.1f}" y="{462 * scale:.1f}" class="champ-label">CHAMPION</text>
    </svg>
  </div>
</body>
</html>"""

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Saved interactive bracket to {output_path}")

    return html_content
