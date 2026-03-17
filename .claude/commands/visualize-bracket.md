# Visualize Tournament Bracket

Generate a color-coded bracket visualization for a given season and method.

## Before Proceeding — Ask the User

Ask the user the following questions before running any code:
1. **Season** — which year? (2022–2025 for historical with real results; 2026 if seeds have been released ~March 15)
2. **Gender** — Men's (M), Women's (W), or both?
3. **Method** — elo_enhanced (recommended), elo, or ensemble?

Use their answers as `SEASON`, `GENDER`, and `METHOD` throughout, and in all output paths.

## Requirements

- Tournament seeds must be available in `data/{SEASON}/M(W)NCAATourneySeeds.csv`
- Predictor must be initialized (run `elo_enhanced.ipynb` or `xgb_ensemble.ipynb` first)

## Quick Usage

```python
from src.visualization.bracket_viz import visualize_bracket, visualize_bracket_html

# PNG bracket (color-coded by win probability)
fig = visualize_bracket(
    predictor=predictor_m,        # MarchMadnessPredictor instance
    season=2026,
    method="elo_enhanced",        # or "elo", "ensemble"
    output_path="output/{SEASON}/{METHOD}/M/bracket.png",
    colormap="RdYlGn",            # Green=favorite, Red=underdog
    show_odds=False,              # True = show point spreads instead
)

# Interactive HTML bracket (requires plotly)
html = visualize_bracket_html(
    predictor=predictor_m,
    season=2026,
    method="elo_enhanced",
    output_path="output/{SEASON}/{METHOD}/M/bracket.html",
)
```

## Using BracketGenerator directly

```python
from src.data_classes.bracket.BracketGenerator import BracketSimulator

sim = BracketSimulator(predictor=predictor_m)
sim.use_predictor_data(season=2026)
sim.build_bracket_tree()

# Static PIL-based visualization
fig = sim.visualize_bracket(
    method="elo_enhanced",
    output_path="output/{SEASON}/{METHOD}/M/bracket.png",
    betting_odds=False,
)

# Historical bracket (shows predicted vs actual)
fig = sim.visualize_historical_bracket(
    season=2025,
    method="elo_enhanced",
    output_path="output/{SEASON}/{METHOD}/M/historical_bracket.png",
)
```

## Color Interpretation

- **Dark green**: Heavy favorite (>85% win probability)
- **Light green**: Moderate favorite (65-85%)
- **Yellow**: Toss-up (near 50/50)
- **Orange/Red**: Underdog (<50% win probability)

## Parameters

- `SEASON`: Year of the tournament (e.g., 2026)
- `METHOD`: `elo_enhanced` (recommended), `elo`, or `ensemble`
- `OUTPUT_PATH`: File path for saving (PNG or HTML)
