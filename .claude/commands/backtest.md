# Backtest Model Performance

Run backtesting across historical seasons to evaluate model accuracy and Brier score.

## Before Proceeding — Ask the User

Ask: "Backtest which season range? (e.g. 2022–2025; default: all available — 2019–2025 excluding 2020)"

Use the selected range to populate `backtest_seasons` in the code below.

## Using the ELO-Enhanced Predictor

In `notebooks/elo_enhanced.ipynb`, after `initialize_models()`:

```python
# Backtest on seasons 2019-2025
backtest_results = {}
for test_season in range(2019, CURRENT_SEASON):
    results = predictor_m.compare_methods(
        season=test_season,
        methods=["elo", "elo_enhanced"],
    )
    backtest_results[test_season] = results
    print(f"{test_season}: {results}")
```

## Using the Scorer Module

For backtesting archived submissions:

```python
from src.evaluation.scorer import load_tournament_results, score_submission
import pandas as pd

sub = pd.read_csv("archive/submissions/2024/xgb_ensemble.csv")
results = load_tournament_results("data/2026", 2024, gender="M")
stats = score_submission(sub, results, 2024)

print(f"Brier score: {stats['brier_score']:.4f}")
print(f"Games scored: {stats['n_games']}")
print("By round:", stats['per_round'])
```

## Metrics Computed

- **Brier score**: mean((pred - actual)^2) — primary Kaggle metric
- **Accuracy**: fraction of games where predicted winner (prob > 0.5) won
- **Log loss**: cross-entropy loss (additional diagnostic)
- **Per-round breakdown**: identifies which rounds each model struggles with

## Seasons with actual results available

- 2022, 2023, 2024, 2025: results in `data/2026/M(W)NCAATourneyCompactResults.csv`
- 2020: SKIPPED (COVID — no tournament)
- 2021: partial data available

## ELO Parameter Tuning

Set `PERFORM_TUNING = True` in `elo_enhanced.ipynb` to run grid search over:
- k_factor: [20, 32, 40]
- recency_factor: [1.0, 1.5, 2.0]
- carry_over_factor: [0.4, 0.6, 0.8]
