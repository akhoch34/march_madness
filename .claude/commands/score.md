# Score Historical Submissions

Score archived submission CSVs against actual tournament results and report Brier scores.

## Before Proceeding — Ask the User

Ask: "Score which years? (default: all available — 2022, 2023, 2024, 2025; or enter specific years e.g. '2024 2025')"

Pass the selected years as the `seasons` list to `score_all_submissions()`.

## Steps

1. Open `notebooks/score_historical.ipynb`
2. Run all cells from top to bottom
3. The notebook will:
   - Load all CSVs from `archive/submissions/{year}/{model}.csv`
   - Load actual results from `data/2026/MNCAATourneyCompactResults.csv` and `WNCAATourneyCompactResults.csv`
   - Compute Brier scores for men's and women's separately
   - Print a summary table (model × year × Brier score)
   - Generate `output/historical_brier_scores.png` bar chart
   - Generate `output/calibration_diagram.png` reliability diagram
   - Show upset detection accuracy per model

## Key scoring logic (src/evaluation/scorer.py)

```python
from src.evaluation.scorer import score_all_submissions, generate_scoring_report

results = score_all_submissions("archive/submissions", "data/2026")
print(generate_scoring_report(results))
```

## Interpreting Results

- **Brier score < 0.20**: Good model
- **Brier score 0.20–0.23**: Average
- **Brier score > 0.23**: Poor (close to random baseline of 0.25)
- Random baseline = 0.25 (always predict 50/50)
- Seed-only baseline ≈ 0.21–0.22

## Scorable years: 2022, 2023, 2024, 2025
