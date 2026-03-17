# Generate Submission

Generate a complete Kaggle submission file for the current season.

## Before Proceeding — Ask the User

Ask: "Which season are you generating a submission for? (default: 2026)"

Use their answer as `SEASON` throughout.

## Steps

### 1. Run the submission generator

```bash
poetry run python utils/generate_submission.py --season {SEASON}
```

This will:
- Read `output/eval_results.csv` to find top-performing methods per gender
- Generate per-method prediction CSVs to `output/{SEASON}/submissions/{method}_{gender}.csv`
- Combine M+W into `output/{SEASON}/submission_{SEASON}_top1.csv`
- Generate bracket PNGs to `output/{SEASON}/brackets/{method}/{gender}/bracket.png`

### 2. (Optional) Use a different strategy

```bash
# Average top 3 methods, rank on recent years only:
poetry run python utils/generate_submission.py \
    --season {SEASON} \
    --strategy average_top3 \
    --years-for-ranking 2024 2025 \
    --top-n 3
```

### 3. (Optional) Run eval first if eval_results.csv is stale

```bash
poetry run python utils/eval_framework.py
```

## Important Notes

- Data directory defaults to `data/{SEASON}/`
- The submission covers ALL possible team matchups (not just seeded teams)
- Women's + Men's predictions go in one combined file
- ID format: `{Season}_{lower_TeamID}_{higher_TeamID}`
- Pred = P(lower TeamID team wins)
- Predictions are clipped to [0.025, 0.975] to avoid extreme log loss
