# Generate Submission

Generate a complete Kaggle submission file for the current season by running all three approach notebooks.

## Before Proceeding — Ask the User

Ask: "Which season are you generating a submission for? (default: 2026)"

Use their answer as `CURRENT_SEASON` throughout, and in all output paths.

## Steps

### 1. ELO-Enhanced (primary model)
Run `notebooks/elo_enhanced.ipynb` with `CURRENT_SEASON = {SEASON}`

Outputs:
- `output/{SEASON}/elo_enhanced/M/submission.csv`
- `output/{SEASON}/elo_enhanced/W/submission.csv`
- `output/{SEASON}/elo_enhanced/submission_combined.csv` (combined)

### 2. XGBoost Ensemble
Run `notebooks/xgb_ensemble.ipynb` with `CURRENT_SEASON = {SEASON}`

Outputs:
- `output/{SEASON}/xgb_ensemble/submission_combined.csv`

### 3. Baseline Massey
Run `notebooks/baseline_massey.ipynb` with `CURRENT_SEASON = {SEASON}`

Outputs:
- `output/{SEASON}/baseline_massey/submission_combined.csv`

### 4. (Optional) Ensemble the outputs

```python
import pandas as pd
elo = pd.read_csv("output/{SEASON}/elo_enhanced/submission_combined.csv")
xgb = pd.read_csv("output/{SEASON}/xgb_ensemble/submission_combined.csv")

merged = elo.merge(xgb, on="ID", suffixes=("_elo","_xgb"))
merged["Pred"] = 0.5 * merged["Pred_elo"] + 0.5 * merged["Pred_xgb"]
merged[["ID","Pred"]].to_csv("output/{SEASON}/submission_final.csv", index=False)
```

## Important Notes

- Data directory: `data/{SEASON}/`
- The submission must cover ALL possible team matchups (not just seeded teams)
- Women's + Men's predictions go in one combined file
- ID format: `{Season}_{lower_TeamID}_{higher_TeamID}`
- Pred = P(lower TeamID team wins)
- Predictions should be clipped to [0.025, 0.975] to avoid extreme log loss

## Current Data Status (as of March 12, 2026)

- Regular season data: available in `data/2026/`
- Tournament seeds: NOT yet announced (Selection Sunday ~March 15)
- Can submit immediately without seeds — use all active D1 teams
- Once seeds released, regenerate bracket visualization
