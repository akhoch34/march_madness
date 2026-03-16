# March Madness ML Predictions

NCAA tournament win-probability predictions for the Kaggle March Machine Learning Mania competition (men's and women's).

**Scoring metric:** Brier score = mean((pred − actual)²). Random = 0.25, lower is better.

## Setup

```bash
poetry install && poetry shell
```

## Prediction Methods

Ten methods are implemented in `src/approaches/notebook_models.py`. Two additional ELO-based methods exist via the `Predictor` class but are excluded from defaults due to poor historical Brier scores (~0.30).

| Method | How It Works | Key Features | Blend |
|--------|-------------|--------------|-------|
| `seed_spline_only` | `UnivariateSpline` fitted on (SeedDiff, result) pairs from all training tourney games. Zero ML. | Seed difference only | 100% spline |
| `baseline_massey` | XGBoost on win%, seed diff, Massey rank. Spline-dominant blend. | Win% (overall/home/away/neutral), SeedDiff, normalized Massey rank | 30% XGB + 70% spline |
| `massey_direct` | Logistic regression trained on per-system Massey rank differences. Women's fallback: seed spline only. | SeedDiff + per-system rank diff (POM, SAG, DOK, MOR, MAS, RPI) | 25% LR + 75% spline |
| `massey_blend` | No training — directly averages normalized ranks across 6 Massey systems, blends with spline. Women's fallback: seed spline only. | Averaged Massey rank diffs across POM/SAG/DOK/MOR/MAS/RPI | 40% Massey avg + 60% spline |
| `xgb_ensemble` | 5-model ensemble (HistGBM, RF, LR, SVC, XGBoost) averaged. Uses advanced shooting and efficiency stats from detailed game logs. | Win%, point margins, FG%/FT%/3P%, assist rate, turnover rate, rebound rate, steals | 30% ensemble + 70% spline |
| `xgb_ensemble_v2` | Extended `xgb_ensemble` with conference tourney signals and individual per-system Massey columns. | All `xgb_ensemble` features + conference tourney wins/pct/champion flag + per-Massey-system rank diffs | 30% ensemble + 70% spline (20% for W) |
| `recency_xgb` | Same features as `baseline_massey` but training rows are weighted by `0.75^(current_season − row_season)`. | Same as `baseline_massey` + sample weights | 30% weighted XGB + 70% spline |
| `poisson_margin` | Computes expected scoring margin `(OffEff_t1 − DefEff_t2) − (OffEff_t2 − DefEff_t1)`, fits logistic regression on historical (margin, result) pairs. Only method derived purely from on-court efficiency. | Offensive/defensive efficiency (pts per game), expected point margin | 35% margin LR + 65% spline |
| `seed_matchup_calibration` | Empirical win-rate lookup keyed by `(seed_lo, seed_hi)` pair (e.g., 5v12 ≈ 35%). Blends calibrated rate with `xgb_ensemble_v2`. Falls back to spline if <10 historical games for that pair. | Historical win rate per seed-pair + `xgb_ensemble_v2` predictions | 35% historical rate + 65% xgb_v2 |
| `meta_ensemble` | Stacked meta-learner: OOF loop trains `xgb_ensemble_v2` and `massey_direct` on prior seasons, fits LogisticRegression on OOF preds. Slow (~5+ min). Women's fallback: `xgb_ensemble_v2` only. | `xgb_ensemble_v2` and `massey_direct` out-of-fold predictions as meta-features | 50% meta-LR + 50% spline |

**ELO-backed methods** (via `Predictor` class, included in `generate_historical_artifacts.py` but not `eval_framework.py` defaults):
- `elo` — pure ELO rating prediction
- `elo_enhanced` — ELO + XGBoost residual correction

## Historical Performance

Results from `output/eval_results.csv` (Brier score, lower is better).

**Men's (2022–2025):**

| Method | 2022 | 2023 | 2024 | 2025 | Avg |
|--------|------|------|------|------|-----|
| `xgb_ensemble` | 0.2173 | 0.2116 | 0.1878 | 0.1566 | **0.1933** |
| `seed_matchup_calibration` | 0.2124 | 0.2157 | 0.1902 | 0.1569 | 0.1938 |
| `recency_xgb` | 0.2173 | 0.2186 | 0.1840 | 0.1566 | 0.1941 |
| `xgb_ensemble_v2` | 0.2173 | 0.2152 | 0.1886 | 0.1561 | 0.1943 |
| `baseline_massey` | 0.2145 | 0.2175 | 0.1833 | 0.1652 | 0.1951 |
| `massey_direct` | 0.2142 | 0.2114 | 0.1939 | 0.1645 | 0.1960 |
| `seed_spline_only` | 0.2142 | 0.2112 | 0.1940 | 0.1648 | 0.1961 |
| `massey_blend` | 0.2142 | 0.2147 | 0.2008 | 0.1811 | 0.2027 |
| `poisson_margin` | 0.2145 | 0.2148 | 0.2027 | 0.1836 | 0.2039 |

**Women's (2023–2025, no 2022 data):**

| Method | 2023 | 2024 | 2025 | Avg |
|--------|------|------|------|-----|
| `recency_xgb` | 0.1728 | 0.1186 | 0.1222 | **0.1379** |
| `xgb_ensemble` | 0.1725 | 0.1233 | 0.1185 | 0.1381 |
| `xgb_ensemble_v2` | 0.1729 | 0.1229 | 0.1217 | 0.1391 |
| `seed_matchup_calibration` | 0.1761 | 0.1245 | 0.1203 | 0.1403 |

## CLI Tools

### Evaluate all methods (leaderboard)

```bash
# Full evaluation across all methods × years × genders:
poetry run python utils/eval_framework.py

# Specific subset:
poetry run python utils/eval_framework.py \
    --methods xgb_ensemble xgb_ensemble_v2 seed_matchup_calibration \
    --years 2024 2025 --genders M

# View existing results without regenerating:
poetry run python utils/eval_framework.py --report-only

# Include slow methods (meta_ensemble, elo, elo_enhanced):
poetry run python utils/eval_framework.py --include-slow
```

All flags:
```
--methods METHOD [...]   Methods to run (default: 9 fast methods)
--years YEAR [...]       Tournament seasons (default: 2022 2023 2024 2025)
--genders M W            Genders to include (default: both)
--base-data-dir PATH     Data root (default: data/)
--output-dir PATH        Output root (default: output/)
--no-skip-existing       Re-generate even if CSV exists
--include-slow           Add meta_ensemble, elo, elo_enhanced
--report-only            Load output/eval_results.csv and print, no generation
--top-n N                Leaderboard rows to show (default: 15)
```

Output: `output/eval_results.csv`

### Generate 2026 submission

```bash
# Generate submission using top methods from eval_results.csv:
poetry run python utils/generate_2026_submission.py

# Custom strategy:
poetry run python utils/generate_2026_submission.py \
    --strategy average_top3 \
    --years-for-ranking 2024 2025 \
    --top-n 3
```

All flags:
```
--eval-results PATH          Path to eval_results.csv (default: output/eval_results.csv)
--top-n N                    Number of top methods to use (default: 3)
--strategy top1|average_top3 Combination strategy (default: top1)
--years-for-ranking Y [...]  Which years to rank on (default: all in eval_results)
--data-dir PATH              Data directory for 2026 (default: data/2026)
--output-dir PATH            Output root (default: output/)
--no-skip-existing           Regenerate even if CSV exists
--season N                   Target season (default: 2026)
```

Output: `output/2026/submission_2026_{strategy}.csv`

### Generate historical artifacts

```bash
# All years, both genders, default methods (elo, elo_enhanced, baseline_massey, xgb_ensemble):
poetry run python utils/generate_historical_artifacts.py

# Specific subset:
poetry run python utils/generate_historical_artifacts.py \
    --years 2024 2025 --genders M --skip-brackets
```

All flags:
```
--years YEAR [...]       Seasons to process (default: 2022 2023 2024 2025)
--genders M W            Genders (default: both)
--methods METHOD [...]   Methods (default: elo, elo_enhanced, baseline_massey, xgb_ensemble)
                         All available: elo, elo_enhanced, ensemble, baseline_massey,
                         xgb_ensemble, xgb_ensemble_v2, massey_direct, seed_spline_only,
                         recency_xgb, massey_blend, poisson_margin, meta_ensemble,
                         seed_matchup_calibration
--base-data-dir PATH     Data root (default: data/)
--lookback-years N       Training window in years (default: 8)
--no-skip-existing       Regenerate existing artifacts
--skip-brackets          Skip bracket PNG generation (faster)
--artifact-suffix SUFFIX Optional suffix for output file naming
```

Output: `output/{year}/submissions/`, `output/{year}/brackets/`, `output/scoring_results.csv`

### Other utilities

```bash
# Download ESPN team logos for bracket visualization:
poetry run python utils/download_logos.py
# → data/logos/{M|W}/{team_id}.png

# Compare historical model performance across archived submissions:
poetry run python utils/compare_historical_models.py

# Split combined M+W submission into separate files:
poetry run python utils/split_submission_file.py <submission.csv>
```

## Notebooks

Notebooks in `notebooks/` are useful for exploration and visualization. The primary workflow is CLI tools above.

| Notebook | Purpose |
|----------|---------|
| `elo_enhanced.ipynb` | ELO ratings + XGBoost residual correction |
| `xgb_ensemble.ipynb` | 5-model ensemble (HistGBM / RF / LR / SVC / XGBoost) |
| `baseline_massey.ipynb` | Fast baseline: win% + seed diff + Massey ranks |
| `score_historical.ipynb` | Score all archived submissions with Brier scores |
| `generate_bracket.ipynb` | Generate color-coded bracket PNG/HTML for any season |

Update `CURRENT_SEASON` at the top of each notebook each year. For backtesting, set `DATA_DIR = "../data/2026"` (cumulative dataset with all results through 2025).

## Output Directory Structure

```
output/
├── eval_results.csv          # Method × year × gender Brier scores
├── scoring_results.csv       # Legacy aggregate scoring table
└── {year}/
    ├── submissions/          # {method}_{gender}.csv
    ├── brackets/             # {method}/{gender}/bracket.{png,html}
    └── features/             # {gender}/feature_dataset.csv
```

## Running Tests

```bash
poetry run pytest -m "not slow"   # Fast suite (< 30s) — data, ELO, Brier scorer
poetry run pytest -m slow         # Slow suite — bracket simulation, end-to-end
poetry run pytest                 # All tests
```

Tests use `data/2026` as the data source and are automatically skipped if it's absent.

## Data

Raw data is gitignored. Download from Kaggle and place under `data/<year>/`.

Important: `MMasseyOrdinals.csv` for `data/2024/`, `data/2025/`, and `data/2026/` exceeds GitHub's file-size limit and is intentionally not tracked. After cloning or moving to a new machine, download or copy those files into the matching directories manually before running evaluations, tests, or submission generation.

- `data/2026/` — cumulative dataset through 2025 (men's + women's). Use for all historical analysis and backtesting.
- Year-specific directories (`data/2025/`, etc.) may contain only men's data.
- 2022 data: flat under `data/2022/MDataFiles_Stage1/` and `MDataFiles_Stage2/`
- 2023 data: under `data/2023/MDataFiles/` and `WDataFiles/`
- 2024+ data: flat in `data/<year>/`
- Season 2020 is always skipped (no tournament — COVID).

## Submission Format

```
ID,Pred
2026_1101_1102,0.6234
```

- `ID` = `{Season}_{lower_TeamID}_{higher_TeamID}`
- `Pred` = P(lower-ID team wins)
- Must include **all** possible matchups for both M and W (~130K rows combined)
- Clip predictions to `[0.025, 0.975]`

## Slash Commands

```
/score                 Score archived submissions (asks: which years)
/generate-submission   Generate submission from all approaches (asks: season)
/backtest              Run backtesting (asks: season range)
/visualize-bracket     Generate bracket PNG or HTML (asks: season, gender, method)
```

## See Also

`CLAUDE.md` — full technical reference (directory structure, data gotchas, API conventions, key classes).
