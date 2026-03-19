# March Madness ML Predictions

NCAA tournament win-probability predictions for the Kaggle March Machine Learning Mania competition (men's and women's).

**Scoring metric:** Brier score = mean((pred − actual)²). Random = 0.25, lower is better.

## Setup

```bash
poetry install && poetry shell
```

## Prediction Methods

Twelve methods are implemented in `src/approaches/notebook_models.py`. Two additional ELO-based methods exist via the `Predictor` class but are excluded from defaults due to poor historical Brier scores (~0.30).

| Method | How It Works | Key Features | Blend |
|--------|-------------|--------------|-------|
| `modeh7` | **2025 Kaggle 1st-place solution.** LOSO XGBoost regression on point differential, calibrated with a k=5 spline. Trains on combined M+W data with a `men_women` flag. Symmetric dataset doubling with overtime adjustment. Separate absolute T1/T2 features (not diffs). ELO base=1000 K=100 width=400. Ridge GLM quality on tournament-adjacent teams. | Seed, box-score season avgs (14 stats × T1 + T2 + opponent), ELO, GLM quality, men_women | 100% LOSO ensemble → spline calibration |
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
| `upset_aware_ensemble` | Extends `xgb_ensemble_v2` with four upset-specific features derived from upset pattern analysis. Reduces spline dominance for men's to let upset signals matter. | All `xgb_ensemble_v2` features + HistMatchupRate_Diff, DefFirstRatio, RecentMomentum, MasseyVsSeedGap diffs | **70% ensemble + 30% spline** (M); 30/70 (W) |

**ELO-backed methods** (via `Predictor` class, included in `generate_historical_artifacts.py` but not `eval_framework.py` defaults):
- `elo` — pure ELO rating prediction
- `elo_enhanced` — ELO + XGBoost residual correction

## Historical Performance

Results from `output/eval_results.csv` (Brier score, lower is better). Refresh with `poetry run python utils/eval_framework.py --report-only`.

**Men's (2022–2025):**

| Method | 2022 | 2023 | 2024 | 2025 | Avg |
|--------|------|------|------|------|-----|
| `modeh7` | 0.2364 | 0.2032 | 0.1878 | **0.1397** | **0.1918** |
| `xgb_ensemble` | 0.2173 | 0.2116 | 0.1878 | 0.1566 | 0.1933 |
| `seed_matchup_calibration` | 0.2124 | 0.2157 | 0.1902 | 0.1569 | 0.1938 |
| `recency_xgb` | 0.2173 | 0.2186 | 0.1840 | 0.1566 | 0.1941 |
| `xgb_ensemble_v2` | 0.2173 | 0.2152 | 0.1886 | 0.1561 | 0.1943 |
| `baseline_massey` | 0.2145 | 0.2175 | 0.1833 | 0.1652 | 0.1951 |
| `upset_aware_ensemble` | 0.2234 | 0.2238 | 0.1866 | 0.1495 | 0.1958 |
| `massey_direct` | 0.2142 | 0.2114 | 0.1939 | 0.1645 | 0.1960 |
| `seed_spline_only` | 0.2142 | 0.2112 | 0.1940 | 0.1648 | 0.1961 |
| `massey_blend` | 0.2142 | 0.2147 | 0.2008 | 0.1811 | 0.2027 |
| `poisson_margin` | 0.2145 | 0.2148 | 0.2027 | 0.1836 | 0.2039 |

**Women's (2023–2025, no 2022 data):**

| Method | 2023 | 2024 | 2025 | Avg |
|--------|------|------|------|-----|
| `modeh7` | 0.1654 | 0.1252 | **0.1075** | **0.1327** |
| `recency_xgb` | 0.1728 | 0.1186 | 0.1222 | 0.1379 |
| `xgb_ensemble` | 0.1725 | 0.1233 | 0.1185 | 0.1381 |
| `upset_aware_ensemble` | 0.1766 | 0.1228 | 0.1172 | 0.1389 |
| `xgb_ensemble_v2` | 0.1729 | 0.1229 | 0.1217 | 0.1391 |
| `seed_matchup_calibration` | 0.1761 | 0.1245 | 0.1203 | 0.1403 |
| `massey_blend` | 0.1752 | 0.1233 | 0.1271 | 0.1419 |
| `massey_direct` | 0.1752 | 0.1233 | 0.1271 | 0.1419 |
| `seed_spline_only` | 0.1752 | 0.1233 | 0.1271 | 0.1419 |
| `baseline_massey` | 0.1766 | 0.1180 | 0.1314 | 0.1420 |
| `poisson_margin` | 0.1788 | 0.1482 | 0.1493 | 0.1588 |

> **Notes:**
> - `modeh7` leads both leaderboards. Its 2022 men's score (0.2364) is weaker because women's data isn't available for `data/2022/`, halving its training set for that year.
> - `upset_aware_ensemble` ranks 7th overall (men's) and 3rd overall (women's) after the rewrite: `ThreePtReliance` (r=0.06) was replaced by `HistMatchupRate_Diff` and the men's blend was raised to 70% ML / 30% spline. It is top-4 in both 2024 and 2025 (men's), suggesting its upset features are gaining relevance in recent tournaments.

---

## Upset Analysis

Analysis of men's tournament upsets from 2010–2025 (948 games, 274 upsets). Run with:

```bash
poetry run python utils/upset_analysis.py --data-dir data/2026
```

### Upset rates by round

| Round | Games | Upset Rate |
|-------|-------|------------|
| Round of 64 | 468 | 28.0% |
| Round of 32 | 247 | 27.1% |
| Sweet 16 | 144 | **37.5%** |
| Elite 8 | 36 | **41.7%** |
| Final Four | 30 | 13.3% |
| Championship | 15 | 6.7% |

Models are most penalized in the Sweet 16 and Elite 8 — the rounds with the highest variance. By the Final Four the remaining teams are closely matched at the top, so upsets are rarer.

### Most upset-prone seed matchups

| Matchup | Games | Upset Rate |
|---------|-------|------------|
| 6 vs 11 | 60 | **51.7%** — effectively a coin flip |
| 8 vs 9 | 60 | **48.3%** — nearly even |
| 3 vs 11 | 28 | 42.9% |
| 7 vs 10 | 59 | 39.0% |
| 5 vs 12 | 60 | 38.3% |
| 4 vs 5 | 31 | 35.5% |
| 1 vs 2 | 21 | 38.1% — seeds converge at top |
| 3 vs 14 | 60 | 13.3% |
| 2 vs 15 | 60 | 11.7% |
| 1 vs 16 | 60 | **3.3%** — only 2 upsets in 60 games |

All current models underpredict upsets in 6v11 and 8v9 matchups because the seed-spline assigns ~22% to the higher-seeded team in those games, while the empirical rate is ~50%.

### What features predict upsets

From point-biserial correlation analysis (winner stats minus loser stats):

| Feature Diff | Correlation | Interpretation |
|---|---|---|
| `MasseyImpliedSeed` | +0.57 | Underseeded teams (Massey says they're better than their seed) win more often |
| `NetEff` | −0.47 | Lower net efficiency in the winner = upset signal |
| `DefFirstRatio` | +0.47 | Defense-first teams punch above their seed |
| `FullSeasonWinPct` | −0.44 | Win% gap: lower seed wins despite worse record |
| `RecentMomentum` | +0.24 | Late-season form matters |
| `ThreePtReliance` | +0.06 | Slight signal; high-variance style (removed from model — too noisy) |

> **Key insight:** The strongest actionable signal is teams whose Massey rankings significantly outperform their official seed. A team seeded 10th but ranked 6th by Massey composite is systematically underseeded and more likely to cause an upset.

### Implications for models

All current methods are anchored 65–75% to the seed-spline, which is calibrated on historical win rates and is inherently chalky. The `upset_aware_ensemble` addresses this by:
1. Adding `MasseyVsSeedGap`, `DefFirstRatio`, `RecentMomentum`, and `HistMatchupRate_Diff` as explicit features
2. Reducing the spline anchor to 30% for men's (from 70%), giving the ML component room to move predictions toward upset-prone teams

---

## CLI Tools

### Evaluate all methods (leaderboard)

```bash
# Full evaluation across all methods × years × genders:
poetry run python utils/eval_framework.py

# Specific subset:
poetry run python utils/eval_framework.py \
    --methods modeh7 xgb_ensemble upset_aware_ensemble \
    --years 2024 2025 --genders M

# View existing results without regenerating:
poetry run python utils/eval_framework.py --report-only

# Include slow methods (meta_ensemble, elo, elo_enhanced):
poetry run python utils/eval_framework.py --include-slow
```

All flags:
```
--methods METHOD [...]   Methods to run (default: 10 fast methods, including modeh7)
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

### Generate a Kaggle submission

The recommended workflow: evaluate methods, then generate a blended submission for the target season.

**Step 1 — Evaluate (or refresh) all methods:**
```bash
poetry run python utils/eval_framework.py --years 2023 2024 2025
```

**Step 2 — Generate and blend all methods into one CSV (default):**
```bash
# Average all methods from eval_results (default behavior):
poetry run python utils/generate_submission.py --season 2026

# Specific methods only:
poetry run python utils/generate_submission.py \
    --season 2026 --methods modeh7 xgb_ensemble upset_aware_ensemble

# Top N methods by historical Brier, averaged:
poetry run python utils/generate_submission.py \
    --season 2026 --top-n 3 --strategy average_topn \
    --years-for-ranking 2024 2025

# Use single best method:
poetry run python utils/generate_submission.py \
    --season 2026 --top-n 1 --strategy top1
```

All `generate_submission.py` flags:
```
--season N                                    Target season year (required)
--methods METHOD [...]                        Explicit method list (skips eval-results ranking)
--top-n N                                     Limit to top N from eval_results (default: all)
--strategy top1|average_topn|average_all|average_top3
                                              Combination strategy (default: average_all)
--years-for-ranking Y [...]                   Years for ranking methods (default: all)
--eval-results PATH                           Path to eval_results.csv
--data-dir PATH                               Data directory (default: data/{season})
--output-dir PATH                             Output root (default: output/)
--mens-method METHOD                          Override method for men's only
--womens-method METHOD                        Override method for women's only
--no-skip-existing                            Regenerate even if CSV exists
--no-brackets                                 Skip bracket PNG generation
```

Output: `output/{season}/submission_{season}_{strategy}.csv`

**Step 3 (optional) — Blend already-generated CSVs with a different weighting:**
```bash
# Equal blend of top 2 methods per gender:
poetry run python utils/kaggle_submission.py --season 2026

# Inverse-Brier weighted blend of top 3, ranked on recent years:
poetry run python utils/kaggle_submission.py \
    --season 2026 \
    --strategy weighted \
    --top-n 3 \
    --years-for-ranking 2024 2025
```

Output: `output/{season}/kaggle_submission_{season}.csv`

### Run upset analysis

```bash
# Full analysis using cumulative 2026 data (covers 2003-2025):
poetry run python utils/upset_analysis.py --data-dir data/2026

# Save to custom path:
poetry run python utils/upset_analysis.py \
    --data-dir data/2026 --output output/upset_analysis.csv
```

Outputs (all in same directory as `--output`):
- `upset_analysis.csv` — full game-level dataset with all features
- `upset_by_round.csv` — upset rates per round
- `upset_by_matchup.csv` — upset rates per seed pair
- `upset_correlations.csv` — point-biserial correlations
- `upset_lr_coefficients.csv` — logistic regression coefficients

### Generate bracket PNGs from existing submissions

Renders bracket visualizations from any already-generated submission CSV — no model retraining needed.

```bash
# All years, both genders, all found submissions:
poetry run python utils/generate_brackets_from_submissions.py

# Specific year/gender/method:
poetry run python utils/generate_brackets_from_submissions.py \
    --years 2025 --genders M --methods modeh7 xgb_ensemble

# Regenerate even if PNGs already exist:
poetry run python utils/generate_brackets_from_submissions.py \
    --years 2025 --no-skip-existing
```

All flags:
```
--years YEAR [...]       Seasons to process (default: 2022 2023 2024 2025)
--genders M W            Genders (default: both)
--methods METHOD [...]   Only process these methods (default: all found)
--data-dir PATH          Cumulative data root (default: data/2026)
--output-dir PATH        Output root (default: output/)
--no-skip-existing       Regenerate even if bracket PNG already exists
```

Output:
- `output/{year}/{method}/bracket_{gender}.png` — predicted bracket
- `output/{year}/{method}/bracket_{gender}_historical.png` — predicted vs actual (2022–2025)

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
--base-data-dir PATH     Data root (default: data/)
--lookback-years N       Training window in years (default: 8)
--no-skip-existing       Regenerate existing artifacts
--skip-brackets          Skip bracket PNG generation (faster)
--artifact-suffix SUFFIX Optional suffix for output file naming
```

Output: `output/{year}/{method}/`, `output/scoring_results.csv`

### Other utilities

```bash
# Download ESPN team logos for bracket visualization:
poetry run python utils/download_logos.py
# → data/logos/{M|W}/{team_id}.png
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
├── scoring_results.csv       # Aggregate scoring table
├── upset_analysis.csv        # Game-level upset dataset (from upset_analysis.py)
├── upset_by_round.csv        # Upset rates per round
├── upset_by_matchup.csv      # Upset rates per seed matchup
├── upset_correlations.csv    # Feature correlations with upset outcome
├── upset_lr_coefficients.csv # Logistic regression upset predictors
└── {year}/
    ├── {method}/             # Per-method flat folder
    │   ├── {method}_{gender}.csv        # Submission predictions
    │   ├── bracket_{gender}.png         # Bracket visualization
    │   ├── bracket_{gender}_historical.png
    │   └── bracket_{gender}.html
    ├── features/             # {gender}/feature_dataset.csv (runtime cache)
    └── submission_{year}_{strategy}.csv  # Final Kaggle submission
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
