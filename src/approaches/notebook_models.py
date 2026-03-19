from itertools import combinations

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data_classes.processing.DataManager import MarchMadnessDataManager

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def generate_notebook_submission(method: str, data_dir: str, season: int, gender: str) -> pd.DataFrame:
    if method == "baseline_massey":
        return _generate_baseline_massey_submission(data_dir, season, gender)
    if method == "xgb_ensemble":
        return _generate_xgb_ensemble_submission(data_dir, season, gender)
    if method == "xgb_ensemble_v2":
        return _generate_xgb_ensemble_v2_submission(data_dir, season, gender)
    if method == "massey_direct":
        return _generate_massey_direct_submission(data_dir, season, gender)
    if method == "seed_spline_only":
        return _generate_seed_spline_only_submission(data_dir, season, gender)
    if method == "recency_xgb":
        return _generate_recency_xgb_submission(data_dir, season, gender)
    if method == "massey_blend":
        return _generate_massey_blend_submission(data_dir, season, gender)
    if method == "poisson_margin":
        return _generate_poisson_margin_submission(data_dir, season, gender)
    if method == "meta_ensemble":
        return _generate_meta_ensemble_submission(data_dir, season, gender)
    if method == "seed_matchup_calibration":
        return _generate_seed_matchup_calibration_submission(data_dir, season, gender)
    if method == "modeh7":
        return _generate_modeh7_submission(data_dir, season, gender)
    if method == "upset_aware_ensemble":
        return _generate_upset_aware_ensemble_submission(data_dir, season, gender)
    raise ValueError(f"Unsupported notebook-backed method: {method}")


def _get_data_manager(data_dir: str, season: int, gender: str) -> MarchMadnessDataManager:
    dm = MarchMadnessDataManager(data_dir, gender=gender, current_season=season)
    dm.load_data()
    return dm


def _seed_number(seed_value: str) -> int:
    return int(str(seed_value)[1:3])


def _get_current_seeded_teams(dm: MarchMadnessDataManager, season: int) -> list[int]:
    seeds = dm.data["tourney_seeds"]
    season_seeds = seeds[seeds["Season"] == season].copy()
    return sorted(season_seeds["TeamID"].unique().tolist())


def _iter_matchups(team_ids: list[int]):
    for team1_id, team2_id in combinations(sorted(team_ids), 2):
        yield int(team1_id), int(team2_id)


def _build_submission_from_predictions(season: int, predictions: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(predictions)
    df["Pred"] = df["Pred"].clip(0.025, 0.975)
    return df


def _log_progress(label: str, index: int, total: int, every: int = 250):
    if total <= 0:
        return
    if index == total or index % every == 0:
        print(f"{label}: {index}/{total} ({index / total:.1%})")


def _xgb_classifier(random_state: int = 42):
    if xgb is None:
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=random_state,
        )
    return xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )


def _compute_win_pcts(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, grp in results_df.groupby("Season"):
        for loc, mask in [("H", grp["WLoc"] == "H"), ("A", grp["WLoc"] == "A"), ("N", grp["WLoc"] == "N")]:
            wins = grp[mask].groupby("WTeamID").size().rename("wins")
            loss_loc = {"H": "A", "A": "H", "N": "N"}[loc]
            losses = grp[grp["WLoc"] == loss_loc].groupby("LTeamID").size().rename("losses")
            combined = pd.concat([wins, losses], axis=1).fillna(0)
            combined["total"] = combined["wins"] + combined["losses"]
            combined["pct"] = combined["wins"] / combined["total"].clip(lower=1)
            combined["Season"] = season
            combined["loc"] = loc
            rows.append(combined.reset_index().rename(columns={"index": "TeamID", "WTeamID": "TeamID"}))

        wins = grp.groupby("WTeamID").size()
        losses = grp.groupby("LTeamID").size()
        idx = wins.index.union(losses.index)
        frame = pd.DataFrame(
            {
                "wins": wins.reindex(idx, fill_value=0),
                "losses": losses.reindex(idx, fill_value=0),
            }
        )
        frame["total"] = frame["wins"] + frame["losses"]
        frame["pct"] = frame["wins"] / frame["total"].clip(lower=1)
        frame["Season"] = season
        frame["loc"] = "Overall"
        rows.append(frame.reset_index().rename(columns={"index": "TeamID"}))

    pivot = pd.concat(rows, ignore_index=True)
    return pivot.pivot_table(index=["Season", "TeamID"], columns="loc", values="pct").reset_index()


def _compute_massey_features(massey_df: pd.DataFrame, cutoff_day: int = 128) -> pd.DataFrame:
    late = massey_df[massey_df["RankingDayNum"] <= cutoff_day].copy()
    avg_rank = (
        late.groupby(["Season", "TeamID"])["OrdinalRank"]
        .mean()
        .reset_index(name="MasseyRankAvg")
    )
    avg_rank["MasseyRankNorm"] = avg_rank.groupby("Season")["MasseyRankAvg"].transform(
        lambda x: 1 - (x - x.min()) / (x.max() - x.min() + 1e-9)
    )
    return avg_rank[["Season", "TeamID", "MasseyRankAvg", "MasseyRankNorm"]]


def _build_baseline_matchup_features(
    tourney_df: pd.DataFrame,
    win_pct_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    massey_df: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    for _, game in tourney_df.iterrows():
        season = int(game["Season"])
        winner = int(game["WTeamID"])
        loser = int(game["LTeamID"])
        team1, team2 = min(winner, loser), max(winner, loser)
        actual = 1.0 if team1 == winner else 0.0

        def get_wp(team_id: int, column: str) -> float:
            row = win_pct_df[(win_pct_df["Season"] == season) & (win_pct_df["TeamID"] == team_id)]
            return float(row[column].values[0]) if len(row) and column in row.columns else 0.5

        def get_seed(team_id: int) -> int:
            row = seeds_df[(seeds_df["Season"] == season) & (seeds_df["TeamID"] == team_id)]
            return _seed_number(row.iloc[0]["Seed"]) if len(row) else 8

        feat = {
            "Season": season,
            "Team1ID": team1,
            "Team2ID": team2,
            "SeedDiff": get_seed(team1) - get_seed(team2),
            "WPct1": get_wp(team1, "Overall"),
            "WPct2": get_wp(team2, "Overall"),
            "WPctH1": get_wp(team1, "H"),
            "WPctH2": get_wp(team2, "H"),
            "WPctA1": get_wp(team1, "A"),
            "WPctA2": get_wp(team2, "A"),
            "WPctN1": get_wp(team1, "N"),
            "WPctN2": get_wp(team2, "N"),
            "Result": actual,
        }

        if massey_df is not None:
            def get_massey(team_id: int) -> float:
                row = massey_df[(massey_df["Season"] == season) & (massey_df["TeamID"] == team_id)]
                return float(row["MasseyRankNorm"].values[0]) if len(row) else 0.5

            feat["Massey1"] = get_massey(team1)
            feat["Massey2"] = get_massey(team2)
            feat["MasseyDiff"] = feat["Massey1"] - feat["Massey2"]

        rows.append(feat)

    return pd.DataFrame(rows)


def _generate_baseline_massey_submission(data_dir: str, season: int, gender: str) -> pd.DataFrame:
    print(f"Preparing baseline_massey submission for {gender} {season}...")
    dm = _get_data_manager(data_dir, season, gender)
    reg = dm.data["regular_season"]
    tourney = dm.data["tourney_results"]
    seeds = dm.data["tourney_seeds"]

    win_pct = _compute_win_pcts(reg)
    massey_features = _compute_massey_features(dm.data["rankings"]) if dm.rankings_available else None

    train_tourney = tourney[(tourney["Season"] >= 2010) & (tourney["Season"] < season)]
    print(f"Building baseline training features from {len(train_tourney)} tournament games...")
    features = _build_baseline_matchup_features(train_tourney, win_pct, seeds, massey_features)
    feat_cols = [col for col in features.columns if col not in ["Season", "Team1ID", "Team2ID", "Result"]]

    model = _xgb_classifier()
    print(f"Fitting baseline model on shape={features[feat_cols].shape}...")
    model.fit(features[feat_cols].fillna(0), features["Result"])

    seed_spline = UnivariateSpline(
        np.sort(features["SeedDiff"].values),
        features.sort_values("SeedDiff")["Result"].values,
        s=len(features),
        ext=3,
    )
    blend = 0.3

    predictions = []
    seeded_teams = _get_current_seeded_teams(dm, season)
    total_matchups = len(seeded_teams) * (len(seeded_teams) - 1) // 2
    print(f"Generating baseline predictions for {total_matchups} matchups...")
    for index, (team1, team2) in enumerate(_iter_matchups(seeded_teams), start=1):
        def get_wp(team_id: int, column: str) -> float:
            row = win_pct[(win_pct["Season"] == season) & (win_pct["TeamID"] == team_id)]
            return float(row[column].values[0]) if len(row) and column in row.columns else 0.5

        def get_seed(team_id: int) -> int:
            row = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team_id)]
            return _seed_number(row.iloc[0]["Seed"]) if len(row) else 8

        feat = {
            "SeedDiff": get_seed(team1) - get_seed(team2),
            "WPct1": get_wp(team1, "Overall"),
            "WPct2": get_wp(team2, "Overall"),
            "WPctH1": get_wp(team1, "H"),
            "WPctH2": get_wp(team2, "H"),
            "WPctA1": get_wp(team1, "A"),
            "WPctA2": get_wp(team2, "A"),
            "WPctN1": get_wp(team1, "N"),
            "WPctN2": get_wp(team2, "N"),
        }
        if massey_features is not None:
            def get_massey(team_id: int) -> float:
                row = massey_features[(massey_features["Season"] == season) & (massey_features["TeamID"] == team_id)]
                return float(row["MasseyRankNorm"].values[0]) if len(row) else 0.5

            feat["Massey1"] = get_massey(team1)
            feat["Massey2"] = get_massey(team2)
            feat["MasseyDiff"] = feat["Massey1"] - feat["Massey2"]

        row_df = pd.DataFrame([feat])[feat_cols].fillna(0)
        model_pred = float(model.predict_proba(row_df)[0, 1])
        spline_pred = float(np.clip(seed_spline(feat["SeedDiff"]), 0.025, 0.975))
        final_pred = blend * model_pred + (1 - blend) * spline_pred
        predictions.append({"ID": f"{season}_{team1}_{team2}", "Pred": final_pred})
        _log_progress("  Baseline predictions", index, total_matchups)

    return _build_submission_from_predictions(season, predictions)


def _compute_compact_features(results_df: pd.DataFrame) -> pd.DataFrame:
    records = {}
    for (season, team_id), grp_w in results_df.groupby(["Season", "WTeamID"]):
        grp_l = results_df[(results_df["Season"] == season) & (results_df["LTeamID"] == team_id)]
        for loc in ["H", "A", "N"]:
            wins_loc = len(grp_w[grp_w["WLoc"] == loc])
            losses_loc = len(grp_l[grp_l["WLoc"] == {"H": "A", "A": "H", "N": "N"}[loc]])
            total_loc = wins_loc + losses_loc
            key = (season, team_id)
            records.setdefault(key, {"Season": season, "TeamID": team_id})
            records[key][f"WinPct_{loc}"] = wins_loc / total_loc if total_loc else 0.5

    df = pd.DataFrame(list(records.values()))
    wins = results_df.groupby(["Season", "WTeamID"]).size().reset_index(name="Wins").rename(columns={"WTeamID": "TeamID"})
    losses = results_df.groupby(["Season", "LTeamID"]).size().reset_index(name="Losses").rename(columns={"LTeamID": "TeamID"})
    games = wins.merge(losses, on=["Season", "TeamID"], how="outer").fillna(0)
    games["WinPct_Overall"] = games["Wins"] / (games["Wins"] + games["Losses"]).clip(lower=1)

    margins = results_df.copy()
    margins["WMargin"] = margins["WScore"] - margins["LScore"]
    win_margin = margins.groupby(["Season", "WTeamID"])["WMargin"].mean().reset_index(name="AvgWinMargin").rename(columns={"WTeamID": "TeamID"})
    loss_margin = margins.groupby(["Season", "LTeamID"])["WMargin"].mean().reset_index(name="AvgLossMargin").rename(columns={"LTeamID": "TeamID"})

    df = df.merge(games[["Season", "TeamID", "WinPct_Overall", "Wins", "Losses"]], on=["Season", "TeamID"], how="outer")
    df = df.merge(win_margin, on=["Season", "TeamID"], how="left")
    df = df.merge(loss_margin, on=["Season", "TeamID"], how="left")
    df["AvgMargin"] = df["AvgWinMargin"].fillna(0) - df["AvgLossMargin"].fillna(0)
    return df.fillna(0.5)


def _compute_detailed_features(detailed_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if detailed_df is None:
        return None

    stats = []
    for season, grp in detailed_df.groupby("Season"):
        winners = (
            grp.groupby("WTeamID")
            .agg(
                WFGM=("WFGM", "mean"),
                WFGA=("WFGA", "mean"),
                WFGM3=("WFGM3", "mean"),
                WFGA3=("WFGA3", "mean"),
                WFTM=("WFTM", "mean"),
                WFTA=("WFTA", "mean"),
                WOR=("WOR", "mean"),
                WDR=("WDR", "mean"),
                WAst=("WAst", "mean"),
                WTO=("WTO", "mean"),
                WStl=("WStl", "mean"),
            )
            .reset_index()
            .rename(columns={"WTeamID": "TeamID"})
        )
        winners["FGPct"] = winners["WFGM"] / winners["WFGA"].clip(lower=1)
        winners["FG3Pct"] = winners["WFGM3"] / winners["WFGA3"].clip(lower=1)
        winners["FTPct"] = winners["WFTM"] / winners["WFTA"].clip(lower=1)
        winners["AstTORate"] = winners["WAst"] / winners["WTO"].clip(lower=1)
        winners["RebRate"] = winners["WOR"] + winners["WDR"]
        winners["Season"] = season
        stats.append(winners[["Season", "TeamID", "FGPct", "FG3Pct", "FTPct", "AstTORate", "RebRate", "WStl"]])

    return pd.concat(stats, ignore_index=True) if stats else None


def _compute_massey_multi(massey_df: pd.DataFrame) -> pd.DataFrame:
    checkpoints = [("Early", 15, 50), ("Mid", 80, 100), ("Late", 120, 133)]
    frames = []
    for label, day_min, day_max in checkpoints:
        sub = massey_df[(massey_df["RankingDayNum"] >= day_min) & (massey_df["RankingDayNum"] <= day_max)]
        avg = sub.groupby(["Season", "TeamID"])["OrdinalRank"].mean().reset_index(name=f"MasseyRank_{label}")
        avg[f"MasseyNorm_{label}"] = avg.groupby("Season")[f"MasseyRank_{label}"].transform(
            lambda x: 1 - (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
        frames.append(avg)

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on=["Season", "TeamID"], how="outer")
    result["MasseyTrajectory"] = result["MasseyNorm_Late"] - result.get("MasseyNorm_Early", result["MasseyNorm_Late"])
    return result.fillna(0.5)


def _build_ensemble_matchup_df(
    tourney_df: pd.DataFrame,
    compact_feats: pd.DataFrame,
    detailed_feats: pd.DataFrame | None,
    seeds_df: pd.DataFrame,
    massey_feats: pd.DataFrame | None,
) -> pd.DataFrame:
    all_feats = compact_feats.copy()
    if detailed_feats is not None:
        all_feats = all_feats.merge(detailed_feats, on=["Season", "TeamID"], how="left")
    if massey_feats is not None:
        all_feats = all_feats.merge(massey_feats, on=["Season", "TeamID"], how="left")

    feat_cols = [col for col in all_feats.columns if col not in ["Season", "TeamID"]]
    rows = []
    for _, game in tourney_df.iterrows():
        season = int(game["Season"])
        winner = int(game["WTeamID"])
        loser = int(game["LTeamID"])
        team1, team2 = min(winner, loser), max(winner, loser)
        actual = 1.0 if team1 == winner else 0.0

        feat1 = all_feats[(all_feats["Season"] == season) & (all_feats["TeamID"] == team1)]
        feat2 = all_feats[(all_feats["Season"] == season) & (all_feats["TeamID"] == team2)]
        if len(feat1) == 0 or len(feat2) == 0:
            continue
        feat1 = feat1.iloc[0]
        feat2 = feat2.iloc[0]

        row = {
            "Season": season,
            "Team1ID": team1,
            "Team2ID": team2,
            "Result": actual,
            "SeedDiff": _seed_number(
                seeds_df[(seeds_df["Season"] == season) & (seeds_df["TeamID"] == team1)].iloc[0]["Seed"]
            ) - _seed_number(
                seeds_df[(seeds_df["Season"] == season) & (seeds_df["TeamID"] == team2)].iloc[0]["Seed"]
            ),
        }

        for col in feat_cols:
            value1 = feat1.get(col, 0)
            value2 = feat2.get(col, 0)
            row[f"{col}_1"] = value1
            row[f"{col}_2"] = value2
            row[f"{col}_Diff"] = value1 - value2

        rows.append(row)

    return pd.DataFrame(rows), all_feats


def _ensemble_models():
    return [
        ("HistGBM", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)),
        ("RF", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
        ("LR", Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=1000))])),
        ("SVC", Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, C=1.0))])),
        ("XGB", _xgb_classifier()),
    ]


def _generate_xgb_ensemble_submission(data_dir: str, season: int, gender: str) -> pd.DataFrame:
    print(f"Preparing xgb_ensemble submission for {gender} {season}...")
    dm = _get_data_manager(data_dir, season, gender)
    reg = dm.data["regular_season"]
    detailed = dm.data.get("regular_season_detailed")
    tourney = dm.data["tourney_results"]
    seeds = dm.data["tourney_seeds"]

    compact_feats = _compute_compact_features(reg)
    detailed_feats = _compute_detailed_features(detailed)
    massey_feats = _compute_massey_multi(dm.data["rankings"]) if dm.rankings_available else None

    train_tourney = tourney[(tourney["Season"] >= 2010) & (tourney["Season"] < season)]
    print(f"Building ensemble training features from {len(train_tourney)} tournament games...")
    feat_df, all_feats = _build_ensemble_matchup_df(
        train_tourney, compact_feats, detailed_feats, seeds, massey_feats
    )
    model_feat_cols = [col for col in feat_df.columns if col not in ["Season", "Team1ID", "Team2ID", "Result"]]
    X = feat_df[model_feat_cols].fillna(0)
    y = feat_df["Result"]
    trained_models = []
    for name, clf in _ensemble_models():
        print(f"Fitting ensemble component {name} on shape={X.shape}...")
        clf.fit(X, y)
        trained_models.append((name, clf))
        print(f"  Component {name} fit complete.")

    seed_spline = UnivariateSpline(
        np.sort(feat_df["SeedDiff"].values),
        feat_df.sort_values("SeedDiff")["Result"].values,
        s=len(feat_df),
        ext=3,
    )
    blend_weight = 0.3

    predictions = []
    current_feats = all_feats[all_feats["Season"] == season].set_index("TeamID")
    seeded_teams = _get_current_seeded_teams(dm, season)
    total_matchups = len(seeded_teams) * (len(seeded_teams) - 1) // 2
    print(f"Generating ensemble predictions for {total_matchups} matchups...")
    for index, (team1, team2) in enumerate(_iter_matchups(seeded_teams), start=1):
        row = {
            "SeedDiff": _seed_number(
                seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team1)].iloc[0]["Seed"]
            ) - _seed_number(
                seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team2)].iloc[0]["Seed"]
            )
        }
        feat1 = current_feats.loc[team1] if team1 in current_feats.index else pd.Series(dtype=float)
        feat2 = current_feats.loc[team2] if team2 in current_feats.index else pd.Series(dtype=float)

        for col in [c for c in all_feats.columns if c not in ["Season", "TeamID"]]:
            value1 = feat1.get(col, 0)
            value2 = feat2.get(col, 0)
            row[f"{col}_1"] = value1
            row[f"{col}_2"] = value2
            row[f"{col}_Diff"] = value1 - value2

        row_df = pd.DataFrame([row])[model_feat_cols].fillna(0)
        preds = [clf.predict_proba(row_df)[0, 1] for _, clf in trained_models]
        ensemble_pred = float(np.mean(preds))
        spline_pred = float(np.clip(seed_spline(row["SeedDiff"]), 0.025, 0.975))
        final_pred = blend_weight * ensemble_pred + (1 - blend_weight) * spline_pred
        predictions.append({"ID": f"{season}_{team1}_{team2}", "Pred": final_pred})
        _log_progress("  Ensemble predictions", index, total_matchups)

    return _build_submission_from_predictions(season, predictions)


# ── Individual Massey system features ────────────────────────────────────────

_KEY_MASSEY_SYSTEMS = ["POM", "SAG", "DOK", "MOR", "MAS", "RPI"]


def _compute_massey_individual(massey_df: pd.DataFrame, cutoff_day: int = 128) -> pd.DataFrame:
    """Per-system normalized Massey ranks for each KEY system."""
    late = massey_df[massey_df["RankingDayNum"] <= cutoff_day].copy()
    result = None
    for sys in _KEY_MASSEY_SYSTEMS:
        sub = late[late["SystemName"] == sys] if "SystemName" in late.columns else pd.DataFrame()
        if sub.empty:
            continue
        avg = sub.groupby(["Season", "TeamID"])["OrdinalRank"].mean().reset_index(name=f"MasseyRank_{sys}")
        avg[f"MasseyNorm_{sys}"] = avg.groupby("Season")[f"MasseyRank_{sys}"].transform(
            lambda x: 1 - (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
        avg = avg[["Season", "TeamID", f"MasseyNorm_{sys}"]]
        result = avg if result is None else result.merge(avg, on=["Season", "TeamID"], how="outer")
    return result.fillna(0.5) if result is not None else pd.DataFrame()


def _compute_conf_tourney_features(conf_tourney_df: pd.DataFrame | None) -> pd.DataFrame:
    """Conference tournament stats: games, wins, win pct, champ flag."""
    if conf_tourney_df is None or conf_tourney_df.empty:
        return pd.DataFrame()

    rows = []
    for (season, team_id), grp in conf_tourney_df.groupby(["Season", "WTeamID"]):
        rows.append({"Season": season, "TeamID": team_id, "_ConfWins": len(grp)})
    for (season, team_id), grp in conf_tourney_df.groupby(["Season", "LTeamID"]):
        rows.append({"Season": season, "TeamID": team_id, "_ConfLosses": len(grp)})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.groupby(["Season", "TeamID"]).sum(numeric_only=True).reset_index()
    df["ConfTourneyWins"] = df.get("_ConfWins", pd.Series(0, index=df.index)).fillna(0)
    df["ConfTourneyGames"] = (
        df.get("_ConfWins", pd.Series(0, index=df.index)).fillna(0)
        + df.get("_ConfLosses", pd.Series(0, index=df.index)).fillna(0)
    )
    df["ConfTourneyWinPct"] = df["ConfTourneyWins"] / df["ConfTourneyGames"].clip(lower=1)

    # Conference champion = winner of the last game played in each conference
    champs = set()
    for (season,), grp in conf_tourney_df.groupby(["Season"]):
        max_day = grp["DayNum"].max()
        champs.update(
            tuple((int(season), int(t)))
            for t in grp[grp["DayNum"] == max_day]["WTeamID"].unique()
        )
    df["ConfTourneyChamp"] = df.apply(
        lambda r: 1.0 if (int(r["Season"]), int(r["TeamID"])) in champs else 0.0, axis=1
    )
    return df[["Season", "TeamID", "ConfTourneyGames", "ConfTourneyWins", "ConfTourneyWinPct", "ConfTourneyChamp"]]


# ── xgb_ensemble_v2 ──────────────────────────────────────────────────────────

def _build_ensemble_v2_matchup_df(
    tourney_df: pd.DataFrame,
    compact_feats: pd.DataFrame,
    detailed_feats: pd.DataFrame | None,
    seeds_df: pd.DataFrame,
    massey_feats: pd.DataFrame | None,
    massey_individual: pd.DataFrame | None,
    conf_feats: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_feats = compact_feats.copy()
    if detailed_feats is not None:
        all_feats = all_feats.merge(detailed_feats, on=["Season", "TeamID"], how="left")
    if massey_feats is not None:
        all_feats = all_feats.merge(massey_feats, on=["Season", "TeamID"], how="left")
    if massey_individual is not None and not massey_individual.empty:
        all_feats = all_feats.merge(massey_individual, on=["Season", "TeamID"], how="left")
    if conf_feats is not None and not conf_feats.empty:
        all_feats = all_feats.merge(conf_feats, on=["Season", "TeamID"], how="left")

    feat_cols = [col for col in all_feats.columns if col not in ["Season", "TeamID"]]
    rows = []
    for _, game in tourney_df.iterrows():
        season = int(game["Season"])
        winner = int(game["WTeamID"])
        loser = int(game["LTeamID"])
        team1, team2 = min(winner, loser), max(winner, loser)
        actual = 1.0 if team1 == winner else 0.0

        feat1 = all_feats[(all_feats["Season"] == season) & (all_feats["TeamID"] == team1)]
        feat2 = all_feats[(all_feats["Season"] == season) & (all_feats["TeamID"] == team2)]
        if len(feat1) == 0 or len(feat2) == 0:
            continue
        feat1 = feat1.iloc[0]
        feat2 = feat2.iloc[0]

        seed1_rows = seeds_df[(seeds_df["Season"] == season) & (seeds_df["TeamID"] == team1)]
        seed2_rows = seeds_df[(seeds_df["Season"] == season) & (seeds_df["TeamID"] == team2)]
        if len(seed1_rows) == 0 or len(seed2_rows) == 0:
            continue

        row = {
            "Season": season,
            "Team1ID": team1,
            "Team2ID": team2,
            "Result": actual,
            "SeedDiff": _seed_number(seed1_rows.iloc[0]["Seed"]) - _seed_number(seed2_rows.iloc[0]["Seed"]),
        }
        for col in feat_cols:
            v1 = feat1.get(col, 0)
            v2 = feat2.get(col, 0)
            row[f"{col}_1"] = v1
            row[f"{col}_2"] = v2
            row[f"{col}_Diff"] = v1 - v2
        rows.append(row)

    return pd.DataFrame(rows), all_feats


def _generate_xgb_ensemble_v2_submission(data_dir: str, season: int, gender: str) -> pd.DataFrame:
    print(f"Preparing xgb_ensemble_v2 submission for {gender} {season}...")
    dm = _get_data_manager(data_dir, season, gender)
    reg = dm.data["regular_season"]
    detailed = dm.data.get("regular_season_detailed")
    tourney = dm.data["tourney_results"]
    seeds = dm.data["tourney_seeds"]

    compact_feats = _compute_compact_features(reg)
    detailed_feats = _compute_detailed_features(detailed)
    massey_feats = _compute_massey_multi(dm.data["rankings"]) if dm.rankings_available else None
    massey_individual = _compute_massey_individual(dm.data["rankings"]) if dm.rankings_available else None

    conf_tourney_df = dm.data.get("conf_tourney")
    conf_feats = _compute_conf_tourney_features(conf_tourney_df)
    if conf_feats.empty:
        conf_feats = None

    train_tourney = tourney[(tourney["Season"] >= 2010) & (tourney["Season"] < season)]
    print(f"Building ensemble_v2 training features from {len(train_tourney)} tournament games...")
    feat_df, all_feats = _build_ensemble_v2_matchup_df(
        train_tourney, compact_feats, detailed_feats, seeds,
        massey_feats, massey_individual, conf_feats,
    )
    model_feat_cols = [col for col in feat_df.columns if col not in ["Season", "Team1ID", "Team2ID", "Result"]]
    X = feat_df[model_feat_cols].fillna(0)
    y = feat_df["Result"]

    trained_models = []
    for name, clf in _ensemble_models():
        print(f"Fitting ensemble_v2 component {name} on shape={X.shape}...")
        clf.fit(X, y)
        trained_models.append((name, clf))
        print(f"  Component {name} fit complete.")

    seed_spline = UnivariateSpline(
        np.sort(feat_df["SeedDiff"].values),
        feat_df.sort_values("SeedDiff")["Result"].values,
        s=len(feat_df),
        ext=3,
    )
    # Women get heavier spline weight since features are weaker
    blend_weight = 0.2 if gender == "W" else 0.3

    predictions = []
    current_feats = all_feats[all_feats["Season"] == season].set_index("TeamID")
    seeded_teams = _get_current_seeded_teams(dm, season)
    total_matchups = len(seeded_teams) * (len(seeded_teams) - 1) // 2
    print(f"Generating ensemble_v2 predictions for {total_matchups} matchups...")
    for index, (team1, team2) in enumerate(_iter_matchups(seeded_teams), start=1):
        seed1_rows = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team1)]
        seed2_rows = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team2)]
        if len(seed1_rows) == 0 or len(seed2_rows) == 0:
            continue
        row = {
            "SeedDiff": _seed_number(seed1_rows.iloc[0]["Seed"]) - _seed_number(seed2_rows.iloc[0]["Seed"])
        }
        feat1 = current_feats.loc[team1] if team1 in current_feats.index else pd.Series(dtype=float)
        feat2 = current_feats.loc[team2] if team2 in current_feats.index else pd.Series(dtype=float)
        for col in [c for c in all_feats.columns if c not in ["Season", "TeamID"]]:
            v1 = feat1.get(col, 0)
            v2 = feat2.get(col, 0)
            row[f"{col}_1"] = v1
            row[f"{col}_2"] = v2
            row[f"{col}_Diff"] = v1 - v2

        row_df = pd.DataFrame([row])[model_feat_cols].fillna(0)
        preds = [clf.predict_proba(row_df)[0, 1] for _, clf in trained_models]
        ensemble_pred = float(np.mean(preds))
        spline_pred = float(np.clip(seed_spline(row["SeedDiff"]), 0.025, 0.975))
        final_pred = blend_weight * ensemble_pred + (1 - blend_weight) * spline_pred
        predictions.append({"ID": f"{season}_{team1}_{team2}", "Pred": final_pred})
        _log_progress("  Ensemble_v2 predictions", index, total_matchups)

    return _build_submission_from_predictions(season, predictions)


# ── massey_direct ─────────────────────────────────────────────────────────────

def _generate_massey_direct_submission(data_dir: str, season: int, gender: str) -> pd.DataFrame:
    """Logistic regression on top Massey systems + seed, blended 0.25/0.75 with spline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    print(f"Preparing massey_direct submission for {gender} {season}...")
    dm = _get_data_manager(data_dir, season, gender)
    tourney = dm.data["tourney_results"]
    seeds = dm.data["tourney_seeds"]

    has_massey = dm.rankings_available
    massey_ind = _compute_massey_individual(dm.data["rankings"]) if has_massey else None
    sys_cols = [f"MasseyNorm_{s}" for s in _KEY_MASSEY_SYSTEMS]

    def _get_feat_row(t1: int, t2: int, s: int) -> dict:
        s1_row = seeds[(seeds["Season"] == s) & (seeds["TeamID"] == t1)]
        s2_row = seeds[(seeds["Season"] == s) & (seeds["TeamID"] == t2)]
        sd = (
            (_seed_number(s1_row.iloc[0]["Seed"]) - _seed_number(s2_row.iloc[0]["Seed"]))
            if len(s1_row) and len(s2_row) else 0
        )
        row = {"SeedDiff": float(sd)}
        if massey_ind is not None and not massey_ind.empty:
            for col in sys_cols:
                if col not in massey_ind.columns:
                    continue
                r1 = massey_ind[(massey_ind["Season"] == s) & (massey_ind["TeamID"] == t1)]
                r2 = massey_ind[(massey_ind["Season"] == s) & (massey_ind["TeamID"] == t2)]
                v1 = float(r1.iloc[0][col]) if len(r1) else 0.5
                v2 = float(r2.iloc[0][col]) if len(r2) else 0.5
                row[f"{col}_Diff"] = v1 - v2
        return row

    train_tourney = tourney[(tourney["Season"] >= 2010) & (tourney["Season"] < season)]
    print(f"Building massey_direct training features from {len(train_tourney)} tournament games...")
    feat_rows = []
    for _, game in train_tourney.iterrows():
        s = int(game["Season"])
        w, l = int(game["WTeamID"]), int(game["LTeamID"])
        t1, t2 = min(w, l), max(w, l)
        row = _get_feat_row(t1, t2, s)
        row["Result"] = 1.0 if t1 == w else 0.0
        feat_rows.append(row)

    feat_df = pd.DataFrame(feat_rows).fillna(0)
    feat_cols = [c for c in feat_df.columns if c != "Result"]

    # If no Massey systems available (women), fall back to spline-only
    model = None
    if len(feat_cols) > 1 or (len(feat_cols) == 1 and feat_cols[0] != "SeedDiff"):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=1000)),
        ])
        print(f"Fitting massey_direct LR on shape={feat_df[feat_cols].shape}...")
        model.fit(feat_df[feat_cols], feat_df["Result"])

    seed_spline = UnivariateSpline(
        np.sort(feat_df["SeedDiff"].values),
        feat_df.sort_values("SeedDiff")["Result"].values,
        s=len(feat_df),
        ext=3,
    )
    blend_weight = 0.25 if model is not None else 0.0

    predictions = []
    seeded_teams = _get_current_seeded_teams(dm, season)
    total_matchups = len(seeded_teams) * (len(seeded_teams) - 1) // 2
    print(f"Generating massey_direct predictions for {total_matchups} matchups...")
    for index, (team1, team2) in enumerate(_iter_matchups(seeded_teams), start=1):
        row = _get_feat_row(team1, team2, season)
        spline_pred = float(np.clip(seed_spline(row["SeedDiff"]), 0.025, 0.975))
        if model is not None:
            row_df = pd.DataFrame([row])[feat_cols].fillna(0)
            ml_pred = float(model.predict_proba(row_df)[0, 1])
            final_pred = blend_weight * ml_pred + (1 - blend_weight) * spline_pred
        else:
            final_pred = spline_pred
        predictions.append({"ID": f"{season}_{team1}_{team2}", "Pred": final_pred})
        _log_progress("  Massey_direct predictions", index, total_matchups)

    return _build_submission_from_predictions(season, predictions)


# ── seed_spline_only ──────────────────────────────────────────────────────────

def _fit_seed_spline(tourney_df: pd.DataFrame, seeds_df: pd.DataFrame) -> "UnivariateSpline":
    """Fit a UnivariateSpline on (SeedDiff, Result) pairs from historical tournament games."""
    seed_diffs = []
    results = []
    for _, game in tourney_df.iterrows():
        season = int(game["Season"])
        winner = int(game["WTeamID"])
        loser = int(game["LTeamID"])
        team1, team2 = min(winner, loser), max(winner, loser)
        actual = 1.0 if team1 == winner else 0.0

        s1 = seeds_df[(seeds_df["Season"] == season) & (seeds_df["TeamID"] == team1)]
        s2 = seeds_df[(seeds_df["Season"] == season) & (seeds_df["TeamID"] == team2)]
        if len(s1) == 0 or len(s2) == 0:
            continue
        sd = _seed_number(s1.iloc[0]["Seed"]) - _seed_number(s2.iloc[0]["Seed"])
        seed_diffs.append(sd)
        results.append(actual)

    seed_diffs = np.array(seed_diffs)
    results = np.array(results)
    order = np.argsort(seed_diffs)
    return UnivariateSpline(seed_diffs[order], results[order], s=len(seed_diffs), ext=3)


def _generate_seed_spline_only_submission(data_dir: str, season: int, gender: str) -> pd.DataFrame:
    """Pure seed-spline prediction — zero ML. Works for both genders."""
    print(f"Preparing seed_spline_only submission for {gender} {season}...")
    dm = _get_data_manager(data_dir, season, gender)
    tourney = dm.data["tourney_results"]
    seeds = dm.data["tourney_seeds"]

    train_tourney = tourney[(tourney["Season"] >= 2010) & (tourney["Season"] < season)]
    print(f"Fitting seed spline from {len(train_tourney)} tournament games...")
    seed_spline = _fit_seed_spline(train_tourney, seeds)

    predictions = []
    seeded_teams = _get_current_seeded_teams(dm, season)
    total_matchups = len(seeded_teams) * (len(seeded_teams) - 1) // 2
    print(f"Generating seed_spline_only predictions for {total_matchups} matchups...")
    for index, (team1, team2) in enumerate(_iter_matchups(seeded_teams), start=1):
        s1 = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team1)]
        s2 = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team2)]
        seed_diff = (
            (_seed_number(s1.iloc[0]["Seed"]) - _seed_number(s2.iloc[0]["Seed"]))
            if len(s1) and len(s2) else 0
        )
        pred = float(np.clip(seed_spline(seed_diff), 0.025, 0.975))
        predictions.append({"ID": f"{season}_{team1}_{team2}", "Pred": pred})
        _log_progress("  Seed spline predictions", index, total_matchups)

    return _build_submission_from_predictions(season, predictions)


# ── recency_xgb ──────────────────────────────────────────────────────────────

def _generate_recency_xgb_submission(data_dir: str, season: int, gender: str) -> pd.DataFrame:
    """Identical to baseline_massey but with recency-weighted sample weights (0.75^age)."""
    print(f"Preparing recency_xgb submission for {gender} {season}...")
    dm = _get_data_manager(data_dir, season, gender)
    reg = dm.data["regular_season"]
    tourney = dm.data["tourney_results"]
    seeds = dm.data["tourney_seeds"]

    win_pct = _compute_win_pcts(reg)
    massey_features = _compute_massey_features(dm.data["rankings"]) if dm.rankings_available else None

    train_tourney = tourney[(tourney["Season"] >= 2010) & (tourney["Season"] < season)]
    print(f"Building recency_xgb training features from {len(train_tourney)} tournament games...")
    features = _build_baseline_matchup_features(train_tourney, win_pct, seeds, massey_features)
    feat_cols = [col for col in features.columns if col not in ["Season", "Team1ID", "Team2ID", "Result"]]

    # Recency weights: 0.75^(season - game_season)
    sample_weights = features["Season"].apply(lambda s: 0.75 ** (season - s)).values

    model = _xgb_classifier()
    print(f"Fitting recency_xgb model on shape={features[feat_cols].shape} with sample weights...")
    model.fit(features[feat_cols].fillna(0), features["Result"], sample_weight=sample_weights)

    seed_spline = UnivariateSpline(
        np.sort(features["SeedDiff"].values),
        features.sort_values("SeedDiff")["Result"].values,
        s=len(features),
        ext=3,
    )
    blend = 0.3

    predictions = []
    seeded_teams = _get_current_seeded_teams(dm, season)
    total_matchups = len(seeded_teams) * (len(seeded_teams) - 1) // 2
    print(f"Generating recency_xgb predictions for {total_matchups} matchups...")
    for index, (team1, team2) in enumerate(_iter_matchups(seeded_teams), start=1):
        def get_wp(team_id: int, column: str) -> float:
            row = win_pct[(win_pct["Season"] == season) & (win_pct["TeamID"] == team_id)]
            return float(row[column].values[0]) if len(row) and column in row.columns else 0.5

        def get_seed(team_id: int) -> int:
            row = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team_id)]
            return _seed_number(row.iloc[0]["Seed"]) if len(row) else 8

        feat = {
            "SeedDiff": get_seed(team1) - get_seed(team2),
            "WPct1": get_wp(team1, "Overall"),
            "WPct2": get_wp(team2, "Overall"),
            "WPctH1": get_wp(team1, "H"),
            "WPctH2": get_wp(team2, "H"),
            "WPctA1": get_wp(team1, "A"),
            "WPctA2": get_wp(team2, "A"),
            "WPctN1": get_wp(team1, "N"),
            "WPctN2": get_wp(team2, "N"),
        }
        if massey_features is not None:
            def get_massey(team_id: int) -> float:
                row = massey_features[(massey_features["Season"] == season) & (massey_features["TeamID"] == team_id)]
                return float(row["MasseyRankNorm"].values[0]) if len(row) else 0.5

            feat["Massey1"] = get_massey(team1)
            feat["Massey2"] = get_massey(team2)
            feat["MasseyDiff"] = feat["Massey1"] - feat["Massey2"]

        row_df = pd.DataFrame([feat])[feat_cols].fillna(0)
        model_pred = float(model.predict_proba(row_df)[0, 1])
        spline_pred = float(np.clip(seed_spline(feat["SeedDiff"]), 0.025, 0.975))
        final_pred = blend * model_pred + (1 - blend) * spline_pred
        predictions.append({"ID": f"{season}_{team1}_{team2}", "Pred": final_pred})
        _log_progress("  Recency_xgb predictions", index, total_matchups)

    return _build_submission_from_predictions(season, predictions)


# ── massey_blend ──────────────────────────────────────────────────────────────

def _generate_massey_blend_submission(data_dir: str, season: int, gender: str) -> pd.DataFrame:
    """Average normalized ranks across Massey systems; blend 0.4 massey + 0.6 seed spline.
    Women's fallback: seed_spline_only (no Massey)."""
    print(f"Preparing massey_blend submission for {gender} {season}...")
    dm = _get_data_manager(data_dir, season, gender)
    tourney = dm.data["tourney_results"]
    seeds = dm.data["tourney_seeds"]

    has_massey = dm.rankings_available
    massey_ind = _compute_massey_individual(dm.data["rankings"]) if has_massey else None

    train_tourney = tourney[(tourney["Season"] >= 2010) & (tourney["Season"] < season)]
    print(f"Building massey_blend spline from {len(train_tourney)} tournament games...")
    seed_spline = _fit_seed_spline(train_tourney, seeds)

    sys_cols = [f"MasseyNorm_{s}" for s in _KEY_MASSEY_SYSTEMS]

    def _massey_signal(team_id: int, s: int) -> float | None:
        """Average of available Massey system normalized ranks for team in season s."""
        if massey_ind is None or massey_ind.empty:
            return None
        row = massey_ind[(massey_ind["Season"] == s) & (massey_ind["TeamID"] == team_id)]
        if len(row) == 0:
            return None
        vals = [float(row.iloc[0][col]) for col in sys_cols if col in row.columns and not np.isnan(row.iloc[0][col])]
        return float(np.mean(vals)) if vals else None

    predictions = []
    seeded_teams = _get_current_seeded_teams(dm, season)
    total_matchups = len(seeded_teams) * (len(seeded_teams) - 1) // 2
    print(f"Generating massey_blend predictions for {total_matchups} matchups...")
    for index, (team1, team2) in enumerate(_iter_matchups(seeded_teams), start=1):
        s1 = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team1)]
        s2 = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team2)]
        seed_diff = (
            (_seed_number(s1.iloc[0]["Seed"]) - _seed_number(s2.iloc[0]["Seed"]))
            if len(s1) and len(s2) else 0
        )
        spline_pred = float(np.clip(seed_spline(seed_diff), 0.025, 0.975))

        m1 = _massey_signal(team1, season)
        m2 = _massey_signal(team2, season)

        if m1 is not None and m2 is not None:
            # massey_signal: higher norm rank = better team, so P(team1 wins) ≈ m1 / (m1 + m2)
            massey_pred = float(np.clip(m1 / (m1 + m2 + 1e-9), 0.025, 0.975))
            final_pred = 0.4 * massey_pred + 0.6 * spline_pred
        else:
            final_pred = spline_pred

        predictions.append({"ID": f"{season}_{team1}_{team2}", "Pred": final_pred})
        _log_progress("  Massey_blend predictions", index, total_matchups)

    return _build_submission_from_predictions(season, predictions)


# ── poisson_margin ────────────────────────────────────────────────────────────

def _compute_scoring_efficiency(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute OffEff (avg points scored) and DefEff (avg points allowed) per team per season."""
    off_rows = []
    def_rows = []
    for season, grp in results_df.groupby("Season"):
        # Winners' perspective
        w_off = grp.groupby("WTeamID")["WScore"].mean().reset_index()
        w_off.columns = ["TeamID", "OffEff"]
        w_off["Season"] = season
        w_def = grp.groupby("WTeamID")["LScore"].mean().reset_index()
        w_def.columns = ["TeamID", "DefEff"]
        w_def["Season"] = season

        # Losers' perspective
        l_off = grp.groupby("LTeamID")["LScore"].mean().reset_index()
        l_off.columns = ["TeamID", "OffEff_L"]
        l_off["Season"] = season
        l_def = grp.groupby("LTeamID")["WScore"].mean().reset_index()
        l_def.columns = ["TeamID", "DefEff_L"]
        l_def["Season"] = season

        combined = w_off.merge(w_def, on=["TeamID", "Season"]).merge(
            l_off, on=["TeamID", "Season"], how="outer"
        ).merge(l_def, on=["TeamID", "Season"], how="outer")
        combined["OffEff"] = combined[["OffEff", "OffEff_L"]].mean(axis=1)
        combined["DefEff"] = combined[["DefEff", "DefEff_L"]].mean(axis=1)
        off_rows.append(combined[["Season", "TeamID", "OffEff", "DefEff"]])

    if not off_rows:
        return pd.DataFrame(columns=["Season", "TeamID", "OffEff", "DefEff"])
    return pd.concat(off_rows, ignore_index=True)


def _generate_poisson_margin_submission(data_dir: str, season: int, gender: str) -> pd.DataFrame:
    """Win probability from expected scoring margin; blend 0.35 margin + 0.65 seed spline."""
    print(f"Preparing poisson_margin submission for {gender} {season}...")
    dm = _get_data_manager(data_dir, season, gender)
    reg = dm.data["regular_season"]
    tourney = dm.data["tourney_results"]
    seeds = dm.data["tourney_seeds"]

    # Scoring efficiency from regular season
    efficiency = _compute_scoring_efficiency(reg)

    train_tourney = tourney[(tourney["Season"] >= 2010) & (tourney["Season"] < season)]
    print(f"Building poisson_margin training data from {len(train_tourney)} tournament games...")

    # Build (historical_margin, Result) training rows
    margin_rows = []
    for _, game in train_tourney.iterrows():
        s = int(game["Season"])
        winner = int(game["WTeamID"])
        loser = int(game["LTeamID"])
        team1, team2 = min(winner, loser), max(winner, loser)
        actual = 1.0 if team1 == winner else 0.0

        eff1 = efficiency[(efficiency["Season"] == s) & (efficiency["TeamID"] == team1)]
        eff2 = efficiency[(efficiency["Season"] == s) & (efficiency["TeamID"] == team2)]
        if len(eff1) == 0 or len(eff2) == 0:
            continue

        o1, d1 = float(eff1.iloc[0]["OffEff"]), float(eff1.iloc[0]["DefEff"])
        o2, d2 = float(eff2.iloc[0]["OffEff"]), float(eff2.iloc[0]["DefEff"])
        expected_margin = (o1 - d2) - (o2 - d1)
        margin_rows.append({"ExpectedMargin": expected_margin, "Result": actual})

    if margin_rows:
        margin_df = pd.DataFrame(margin_rows)
        margin_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000)),
        ])
        print(f"Fitting margin LogisticRegression on {len(margin_df)} games...")
        margin_model.fit(margin_df[["ExpectedMargin"]], margin_df["Result"])
    else:
        margin_model = None

    seed_spline = _fit_seed_spline(train_tourney, seeds)

    predictions = []
    seeded_teams = _get_current_seeded_teams(dm, season)
    total_matchups = len(seeded_teams) * (len(seeded_teams) - 1) // 2
    print(f"Generating poisson_margin predictions for {total_matchups} matchups...")
    for index, (team1, team2) in enumerate(_iter_matchups(seeded_teams), start=1):
        s1 = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team1)]
        s2 = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team2)]
        seed_diff = (
            (_seed_number(s1.iloc[0]["Seed"]) - _seed_number(s2.iloc[0]["Seed"]))
            if len(s1) and len(s2) else 0
        )
        spline_pred = float(np.clip(seed_spline(seed_diff), 0.025, 0.975))

        if margin_model is not None:
            eff1 = efficiency[(efficiency["Season"] == season) & (efficiency["TeamID"] == team1)]
            eff2 = efficiency[(efficiency["Season"] == season) & (efficiency["TeamID"] == team2)]
            if len(eff1) and len(eff2):
                o1, d1 = float(eff1.iloc[0]["OffEff"]), float(eff1.iloc[0]["DefEff"])
                o2, d2 = float(eff2.iloc[0]["OffEff"]), float(eff2.iloc[0]["DefEff"])
                expected_margin = (o1 - d2) - (o2 - d1)
                margin_pred = float(margin_model.predict_proba([[expected_margin]])[0, 1])
                final_pred = 0.35 * margin_pred + 0.65 * spline_pred
            else:
                final_pred = spline_pred
        else:
            final_pred = spline_pred

        predictions.append({"ID": f"{season}_{team1}_{team2}", "Pred": final_pred})
        _log_progress("  Poisson_margin predictions", index, total_matchups)

    return _build_submission_from_predictions(season, predictions)


# ── meta_ensemble ─────────────────────────────────────────────────────────────

def _generate_meta_ensemble_submission(data_dir: str, season: int, gender: str) -> pd.DataFrame:
    """Stacked meta-learner: OOF on xgb_ensemble_v2 + massey_direct; blend 0.5 meta + 0.5 spline.
    Women's fallback: degrade to xgb_ensemble_v2 only."""
    print(f"Preparing meta_ensemble submission for {gender} {season}...")
    dm = _get_data_manager(data_dir, season, gender)
    tourney = dm.data["tourney_results"]
    seeds = dm.data["tourney_seeds"]

    train_seasons = sorted(
        s for s in tourney["Season"].unique()
        if 2010 <= s < season
    )

    has_massey = dm.rankings_available and gender == "M"

    # OOF loop: for each training season S, train L0 on seasons < S, predict S
    oof_rows = []  # list of {season, team1, team2, l0_xgb, l0_massey, result}

    print(f"Running OOF meta-ensemble loop over {len(train_seasons)} seasons...")
    for i, target_s in enumerate(train_seasons):
        prior_seasons = [s for s in train_seasons if s < target_s]
        if len(prior_seasons) < 3:
            continue  # not enough history

        # Get L0 predictions for target_s using models trained on prior_seasons
        try:
            xgb_sub = _generate_xgb_ensemble_v2_submission(data_dir, target_s, gender)
            xgb_lookup = dict(zip(xgb_sub["ID"], xgb_sub["Pred"]))
        except Exception as e:
            print(f"  [warn] xgb_ensemble_v2 OOF failed for {target_s}: {e}")
            xgb_lookup = {}

        if has_massey:
            try:
                massey_sub = _generate_massey_direct_submission(data_dir, target_s, gender)
                massey_lookup = dict(zip(massey_sub["ID"], massey_sub["Pred"]))
            except Exception as e:
                print(f"  [warn] massey_direct OOF failed for {target_s}: {e}")
                massey_lookup = {}
        else:
            massey_lookup = {}

        # Get actual results for target_s
        target_games = tourney[tourney["Season"] == target_s]
        for _, game in target_games.iterrows():
            w, l = int(game["WTeamID"]), int(game["LTeamID"])
            t1, t2 = min(w, l), max(w, l)
            actual = 1.0 if t1 == w else 0.0
            gid = f"{target_s}_{t1}_{t2}"
            xgb_pred = xgb_lookup.get(gid, 0.5)
            massey_pred = massey_lookup.get(gid, 0.5) if massey_lookup else xgb_pred
            oof_rows.append({
                "l0_xgb": xgb_pred,
                "l0_massey": massey_pred,
                "Result": actual,
            })

    # Train meta-learner on OOF predictions
    meta_model = None
    if len(oof_rows) >= 20:
        oof_df = pd.DataFrame(oof_rows)
        if has_massey:
            meta_X = oof_df[["l0_xgb", "l0_massey"]]
        else:
            meta_X = oof_df[["l0_xgb"]]
        meta_y = oof_df["Result"]
        meta_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.5, max_iter=1000)),
        ])
        print(f"Fitting meta LogisticRegression on {len(oof_df)} OOF rows...")
        meta_model.fit(meta_X, meta_y)

    # Generate L0 predictions for the target season
    print(f"Generating L0 base predictions for {season}...")
    try:
        xgb_sub = _generate_xgb_ensemble_v2_submission(data_dir, season, gender)
        xgb_lookup = dict(zip(xgb_sub["ID"], xgb_sub["Pred"]))
    except Exception as e:
        print(f"  [warn] xgb_ensemble_v2 for {season} failed: {e}; using 0.5")
        xgb_lookup = {}

    if has_massey:
        try:
            massey_sub = _generate_massey_direct_submission(data_dir, season, gender)
            massey_lookup = dict(zip(massey_sub["ID"], massey_sub["Pred"]))
        except Exception as e:
            print(f"  [warn] massey_direct for {season} failed: {e}; using 0.5")
            massey_lookup = {}
    else:
        massey_lookup = {}

    train_tourney = tourney[(tourney["Season"] >= 2010) & (tourney["Season"] < season)]
    seed_spline = _fit_seed_spline(train_tourney, seeds)

    predictions = []
    seeded_teams = _get_current_seeded_teams(dm, season)
    total_matchups = len(seeded_teams) * (len(seeded_teams) - 1) // 2
    print(f"Generating meta_ensemble predictions for {total_matchups} matchups...")
    for index, (team1, team2) in enumerate(_iter_matchups(seeded_teams), start=1):
        gid = f"{season}_{team1}_{team2}"
        s1 = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team1)]
        s2 = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team2)]
        seed_diff = (
            (_seed_number(s1.iloc[0]["Seed"]) - _seed_number(s2.iloc[0]["Seed"]))
            if len(s1) and len(s2) else 0
        )
        spline_pred = float(np.clip(seed_spline(seed_diff), 0.025, 0.975))

        xgb_pred = xgb_lookup.get(gid, 0.5)
        massey_pred = massey_lookup.get(gid, xgb_pred)

        if meta_model is not None:
            if has_massey:
                meta_input = [[xgb_pred, massey_pred]]
            else:
                meta_input = [[xgb_pred]]
            meta_pred = float(meta_model.predict_proba(meta_input)[0, 1])
            final_pred = 0.5 * meta_pred + 0.5 * spline_pred
        else:
            final_pred = 0.5 * xgb_pred + 0.5 * spline_pred

        predictions.append({"ID": gid, "Pred": final_pred})
        _log_progress("  Meta_ensemble predictions", index, total_matchups)

    return _build_submission_from_predictions(season, predictions)


# ── seed_matchup_calibration ──────────────────────────────────────────────────

def _build_seed_pair_win_rates(tourney_df: pd.DataFrame, seeds_df: pd.DataFrame) -> dict:
    """Compute empirical win rate for each (seed_lo, seed_hi) pair (lower seed = better rank)."""
    counts: dict[tuple, list] = {}
    for _, game in tourney_df.iterrows():
        season = int(game["Season"])
        winner = int(game["WTeamID"])
        loser = int(game["LTeamID"])
        t1, t2 = min(winner, loser), max(winner, loser)
        actual = 1.0 if t1 == winner else 0.0

        s1 = seeds_df[(seeds_df["Season"] == season) & (seeds_df["TeamID"] == t1)]
        s2 = seeds_df[(seeds_df["Season"] == season) & (seeds_df["TeamID"] == t2)]
        if len(s1) == 0 or len(s2) == 0:
            continue
        seed1 = _seed_number(s1.iloc[0]["Seed"])
        seed2 = _seed_number(s2.iloc[0]["Seed"])
        pair = (min(seed1, seed2), max(seed1, seed2))
        # actual is P(lower teamID wins), not P(lower seed wins)
        # We want P(lower-seed team wins); lower-seed team may or may not be team1
        if seed1 < seed2:
            # team1 is the lower seed; actual = P(team1 wins)
            win_for_lower_seed = actual
        elif seed2 < seed1:
            win_for_lower_seed = 1.0 - actual
        else:
            win_for_lower_seed = actual  # same seed
        counts.setdefault(pair, []).append(win_for_lower_seed)

    return {pair: (float(np.mean(vals)), len(vals)) for pair, vals in counts.items()}


def _generate_seed_matchup_calibration_submission(data_dir: str, season: int, gender: str) -> pd.DataFrame:
    """Blend empirical seed-pair win rates (0.35) with xgb_ensemble_v2 (0.65).
    Falls back to seed spline if < 10 historical games for that seed pair."""
    print(f"Preparing seed_matchup_calibration submission for {gender} {season}...")
    dm = _get_data_manager(data_dir, season, gender)
    tourney = dm.data["tourney_results"]
    seeds = dm.data["tourney_seeds"]

    train_tourney = tourney[(tourney["Season"] >= 2010) & (tourney["Season"] < season)]
    print(f"Computing seed pair win rates from {len(train_tourney)} tournament games...")
    pair_rates = _build_seed_pair_win_rates(train_tourney, seeds)
    seed_spline = _fit_seed_spline(train_tourney, seeds)

    # Get xgb_ensemble_v2 predictions for this season
    print(f"Getting xgb_ensemble_v2 base predictions for {season}...")
    try:
        xgb_sub = _generate_xgb_ensemble_v2_submission(data_dir, season, gender)
        xgb_lookup = dict(zip(xgb_sub["ID"], xgb_sub["Pred"]))
    except Exception as e:
        print(f"  [warn] xgb_ensemble_v2 for {season} failed: {e}; falling back to spline")
        xgb_lookup = {}

    predictions = []
    seeded_teams = _get_current_seeded_teams(dm, season)
    total_matchups = len(seeded_teams) * (len(seeded_teams) - 1) // 2
    print(f"Generating seed_matchup_calibration predictions for {total_matchups} matchups...")
    for index, (team1, team2) in enumerate(_iter_matchups(seeded_teams), start=1):
        gid = f"{season}_{team1}_{team2}"
        s1 = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team1)]
        s2 = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team2)]
        if len(s1) == 0 or len(s2) == 0:
            seed1, seed2 = 8, 8
        else:
            seed1 = _seed_number(s1.iloc[0]["Seed"])
            seed2 = _seed_number(s2.iloc[0]["Seed"])

        seed_diff = seed1 - seed2
        spline_pred = float(np.clip(seed_spline(seed_diff), 0.025, 0.975))
        xgb_pred = xgb_lookup.get(gid, spline_pred)

        pair = (min(seed1, seed2), max(seed1, seed2))
        rate_info = pair_rates.get(pair)
        if rate_info is not None and rate_info[1] >= 10:
            historical_rate, _ = rate_info
            # historical_rate = P(lower-seed team wins). team1 might not be lower seed.
            if seed1 <= seed2:
                hist_pred = historical_rate
            else:
                hist_pred = 1.0 - historical_rate
            final_pred = 0.35 * hist_pred + 0.65 * xgb_pred
        else:
            final_pred = xgb_pred  # fall back to xgb_ensemble_v2

        predictions.append({"ID": gid, "Pred": final_pred})
        _log_progress("  Seed calibration predictions", index, total_matchups)

    return _build_submission_from_predictions(season, predictions)


# ── modeh7 (2025 Kaggle 1st-place adaptation) ─────────────────────────────────
# Faithful port of the original notebook. Key design choices:
#   - M and W data combined in one model; men_women flag as a feature
#   - Symmetric dataset doubling with overtime adjustment
#   - Separate T1/T2 avg-stat features (not differences)
#   - ELO: base=1000, width=400, K=100
#   - GLM quality via Ridge on tournament-adjacent teams
#   - LOSO XGBoost regressor on PointDiff
#   - Spline calibration k=5, clipped to [-25, 25]

# Stat columns used in the notebook's box-score feature set
_M7_STAT_COLS = [
    "Score", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
    "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF",
]

_M7_FEATURES = (
    ["men_women", "T1_seed", "T2_seed", "Seed_diff"]
    + [f"T1_avg_{c}" for c in _M7_STAT_COLS]
    + [f"T1_avg_opponent_{c}" for c in _M7_STAT_COLS]
    + ["T1_avg_PointDiff"]
    + [f"T2_avg_{c}" for c in _M7_STAT_COLS]
    + [f"T2_avg_opponent_{c}" for c in _M7_STAT_COLS]
    + ["T2_avg_PointDiff"]
    + ["T1_elo", "T2_elo", "elo_diff"]
    + ["T1_quality", "T2_quality"]
)

_M7_PARAMS = {
    "objective": "reg:squarederror", "booster": "gbtree",
    "eta": 0.0093, "subsample": 0.6, "colsample_bynode": 0.8,
    "num_parallel_tree": 2, "min_child_weight": 4, "max_depth": 4,
    "tree_method": "hist", "grow_policy": "lossguide", "max_bin": 38,
    "verbosity": 0,
}
_M7_NUM_ROUNDS = 704
_M7_SPLINE_CLIP = 25   # clip point-diff before spline


def _m7_load_raw(data_dir: str, season: int) -> tuple:
    """Load M and W raw DataFrames for modeh7. Returns (M_reg_det, M_tourney_det, M_seeds,
    W_reg_det, W_tourney_det, W_seeds). Any missing DataFrame returned as empty."""
    def _safe_load(data_dir, gender):
        try:
            dm = MarchMadnessDataManager(data_dir, gender=gender, current_season=season)
            dm.load_data()
            reg_det   = dm.data.get("regular_season_detailed", pd.DataFrame())
            tour_det  = dm.data.get("tourney_detailed", pd.DataFrame())
            reg_comp  = dm.data.get("regular_season", pd.DataFrame())
            tourney   = dm.data.get("tourney_results", pd.DataFrame())
            seeds_df  = dm.data.get("tourney_seeds", pd.DataFrame())
            # Prefer detailed; fall back to compact
            if reg_det is None or reg_det.empty:
                reg_det = reg_comp
            if tour_det is None or tour_det.empty:
                tour_det = tourney
            return reg_det, tour_det, seeds_df
        except Exception as e:
            print(f"  [warn] Could not load {gender} data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    m_reg, m_tour, m_seeds = _safe_load(data_dir, "M")
    w_reg, w_tour, w_seeds = _safe_load(data_dir, "W")
    return m_reg, m_tour, m_seeds, w_reg, w_tour, w_seeds


def _m7_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Symmetric dataset doubling with overtime adjustment (notebook cell 9)."""
    if df is None or df.empty:
        return pd.DataFrame()

    stat_cols = [c for c in [
        "Score", "FGM", "FGA", "FGM3", "FGA3", "FTM", "FTA",
        "OR", "DR", "Ast", "TO", "Stl", "Blk", "PF",
    ] if f"W{c}" in df.columns]

    keep_cols = (
        ["Season", "DayNum", "NumOT", "WTeamID", "LTeamID"]
        + [f"W{c}" for c in stat_cols]
        + [f"L{c}" for c in stat_cols]
    )
    # deduplicate while preserving order
    seen = set()
    keep_cols = [c for c in keep_cols if c in df.columns and not (c in seen or seen.add(c))]
    df = df[keep_cols].copy().reset_index(drop=True)

    # Overtime adjustment (use numpy to avoid index alignment issues)
    num_ot = df["NumOT"].values if "NumOT" in df.columns else np.zeros(len(df))
    adjot = (40 + 5 * num_ot) / 40
    for base in stat_cols:
        for prefix in ("W", "L"):
            col = f"{prefix}{base}"
            if col in df.columns:
                df[col] = df[col].values / adjot
    if "WScore" in df.columns:
        df["WScore"] = df["WScore"].values / adjot
    if "LScore" in df.columns:
        df["LScore"] = df["LScore"].values / adjot

    # Build T1_ = winner, T2_ = loser
    w_cols = {f"W{c}": f"T1_{c}" for c in stat_cols + ["TeamID"]}
    l_cols = {f"L{c}": f"T2_{c}" for c in stat_cols + ["TeamID"]}
    df_a = df.rename(columns={**w_cols, **l_cols})

    # Swapped: T1_ = loser, T2_ = winner
    w_cols2 = {f"W{c}": f"T2_{c}" for c in stat_cols + ["TeamID"]}
    l_cols2 = {f"L{c}": f"T1_{c}" for c in stat_cols + ["TeamID"]}
    df_b = df.rename(columns={**w_cols2, **l_cols2})

    out = pd.concat([df_a, df_b], ignore_index=True)
    out["PointDiff"] = out["T1_Score"] - out["T2_Score"]
    out["win"] = (out["PointDiff"] > 0).astype(int)
    out["men_women"] = out["T1_TeamID"].apply(lambda t: 1 if str(int(t)).startswith("1") else 0)
    return out


def _m7_season_averages(regular_doubled: pd.DataFrame) -> tuple:
    """Compute per-(Season, T1_TeamID) averages. Returns (ss_T1, ss_T2) DataFrames."""
    stat_cols = [c for c in _M7_STAT_COLS if f"T1_{c}" in regular_doubled.columns]
    t1_stats = [f"T1_{c}" for c in stat_cols]
    t2_stats = [f"T2_{c}" for c in stat_cols]
    box_cols = t1_stats + t2_stats + ["PointDiff"]
    available = [c for c in box_cols if c in regular_doubled.columns]

    ss = (
        regular_doubled.groupby(["Season", "T1_TeamID"])[available]
        .mean()
        .reset_index()
    )

    def _rename(col):
        return col.replace("T1_", "").replace("T2_", "opponent_")

    ss_T1 = ss.copy()
    ss_T1.columns = (
        ["Season", "T1_TeamID"]
        + [f"T1_avg_{_rename(c)}" for c in available]
    )

    ss_T2 = ss.copy()
    ss_T2.columns = (
        ["Season", "T2_TeamID"]
        + [f"T2_avg_{_rename(c)}" for c in available]
    )

    return ss_T1, ss_T2


def _m7_elo(regular_doubled: pd.DataFrame, seeds_df: pd.DataFrame) -> tuple:
    """ELO ratings from regular season wins. base=1000, width=400, K=100.
    Only processes win==1 rows (actual winners as T1).
    Returns (elos_T1, elos_T2) DataFrames."""
    base_elo = 1000.0
    elo_width = 400.0
    k_factor = 100.0

    def expected(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / elo_width))

    seasons = sorted(seeds_df["Season"].unique())
    elo_rows = []

    for s in seasons:
        ss = regular_doubled[(regular_doubled["Season"] == s) & (regular_doubled["win"] == 1)].reset_index(drop=True)
        if ss.empty:
            continue
        teams = set(ss["T1_TeamID"]) | set(ss["T2_TeamID"])
        elo: dict = {t: base_elo for t in teams}
        for _, row in ss.iterrows():
            w, l = row["T1_TeamID"], row["T2_TeamID"]
            ew = expected(elo[w], elo[l])
            elo[w] += k_factor * (1.0 - ew)
            elo[l] -= k_factor * (1.0 - ew)
        for team, rating in elo.items():
            elo_rows.append({"Season": s, "TeamID": team, "elo": rating})

    elos = pd.DataFrame(elo_rows)
    elos_T1 = elos.rename(columns={"TeamID": "T1_TeamID", "elo": "T1_elo"})
    elos_T2 = elos.rename(columns={"TeamID": "T2_TeamID", "elo": "T2_elo"})
    return elos_T1, elos_T2


def _m7_glm_quality(regular_doubled: pd.DataFrame, seeds_df: pd.DataFrame) -> tuple:
    """Bradley-Terry quality via Ridge on tournament-adjacent teams (notebook cell 23).
    Returns (quality_T1, quality_T2) DataFrames."""
    from sklearn.linear_model import Ridge

    # Build set of ST = season/teamid strings for tournament teams
    seeds_df = seeds_df.copy()
    seeds_df["ST"] = seeds_df["Season"].astype(str) + "/" + seeds_df["TeamID"].astype(str)
    st = set(seeds_df["ST"])

    # Add teams that beat a tourney team at least once in regular season
    rd = regular_doubled.copy()
    rd["ST1"] = rd["Season"].astype(str) + "/" + rd["T1_TeamID"].astype(str)
    rd["ST2"] = rd["Season"].astype(str) + "/" + rd["T2_TeamID"].astype(str)
    beaters = set(rd.loc[(rd["PointDiff"] > 0) & (rd["ST2"].isin(st)), "ST1"])
    st = st | beaters

    # Filter dataset to rows involving relevant teams
    dt = rd.loc[rd["ST1"].isin(st) | rd["ST2"].isin(st)].copy()

    quality_rows = []
    for (s, mw), grp in dt.groupby(["Season", "men_women"]):
        teams = sorted(set(grp["T1_TeamID"].tolist() + grp["T2_TeamID"].tolist()))
        if len(teams) < 2 or len(grp) < len(teams):
            continue
        team_idx = {t: i for i, t in enumerate(teams)}
        n_games, n_teams = len(grp), len(teams)
        X = np.zeros((n_games, n_teams), dtype=np.float32)
        y = grp["PointDiff"].values.astype(np.float32)
        for i, (_, row) in enumerate(grp.iterrows()):
            t1, t2 = row["T1_TeamID"], row["T2_TeamID"]
            if t1 in team_idx:
                X[i, team_idx[t1]] = 1.0
            if t2 in team_idx:
                X[i, team_idx[t2]] = -1.0
        try:
            model = Ridge(alpha=0.01, fit_intercept=False)
            model.fit(X, y)
            for t, idx in team_idx.items():
                quality_rows.append({"Season": s, "TeamID": t, "quality": float(model.coef_[idx])})
        except Exception:
            pass

    if not quality_rows:
        empty = pd.DataFrame(columns=["Season", "T1_TeamID", "T1_quality"])
        return empty, empty.rename(columns={"T1_TeamID": "T2_TeamID", "T1_quality": "T2_quality"})

    quality = pd.DataFrame(quality_rows)
    q_T1 = quality.rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"})
    q_T2 = quality.rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"})
    return q_T1, q_T2


def _m7_build_tourney_data(
    tourney_doubled: pd.DataFrame,
    seeds_df: pd.DataFrame,
    ss_T1: pd.DataFrame,
    ss_T2: pd.DataFrame,
    elos_T1: pd.DataFrame,
    elos_T2: pd.DataFrame,
    quality_T1: pd.DataFrame,
    quality_T2: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all features onto tournament data (notebook cells 14-23)."""
    seeds_df = seeds_df.copy()
    seeds_df["seed"] = seeds_df["Seed"].apply(lambda x: int(str(x)[1:3]))
    s_T1 = seeds_df[["Season", "TeamID", "seed"]].rename(
        columns={"TeamID": "T1_TeamID", "seed": "T1_seed"})
    s_T2 = seeds_df[["Season", "TeamID", "seed"]].rename(
        columns={"TeamID": "T2_TeamID", "seed": "T2_seed"})

    td = tourney_doubled[
        ["Season", "T1_TeamID", "T2_TeamID", "PointDiff", "win", "men_women"]
    ].copy()
    # Skip First Four (DayNum < 136)
    if "DayNum" in tourney_doubled.columns:
        td = td[tourney_doubled["DayNum"] >= 136].copy()

    td = td.merge(s_T1, on=["Season", "T1_TeamID"], how="left")
    td = td.merge(s_T2, on=["Season", "T2_TeamID"], how="left")
    td["Seed_diff"] = td["T2_seed"] - td["T1_seed"]

    td = td.merge(ss_T1, on=["Season", "T1_TeamID"], how="left")
    td = td.merge(ss_T2, on=["Season", "T2_TeamID"], how="left")
    td = td.merge(elos_T1, on=["Season", "T1_TeamID"], how="left")
    td = td.merge(elos_T2, on=["Season", "T2_TeamID"], how="left")
    td["elo_diff"] = td["T1_elo"] - td["T2_elo"]
    td = td.merge(quality_T1, on=["Season", "T1_TeamID"], how="left")
    td = td.merge(quality_T2, on=["Season", "T2_TeamID"], how="left")

    # Fill missing feature columns with 0
    for col in _M7_FEATURES:
        if col not in td.columns:
            td[col] = 0.0
    td[_M7_FEATURES] = td[_M7_FEATURES].fillna(0.0)
    return td


def _m7_build_test_rows(
    season: int,
    gender: str,
    seeds_df: pd.DataFrame,
    ss_T1: pd.DataFrame,
    ss_T2: pd.DataFrame,
    elos_T1: pd.DataFrame,
    elos_T2: pd.DataFrame,
    quality_T1: pd.DataFrame,
    quality_T2: pd.DataFrame,
) -> pd.DataFrame:
    """Build test feature rows for all (T1, T2) pairs with T1 < T2 for seeded teams."""
    seeds_season = seeds_df[seeds_df["Season"] == season].copy()
    seeds_season["seed"] = seeds_season["Seed"].apply(lambda x: int(str(x)[1:3]))
    team_seeds = dict(zip(seeds_season["TeamID"], seeds_season["seed"]))
    teams = sorted(team_seeds.keys())
    men_women = 1 if gender == "M" else 0

    rows = []
    for t1, t2 in combinations(teams, 2):
        rows.append({
            "Season": season, "T1_TeamID": t1, "T2_TeamID": t2,
            "men_women": men_women,
            "T1_seed": team_seeds.get(t1, 8),
            "T2_seed": team_seeds.get(t2, 8),
            "Seed_diff": team_seeds.get(t2, 8) - team_seeds.get(t1, 8),
        })
    X = pd.DataFrame(rows)

    X = X.merge(ss_T1, on=["Season", "T1_TeamID"], how="left")
    X = X.merge(ss_T2, on=["Season", "T2_TeamID"], how="left")
    X = X.merge(elos_T1, on=["Season", "T1_TeamID"], how="left")
    X = X.merge(elos_T2, on=["Season", "T2_TeamID"], how="left")
    X["elo_diff"] = X["T1_elo"] - X["T2_elo"]
    X = X.merge(quality_T1, on=["Season", "T1_TeamID"], how="left")
    X = X.merge(quality_T2, on=["Season", "T2_TeamID"], how="left")

    for col in _M7_FEATURES:
        if col not in X.columns:
            X[col] = 0.0
    X[_M7_FEATURES] = X[_M7_FEATURES].fillna(0.0)
    return X


def _compute_modeh7_box_features(detailed_df: pd.DataFrame, compact_df: pd.DataFrame) -> pd.DataFrame:
    """Per-team per-season averages of scoring and efficiency box stats.

    Returns DataFrame with columns: [Season, TeamID, Score, FGA, OR, DR, Blk, PF,
    OppScore, OppFGA, OppOR, OppDR, OppBlk, OppPF].
    Falls back to compact results (score only) when detailed unavailable.
    """
    rows = []

    def _agg_side(df, team_col, opp_col, prefix, opp_prefix, stat_cols, opp_stat_cols):
        """Aggregate stats for one side (W or L) of a detailed results DataFrame."""
        records = []
        for season, grp in df.groupby("Season"):
            for team_id, tgrp in grp.groupby(team_col):
                r = {"Season": season, "TeamID": int(team_id)}
                for col, out in zip(stat_cols, ["Score", "FGA", "OR", "DR", "Blk", "PF"]):
                    r[out] = tgrp[col].mean() if col in tgrp.columns else 0.0
                for col, out in zip(opp_stat_cols, ["OppScore", "OppFGA", "OppOR", "OppDR", "OppBlk", "OppPF"]):
                    r[out] = tgrp[col].mean() if col in tgrp.columns else 0.0
                records.append(r)
        return records

    stat_cols = []
    if detailed_df is not None and not detailed_df.empty:
        w_stat_cols = ["WScore", "WFGA", "WOR", "WDR", "WBlk", "WPF"]
        w_opp_cols  = ["LScore", "LFGA", "LOR", "LDR", "LBlk", "LPF"]
        l_stat_cols = ["LScore", "LFGA", "LOR", "LDR", "LBlk", "LPF"]
        l_opp_cols  = ["WScore", "WFGA", "WOR", "WDR", "WBlk", "WPF"]

        # Only keep columns that actually exist
        w_stat_cols = [c for c in w_stat_cols if c in detailed_df.columns]
        w_opp_cols  = [c for c in w_opp_cols  if c in detailed_df.columns]
        l_stat_cols = [c for c in l_stat_cols if c in detailed_df.columns]
        l_opp_cols  = [c for c in l_opp_cols  if c in detailed_df.columns]

        out_cols = ["Score", "FGA", "OR", "DR", "Blk", "PF"][: len(w_stat_cols)]
        opp_cols_out = ["OppScore", "OppFGA", "OppOR", "OppDR", "OppBlk", "OppPF"][: len(w_opp_cols)]

        for season, grp in detailed_df.groupby("Season"):
            for side, id_col, s_cols, o_cols in [
                ("W", "WTeamID", w_stat_cols, w_opp_cols),
                ("L", "LTeamID", l_stat_cols, l_opp_cols),
            ]:
                for team_id, tgrp in grp.groupby(id_col):
                    r = {"Season": int(season), "TeamID": int(team_id)}
                    for col, out in zip(s_cols, out_cols):
                        r[out] = float(tgrp[col].mean())
                    for col, out in zip(o_cols, opp_cols_out):
                        r[out] = float(tgrp[col].mean())
                    rows.append(r)

        if rows:
            box = (
                pd.DataFrame(rows)
                .groupby(["Season", "TeamID"])
                .mean()
                .reset_index()
            )
            # Ensure all expected columns exist with 0 fallback
            for col in ["Score", "FGA", "OR", "DR", "Blk", "PF", "OppScore", "OppFGA", "OppOR", "OppDR", "OppBlk", "OppPF"]:
                if col not in box.columns:
                    box[col] = 0.0
            return box

    # Fallback: compact results (score only)
    if compact_df is None or compact_df.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "Score", "FGA", "OR", "DR", "Blk", "PF",
                                      "OppScore", "OppFGA", "OppOR", "OppDR", "OppBlk", "OppPF"])

    for side, id_col, score_col, opp_score_col in [
        ("W", "WTeamID", "WScore", "LScore"),
        ("L", "LTeamID", "LScore", "WScore"),
    ]:
        for (season, team_id), grp in compact_df.groupby(["Season", id_col]):
            rows.append({
                "Season": int(season), "TeamID": int(team_id),
                "Score": float(grp[score_col].mean()) if score_col in grp.columns else 0.0,
                "FGA": 0.0, "OR": 0.0, "DR": 0.0, "Blk": 0.0, "PF": 0.0,
                "OppScore": float(grp[opp_score_col].mean()) if opp_score_col in grp.columns else 0.0,
                "OppFGA": 0.0, "OppOR": 0.0, "OppDR": 0.0, "OppBlk": 0.0, "OppPF": 0.0,
            })

    if not rows:
        return pd.DataFrame(columns=["Season", "TeamID", "Score", "FGA", "OR", "DR", "Blk", "PF",
                                      "OppScore", "OppFGA", "OppOR", "OppDR", "OppBlk", "OppPF"])

    return (
        pd.DataFrame(rows)
        .groupby(["Season", "TeamID"])
        .mean()
        .reset_index()
    )


def _compute_modeh7_elo(compact_df: pd.DataFrame) -> dict:
    """Sequential ELO ratings (K=20, start=1500). Returns {(season, team_id): final_elo}."""
    elo: dict = {}
    K = 20

    def _expected(ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    for season, grp in compact_df.sort_values("DayNum").groupby("Season"):
        # Initialize ratings for this season
        season_elo: dict = {}
        for _, row in grp.iterrows():
            w = int(row["WTeamID"])
            l = int(row["LTeamID"])
            rw = season_elo.get(w, 1500.0)
            rl = season_elo.get(l, 1500.0)
            ew = _expected(rw, rl)
            season_elo[w] = rw + K * (1.0 - ew)
            season_elo[l] = rl + K * (0.0 - (1.0 - ew))

        for team_id, rating in season_elo.items():
            elo[(season, team_id)] = rating

    return elo


def _compute_modeh7_glm_quality(compact_df: pd.DataFrame) -> dict:
    """Team quality via Ridge regression on team indicators (log-odds point diff).

    For each season: build indicator matrix, label = point differential, fit Ridge.
    Returns {(season, team_id): float}.
    """
    from sklearn.linear_model import Ridge

    quality: dict = {}

    for season, grp in compact_df.groupby("Season"):
        teams = sorted(set(grp["WTeamID"].tolist() + grp["LTeamID"].tolist()))
        team_idx = {t: i for i, t in enumerate(teams)}
        n_games = len(grp)
        n_teams = len(teams)

        if n_games < 5 or n_teams < 2:
            continue

        X = np.zeros((n_games, n_teams), dtype=np.float32)
        y = np.zeros(n_games, dtype=np.float32)

        for i, (_, row) in enumerate(grp.iterrows()):
            w = int(row["WTeamID"])
            l = int(row["LTeamID"])
            margin = float(row["WScore"]) - float(row["LScore"]) if "WScore" in row and "LScore" in row else 1.0
            X[i, team_idx[w]] = 1.0
            X[i, team_idx[l]] = -1.0
            y[i] = margin

        try:
            model = Ridge(alpha=0.01, fit_intercept=False)
            model.fit(X, y)
            for t, idx in team_idx.items():
                quality[(season, t)] = float(model.coef_[idx])
        except Exception:
            pass

    return quality


def _build_modeh7_matchup_rows(
    tourney_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    box_features: pd.DataFrame,
    elo: dict,
    glm: dict,
    seasons: list,
) -> pd.DataFrame:
    """Build symmetric training dataset (each game appears twice, T1↔T2 swapped).

    Features = T1 stats − T2 stats for: Score, FGA, OR, DR, Blk, PF, OppScore, …,
    Elo, GLM_quality, seed_diff.
    Label = PointDiff (T1_score − T2_score).
    """
    box_idx = {}
    for _, row in box_features.iterrows():
        box_idx[(int(row["Season"]), int(row["TeamID"]))] = row

    seed_idx = {}
    for _, row in seeds_df.iterrows():
        s = str(row["Seed"])
        num = int(s[1:3]) if len(s) >= 3 else 8
        seed_idx[(int(row["Season"]), int(row["TeamID"]))] = num

    stat_cols = ["Score", "FGA", "OR", "DR", "Blk", "PF", "OppScore", "OppFGA", "OppOR", "OppDR", "OppBlk", "OppPF"]
    rows = []

    # Skip First Four games (DayNum < 136 typically, or use WScore==0 trick)
    # First Four games have DayNum <= 135; main tournament starts DayNum 136
    main_games = tourney_df[tourney_df["Season"].isin(seasons)].copy()
    if "DayNum" in main_games.columns:
        main_games = main_games[main_games["DayNum"] >= 136]

    for _, game in main_games.iterrows():
        season = int(game["Season"])
        w = int(game["WTeamID"])
        l = int(game["LTeamID"])
        margin = float(game["WScore"]) - float(game["LScore"]) if "WScore" in game and "LScore" in game else 1.0

        for t1, t2, point_diff, win_label in [(w, l, margin, 1), (l, w, -margin, 0)]:
            feat: dict = {"Season": season, "T1": t1, "T2": t2,
                          "PointDiff": point_diff, "WinLabel": win_label}

            b1 = box_idx.get((season, t1))
            b2 = box_idx.get((season, t2))
            for col in stat_cols:
                v1 = float(b1[col]) if b1 is not None and col in b1 else 0.0
                v2 = float(b2[col]) if b2 is not None and col in b2 else 0.0
                feat[f"diff_{col}"] = v1 - v2

            feat["diff_Elo"] = elo.get((season, t1), 1500.0) - elo.get((season, t2), 1500.0)
            feat["diff_GLM"] = glm.get((season, t1), 0.0) - glm.get((season, t2), 0.0)
            feat["diff_Seed"] = seed_idx.get((season, t1), 8) - seed_idx.get((season, t2), 8)

            rows.append(feat)

    return pd.DataFrame(rows)


def _generate_modeh7_submission(data_dir: str, season: int, gender: str) -> pd.DataFrame:
    """2025 Kaggle 1st-place solution (modeh7) — faithful notebook port.

    Trains on combined M+W data with men_women feature. Uses LOSO XGBoost
    regression on point differential, calibrated with a k=5 spline.
    """
    if xgb is None:
        print("[warn] modeh7: xgboost not available, falling back to seed_spline_only")
        return _generate_seed_spline_only_submission(data_dir, season, gender)

    print(f"Preparing modeh7 submission for {gender} {season}...")

    # 1. Load raw data for both genders
    m_reg, m_tour, m_seeds, w_reg, w_tour, w_seeds = _m7_load_raw(data_dir, season)

    if m_reg.empty and w_reg.empty:
        print("  [warn] No data loaded; falling back to seed_spline_only")
        return _generate_seed_spline_only_submission(data_dir, season, gender)

    # 2. Symmetric doubling with OT adjustment
    print("  Preparing doubled datasets...")
    regular_doubled = pd.concat([_m7_prepare(m_reg), _m7_prepare(w_reg)], ignore_index=True)
    tourney_doubled  = pd.concat([_m7_prepare(m_tour), _m7_prepare(w_tour)], ignore_index=True)
    seeds_all = pd.concat([m_seeds, w_seeds], ignore_index=True)

    if regular_doubled.empty or tourney_doubled.empty:
        print("  [warn] Doubled data empty; falling back to seed_spline_only")
        return _generate_seed_spline_only_submission(data_dir, season, gender)

    # Filter to seasons with data (cut season: 2003 for M, 2010 for W — use 2010)
    regular_doubled = regular_doubled[regular_doubled["Season"] >= 2003]
    tourney_doubled  = tourney_doubled[tourney_doubled["Season"] >= 2003]
    seeds_all        = seeds_all[seeds_all["Season"] >= 2003]

    # 3. Season-average box features
    print("  Computing season averages...")
    ss_T1, ss_T2 = _m7_season_averages(regular_doubled)

    # 4. ELO ratings
    print("  Computing ELO ratings...")
    elos_T1, elos_T2 = _m7_elo(regular_doubled, seeds_all)

    # 5. GLM quality
    print("  Computing GLM quality...")
    quality_T1, quality_T2 = _m7_glm_quality(regular_doubled, seeds_all)

    # 6. Build tourney_data with all features
    print("  Building tournament feature dataset...")
    tourney_data = _m7_build_tourney_data(
        tourney_doubled, seeds_all, ss_T1, ss_T2,
        elos_T1, elos_T2, quality_T1, quality_T2,
    )

    # Only train on seasons with sufficient data, excluding 2020 (COVID) and target season
    train_seasons = sorted([
        s for s in tourney_data["Season"].unique()
        if s != 2020 and s < season
    ])
    if len(train_seasons) < 3:
        print("  [warn] Not enough training seasons; falling back to seed_spline_only")
        return _generate_seed_spline_only_submission(data_dir, season, gender)

    td_train = tourney_data[tourney_data["Season"].isin(train_seasons)].copy()
    print(f"  Training on {len(train_seasons)} seasons ({train_seasons[0]}–{train_seasons[-1]}), "
          f"{len(td_train)} rows (M+W combined)")

    # 7. LOSO loop — matches notebook cell 30 exactly
    oof_preds = []
    oof_targets = []
    loso_models = []

    feat_cols_avail = [c for c in _M7_FEATURES if c in td_train.columns]

    for oof_season in train_seasons:
        x_train = td_train.loc[td_train["Season"] != oof_season, feat_cols_avail].values
        y_train = td_train.loc[td_train["Season"] != oof_season, "PointDiff"].values
        x_val   = td_train.loc[td_train["Season"] == oof_season, feat_cols_avail].values
        y_val   = td_train.loc[td_train["Season"] == oof_season, "PointDiff"].values

        if len(x_train) < 10 or len(x_val) == 0:
            continue

        dtrain = xgb.DMatrix(x_train, label=y_train)
        model = xgb.train(_M7_PARAMS, dtrain, num_boost_round=_M7_NUM_ROUNDS)
        preds = model.predict(xgb.DMatrix(x_val))

        oof_preds.extend(preds.tolist())
        oof_targets.extend(y_val.tolist())
        loso_models.append(model)

    if not oof_preds:
        print("  [warn] No OOF predictions; falling back to seed_spline_only")
        return _generate_seed_spline_only_submission(data_dir, season, gender)

    print(f"  Collected {len(oof_preds)} OOF predictions from {len(loso_models)} models")

    # 8. Spline calibration — matches notebook cell 32 exactly
    t = _M7_SPLINE_CLIP
    dat = sorted(zip(oof_preds, [v > 0 for v in oof_targets]), key=lambda x: x[0])
    pred_sorted, label_sorted = zip(*dat)
    pred_clipped = np.clip(pred_sorted, -t, t)
    spline_model = UnivariateSpline(pred_clipped, label_sorted, k=5)

    # 9. Build test rows for the target gender/season
    seeds_gender = m_seeds if gender == "M" else w_seeds
    if seeds_gender.empty:
        print(f"  [warn] No seeds for {gender} {season}; falling back to seed_spline_only")
        return _generate_seed_spline_only_submission(data_dir, season, gender)

    print("  Building test matchup features...")
    X_test = _m7_build_test_rows(
        season, gender, seeds_gender, ss_T1, ss_T2,
        elos_T1, elos_T2, quality_T1, quality_T2,
    )
    feat_cols_test = [c for c in feat_cols_avail if c in X_test.columns]
    dtest = xgb.DMatrix(X_test[feat_cols_test].values)

    # 10. Average predictions across LOSO models + apply spline
    print(f"  Running {len(loso_models)} LOSO models on {len(X_test)} test matchups...")
    all_margin_preds = np.array([m.predict(dtest) for m in loso_models])
    avg_margins = all_margin_preds.mean(axis=0)
    probs = np.clip(spline_model(np.clip(avg_margins, -t, t)), 0.025, 0.975)

    predictions = [
        {"ID": f"{season}_{int(row.T1_TeamID)}_{int(row.T2_TeamID)}", "Pred": float(p)}
        for row, p in zip(X_test.itertuples(index=False), probs)
    ]
    return _build_submission_from_predictions(season, predictions)


# ── upset_aware_ensemble ──────────────────────────────────────────────────────

def _compute_upset_features(
    reg_detailed: pd.DataFrame | None,
    reg_compact: pd.DataFrame,
    seeds_df: pd.DataFrame,
    massey_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compute upset-predictive features per (Season, TeamID).

    Returns DataFrame with columns:
      - RecentMomentum: last_10_win_pct - full_season_win_pct
      - MasseyVsSeedGap: ActualSeed - MasseyImpliedSeed (positive = underseeded)
      - DefFirstRatio: DefEff / OffEff (higher = more defense-reliant, more consistent)

    Graceful fallback: missing detailed data → DefFirstRatio = 0.
    Massey unavailable → MasseyVsSeedGap = 0.
    """
    records: dict[tuple, dict] = {}

    # ── Detailed stats: ThreePtReliance, DefFirstRatio ────────────────────────
    if reg_detailed is not None and not reg_detailed.empty:
        for _, row in reg_detailed.iterrows():
            season = int(row["Season"])
            w_poss = float(row["WFGA"]) - float(row["WOR"]) + float(row["WTO"]) + 0.44 * float(row["WFTA"])
            l_poss = float(row["LFGA"]) - float(row["LOR"]) + float(row["LTO"]) + 0.44 * float(row["LFTA"])
            poss = (w_poss + l_poss) / 2.0

            for team_id, pts, opp_pts in [
                (int(row["WTeamID"]), float(row["WScore"]), float(row["LScore"])),
                (int(row["LTeamID"]), float(row["LScore"]), float(row["WScore"])),
            ]:
                key = (season, team_id)
                if key not in records:
                    records[key] = {
                        "Season": season, "TeamID": team_id,
                        "Pts": 0.0, "OppPts": 0.0, "Poss": 0.0, "Games": 0,
                    }
                r = records[key]
                r["Pts"] += pts
                r["OppPts"] += opp_pts
                r["Poss"] += poss
                r["Games"] += 1

    # ── Compact stats: RecentMomentum ─────────────────────────────────────────
    momentum_records: dict[tuple, list] = {}
    for _, row in reg_compact.iterrows():
        season, day = int(row["Season"]), int(row["DayNum"])
        for team_id, won in [(int(row["WTeamID"]), True), (int(row["LTeamID"]), False)]:
            key = (season, team_id)
            momentum_records.setdefault(key, []).append((day, won))

    # ── Massey vs Seed gap ────────────────────────────────────────────────────
    massey_gap_map: dict[tuple, float] = {}
    if massey_df is not None and not massey_df.empty:
        late = massey_df[massey_df["RankingDayNum"] <= 128].copy()
        avg_rank = (
            late.groupby(["Season", "TeamID"])["OrdinalRank"]
            .mean()
            .reset_index(name="MasseyRankAvg")
        )
        avg_rank["MasseyRankNorm"] = avg_rank.groupby("Season")["MasseyRankAvg"].transform(
            lambda x: 1 - (x - x.min()) / (x.max() - x.min() + 1e-9)
        )
        avg_rank["MasseyImpliedSeed"] = (1 - avg_rank["MasseyRankNorm"]) * 15 + 1

        for _, mrow in avg_rank.iterrows():
            s = int(mrow["Season"])
            t = int(mrow["TeamID"])
            seed_rows = seeds_df[(seeds_df["Season"] == s) & (seeds_df["TeamID"] == t)]
            if len(seed_rows) > 0:
                actual_seed = _seed_number(seed_rows.iloc[0]["Seed"])
                massey_gap_map[(s, t)] = actual_seed - float(mrow["MasseyImpliedSeed"])

    # ── Assemble output ───────────────────────────────────────────────────────
    all_keys = set(records.keys()) | set(momentum_records.keys())
    rows = []
    for key in all_keys:
        season, team_id = key
        r = records.get(key, {})

        # DefFirstRatio
        poss = r.get("Poss", 0.0)
        games = r.get("Games", 0)
        if poss > 0 and games > 0:
            off_eff = (r.get("Pts", 0.0) / poss) * 100
            def_eff = (r.get("OppPts", 0.0) / poss) * 100
            def_first_ratio = def_eff / off_eff if off_eff > 0 else 1.0
        else:
            def_first_ratio = 1.0

        # RecentMomentum
        games_list = sorted(momentum_records.get(key, []), key=lambda x: x[0])
        if games_list:
            total = len(games_list)
            full_win_pct = sum(1 for _, w in games_list if w) / total
            last10 = games_list[-10:]
            last10_win_pct = sum(1 for _, w in last10 if w) / len(last10)
            momentum = last10_win_pct - full_win_pct
        else:
            momentum = 0.0

        # MasseyVsSeedGap
        massey_gap = massey_gap_map.get(key, 0.0)

        rows.append({
            "Season": season,
            "TeamID": team_id,
            "DefFirstRatio": def_first_ratio,
            "RecentMomentum": momentum,
            "MasseyVsSeedGap": massey_gap,
        })

    return pd.DataFrame(rows)


def _generate_upset_aware_ensemble_submission(data_dir: str, season: int, gender: str) -> pd.DataFrame:
    """xgb_ensemble_v2 + upset-specific features + aggressive ensemble weighting.

    Key changes vs xgb_ensemble_v2:
      - Adds DefFirstRatio, RecentMomentum, MasseyVsSeedGap diffs (per-team features)
      - Adds HistMatchupRate_Diff (matchup-level historical seed-pair upset rate)
      - ThreePtReliance removed (r=0.06 with upsets — essentially noise)
      - Men's blend: 70% ML + 30% spline — strongly favors upset-aware ensemble
      - Women's blend: 30% ML + 70% spline (unchanged — fewer features available)
    """
    print(f"Preparing upset_aware_ensemble submission for {gender} {season}...")
    dm = _get_data_manager(data_dir, season, gender)
    reg = dm.data["regular_season"]
    detailed = dm.data.get("regular_season_detailed")
    tourney = dm.data["tourney_results"]
    seeds = dm.data["tourney_seeds"]

    # Build base v2 features
    compact_feats = _compute_compact_features(reg)
    detailed_feats = _compute_detailed_features(detailed)
    massey_feats = _compute_massey_multi(dm.data["rankings"]) if dm.rankings_available else None
    massey_individual = _compute_massey_individual(dm.data["rankings"]) if dm.rankings_available else None
    conf_feats = _compute_conf_tourney_features(dm.data.get("conf_tourney"))
    if conf_feats.empty:
        conf_feats = None

    # Build upset-specific features
    massey_raw = dm.data["rankings"] if dm.rankings_available else None
    upset_feats = _compute_upset_features(detailed, reg, seeds, massey_raw)

    # Merge into combined per-team features
    def _merge_all(base: pd.DataFrame) -> pd.DataFrame:
        df = base.copy()
        for extra in [detailed_feats, massey_feats, massey_individual, conf_feats, upset_feats]:
            if extra is not None and not extra.empty:
                df = df.merge(extra, on=["Season", "TeamID"], how="left")
        return df

    all_feats_raw = _merge_all(compact_feats)

    train_tourney = tourney[(tourney["Season"] >= 2010) & (tourney["Season"] < season)]
    print(f"Building upset_aware training features from {len(train_tourney)} tournament games...")

    # Build historical seed-pair upset rates for HistMatchupRate feature
    seed_pair_games: dict[tuple, int] = {}
    seed_pair_upsets: dict[tuple, int] = {}
    for _, g in train_tourney.iterrows():
        s = int(g["Season"])
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        ws_rows = seeds[(seeds["Season"] == s) & (seeds["TeamID"] == w)]
        ls_rows = seeds[(seeds["Season"] == s) & (seeds["TeamID"] == l)]
        if ws_rows.empty or ls_rows.empty:
            continue
        wseed = _seed_number(ws_rows.iloc[0]["Seed"])
        lseed = _seed_number(ls_rows.iloc[0]["Seed"])
        lo, hi = min(wseed, lseed), max(wseed, lseed)
        pair = (lo, hi)
        seed_pair_games[pair] = seed_pair_games.get(pair, 0) + 1
        if wseed > lseed:  # higher seed number won = upset
            seed_pair_upsets[pair] = seed_pair_upsets.get(pair, 0) + 1
    seed_pair_upset_rate: dict[tuple, float] = {
        pair: seed_pair_upsets.get(pair, 0) / cnt
        for pair, cnt in seed_pair_games.items() if cnt >= 3
    }

    # Build matchup rows (reuse v2 builder with all_feats already merged)
    feat_cols = [col for col in all_feats_raw.columns if col not in ["Season", "TeamID"]]
    rows = []
    for _, game in train_tourney.iterrows():
        s = int(game["Season"])
        winner = int(game["WTeamID"])
        loser = int(game["LTeamID"])
        team1, team2 = min(winner, loser), max(winner, loser)
        actual = 1.0 if team1 == winner else 0.0

        feat1 = all_feats_raw[(all_feats_raw["Season"] == s) & (all_feats_raw["TeamID"] == team1)]
        feat2 = all_feats_raw[(all_feats_raw["Season"] == s) & (all_feats_raw["TeamID"] == team2)]
        if len(feat1) == 0 or len(feat2) == 0:
            continue

        seed1_rows = seeds[(seeds["Season"] == s) & (seeds["TeamID"] == team1)]
        seed2_rows = seeds[(seeds["Season"] == s) & (seeds["TeamID"] == team2)]
        if len(seed1_rows) == 0 or len(seed2_rows) == 0:
            continue

        feat1 = feat1.iloc[0]
        feat2 = feat2.iloc[0]
        s1_num = _seed_number(seed1_rows.iloc[0]["Seed"])
        s2_num = _seed_number(seed2_rows.iloc[0]["Seed"])
        lo, hi = min(s1_num, s2_num), max(s1_num, s2_num)
        pair = (lo, hi)
        upset_rate = seed_pair_upset_rate.get(pair, 0.5)
        if s1_num > s2_num:
            hist_diff = upset_rate - (1.0 - upset_rate)
        else:
            hist_diff = (1.0 - upset_rate) - upset_rate
        row = {
            "Season": s,
            "Team1ID": team1,
            "Team2ID": team2,
            "Result": actual,
            "SeedDiff": s1_num - s2_num,
            "HistMatchupRate_Diff": hist_diff,
        }
        for col in feat_cols:
            v1 = feat1.get(col, 0)
            v2 = feat2.get(col, 0)
            row[f"{col}_1"] = v1
            row[f"{col}_2"] = v2
            row[f"{col}_Diff"] = v1 - v2
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    model_feat_cols = [col for col in feat_df.columns if col not in ["Season", "Team1ID", "Team2ID", "Result"]]
    X = feat_df[model_feat_cols].fillna(0)
    y = feat_df["Result"]

    trained_models = []
    for name, clf in _ensemble_models():
        print(f"Fitting upset_aware component {name} on shape={X.shape}...")
        clf.fit(X, y)
        trained_models.append((name, clf))
        print(f"  Component {name} fit complete.")

    seed_spline = UnivariateSpline(
        np.sort(feat_df["SeedDiff"].values),
        feat_df.sort_values("SeedDiff")["Result"].values,
        s=len(feat_df),
        ext=3,
    )
    # Increased ensemble weight for men — lets upset features dominate over chalk spline
    blend_weight = 0.3 if gender == "W" else 0.70

    predictions = []
    current_feats = all_feats_raw[all_feats_raw["Season"] == season].set_index("TeamID")
    seeded_teams = _get_current_seeded_teams(dm, season)
    total_matchups = len(seeded_teams) * (len(seeded_teams) - 1) // 2
    print(f"Generating upset_aware_ensemble predictions for {total_matchups} matchups...")
    for index, (team1, team2) in enumerate(_iter_matchups(seeded_teams), start=1):
        seed1_rows = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team1)]
        seed2_rows = seeds[(seeds["Season"] == season) & (seeds["TeamID"] == team2)]
        if len(seed1_rows) == 0 or len(seed2_rows) == 0:
            continue
        s1_num = _seed_number(seed1_rows.iloc[0]["Seed"])
        s2_num = _seed_number(seed2_rows.iloc[0]["Seed"])
        lo, hi = min(s1_num, s2_num), max(s1_num, s2_num)
        pair = (lo, hi)
        upset_rate = seed_pair_upset_rate.get(pair, 0.5)
        if s1_num > s2_num:
            hist_diff = upset_rate - (1.0 - upset_rate)
        else:
            hist_diff = (1.0 - upset_rate) - upset_rate
        row = {
            "SeedDiff": s1_num - s2_num,
            "HistMatchupRate_Diff": hist_diff,
        }
        feat1 = current_feats.loc[team1] if team1 in current_feats.index else pd.Series(dtype=float)
        feat2 = current_feats.loc[team2] if team2 in current_feats.index else pd.Series(dtype=float)
        for col in feat_cols:
            v1 = feat1.get(col, 0)
            v2 = feat2.get(col, 0)
            row[f"{col}_1"] = v1
            row[f"{col}_2"] = v2
            row[f"{col}_Diff"] = v1 - v2

        row_df = pd.DataFrame([row])[model_feat_cols].fillna(0)
        preds = [clf.predict_proba(row_df)[0, 1] for _, clf in trained_models]
        ensemble_pred = float(np.mean(preds))
        spline_pred = float(np.clip(seed_spline(row["SeedDiff"]), 0.025, 0.975))
        final_pred = blend_weight * ensemble_pred + (1 - blend_weight) * spline_pred
        predictions.append({"ID": f"{season}_{team1}_{team2}", "Pred": final_pred})
        _log_progress("  Upset_aware predictions", index, total_matchups)

    return _build_submission_from_predictions(season, predictions)
