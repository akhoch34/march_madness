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
