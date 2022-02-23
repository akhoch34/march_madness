import sys,os
from itertools import combinations

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

folder_name="march_madness"
BASE_DIR=os.path.abspath(".").split(folder_name)[0]+folder_name
DATA_ROOT=os.path.join(BASE_DIR,"data")
#todo setup shared utilities folder
sys.path.insert(0, DATA_ROOT)
import pandas as pd
import numpy as np
import featuretools as ft

# Load files and minimal processing
regular_season_results_df = pd.read_csv(DATA_ROOT + "/DataFiles/RegularSeasonDetailedResults.csv")
regular_season_results_df['game_type'] = 'regular_season'
tourney_results = pd.read_csv(DATA_ROOT + "/DataFiles/NCAATourneyDetailedResults.csv")
tourney_results['game_type'] = 'march_madness'
#season results is from 2003-2018
season_results = pd.concat([regular_season_results_df, tourney_results]).reset_index(drop=True)
seasons = pd.read_csv(DATA_ROOT + "/DataFiles/Seasons.csv").set_index("Season", drop=False)
seasons['DayZero'] = pd.to_datetime(seasons['DayZero'])
teams = pd.read_csv(DATA_ROOT + "/DataFiles/Teams.csv")
team_conferences = pd.read_csv(DATA_ROOT + "/DataFiles/TeamConferences.csv")
tourney_seeds = pd.read_csv(DATA_ROOT + "/DataFiles/NCAATourneySeeds.csv")

def feature_importances(model, features, n=10):
    #Allows features to be pandas columns or features_encoded
    #NOTE: I did this because I found it difficult to modify
    importances = model.feature_importances_
    zipped = sorted(zip(features, importances), key=lambda x: -x[1])
    for i, f in enumerate(zipped[:n]):
        if type(f[0])!=str:
            print("%d: Feature: %s, %.3f" % (i+1, f[0].get_name(), f[1]))
        else:
            print("%d: Feature: %s, %.3f" % (i+1, f[0], f[1]))

    return [f[0] for f in zipped[:n]]

def build_tourney_seeds():
    # process seeds into usable columns
    import re
    def unpack_seed(seed):
        # 16 seeds per region plus 2 play-in teams 16a and 16b
        d = {}
        d['region'] = seed[0]
        d['seed'] = int(re.findall("[0-9]+", seed)[0])
        # below doesn't identify 65 vs 66 but seeds past 64 will be ignored in Kaggle comp
        if 'a' in seed:
            d['seed'] += 1
        if 'b' in seed:
            d['seed'] += 2
        return pd.Series(d)

    tourney_seeds = pd.read_csv(DATA_ROOT + "/DataFiles/NCAATourneySeeds.csv")
    seeds_data = tourney_seeds['Seed'].apply(unpack_seed)
    # separate out so code is re-runnable
    if 'seed' not in tourney_seeds:
        tourney_seeds = tourney_seeds.join(seeds_data)
    del tourney_seeds['Seed']
    tourney_seeds=tourney_seeds.rename(columns={"Season": "season"})
    tourney_seeds['team_season_id']=tourney_seeds['season'].astype(str)+"_"+tourney_seeds['TeamID'].astype(str)

    tourney_seeds_full = tourney_seeds.merge(seasons.reset_index(drop=True).rename(columns={"Season": "season"}))

    def getRegion(row):

        return row['Region' + row['region']]

    tourney_seeds_full['determined_region'] = tourney_seeds_full.apply(getRegion, axis=1)
    tourney_seeds_full = tourney_seeds_full[['season', 'TeamID', 'region', 'seed', 'team_season_id',
                                             'determined_region']]
    return tourney_seeds_full

def build_possible_matchups_df():
    #generate all possible matchups in it's own entity
    #This well help us generate sepcific head to head features using a many to one relationship
    from itertools import combinations
    possible_matchups = []
    for c in combinations(sorted(teams['TeamID']), 2):
        possible_matchups.append(c)
    possible_matchups_df = pd.DataFrame(possible_matchups, columns=["team_1_id", "team_2_id"]).astype(str)
    possible_matchups_df['matchup_id'] = possible_matchups_df['team_1_id'] + "_" + possible_matchups_df['team_2_id']
    return possible_matchups_df

def build_team_game_log():
    #Generates two rows for each matchup, both from a different team point of view
    #This base lets us calculate team_game specific features and head to head metrics
    import datetime
    def get_game_index(row):
        team_1_id, team_2_id = sorted([row['WTeamID'], row['LTeamID']])
        return "%s_%s_%s_%s" % (row['Season'], row['DayNum'], team_1_id, team_2_id)

    def get_game_daytime(row):
        # DayNum is days since season start
        game_day = seasons.loc[row['Season']]['DayZero'] + datetime.timedelta(days=row['DayNum'])
        return game_day

    season_results['game_index'] = season_results.apply(get_game_index, axis=1)
    season_results['game_datetime'] = pd.to_datetime(season_results.apply(get_game_daytime, axis=1))

    def get_team_game_log(winner_pov=True):
        #generate POV based data
        team_game_log = season_results.copy()
        prefixes = ["team_", "opp_"]
        if not winner_pov:
            prefixes = ["opp_", "team_"]
        rename_dict = {}
        for col in season_results.columns:
            if col == "WLoc":
                continue
            elif col.startswith("W"):
                rename_dict[col] = prefixes[0] + col[1:]
            elif col.startswith("L") and col not in ["Loc"]:
                rename_dict[col] = prefixes[1] + col[1:]
        team_game_log = team_game_log.rename(columns=rename_dict)
        return team_game_log

    winner_games = get_team_game_log(winner_pov=True)
    loser_games = get_team_game_log(winner_pov=False)
    team_game_log = pd.concat([winner_games, loser_games], sort=False)
    team_game_log = team_game_log.sort_values(by=["Season", "DayNum", "game_index"])
    team_game_log['team_margin'] = team_game_log['team_Score'] - team_game_log['opp_Score']
    team_game_log['opp_margin'] = -team_game_log['team_margin']
    team_game_log['team_wins'] = team_game_log['team_margin'] > 0
    team_game_log['team_wins_int'] = team_game_log['team_wins'].astype(int)
    team_game_log['opp_wins'] = ~team_game_log['team_wins']
    team_game_log['opp_wins_int'] = team_game_log['opp_wins'].astype(int)
    # teams can't tie in basketball, above code assumes that by using the ~
    #test assumption
    assert (len(team_game_log[team_game_log['team_margin'] == 0]) == 0)

    def get_game_loc(row):
        if row['team_wins']:
            return row['WLoc']
        else:
            if row['WLoc'] == 'H':
                return 'A'
            elif row['WLoc'] == 'A':
                return 'H'
            else:
                return row['WLoc']

    def get_matchup_id(row):
        # sort id's so we have a unique and easily determined hash from either team's pov
        ids = sorted([row['team_TeamID'], row['opp_TeamID']])
        return "%s_%s" % tuple(ids)

    team_game_log['team_game_loc'] = team_game_log.apply(get_game_loc, axis=1)
    del team_game_log['WLoc']
    team_game_log['matchup_id'] = team_game_log.apply(get_matchup_id, axis=1)

    team_game_log['team_season_id'] = team_game_log['Season'].astype(str) + "_" + team_game_log['team_TeamID'].astype(
        str)
    team_game_log['opp_season_id'] = team_game_log['Season'].astype(str) + "_" + team_game_log['opp_TeamID'].astype(str)
    team_game_log['team_game_index'] = team_game_log['game_index'] + "_" + team_game_log['team_TeamID'].astype(str)
    team_game_log['opp_game_index'] = team_game_log['game_index'] + "_" + team_game_log['opp_TeamID'].astype(str)

    #use game_end to help us with a secondary time index that handles rows containing data at different times
    team_game_log['game_end'] = team_game_log['game_datetime'] + pd.Timedelta("1 day")

    team_game_log = team_game_log.rename(columns={"Season": "season"})
    team_game_log = team_game_log.sort_values(by=["game_datetime","team_game_index"])

    team_game_log['game_PF'] = team_game_log['team_PF'] + team_game_log['opp_PF']
    team_game_log['game_MOV'] = abs(team_game_log['team_margin'])
    attempt_fields = [
        ("FGM3", "FGP3"),
        ("FTM", "FTP"),
        ("FGM", "FGP"),
        ("FGM2", "FGP2")
    ]
    for prefix in ["opp_", "team_"]:
        team_game_log[prefix+"poss"]=team_game_log[prefix+'FGA']-team_game_log[prefix + 'OR']+team_game_log[prefix+"TO"]+.475*team_game_log[prefix+"FTA"]
        team_game_log[prefix+'FGA2']=team_game_log[prefix+'FGA']-team_game_log[prefix+'FGA3']
        team_game_log[prefix + 'FGM2'] = team_game_log[prefix + 'FGM'] - team_game_log[prefix + 'FGM3']

        team_game_log[prefix+'PFP']=team_game_log[prefix+'PF'] / team_game_log['game_PF']
        team_game_log[prefix+'TSA']=team_game_log[prefix+'FGA']+0.44*team_game_log[prefix+'FTA']
        team_game_log[prefix+'TS']=team_game_log[prefix+'Score']/(2*team_game_log[prefix+'TSA'])

        team_game_log[prefix + 'FG2_points'] = 2*team_game_log[prefix + 'FGM2']
        team_game_log[prefix + 'FG3_points'] = 3*team_game_log[prefix + 'FGM3']
        team_game_log[prefix + 'FG_points'] = team_game_log[prefix + 'FG2_points']+team_game_log[prefix + 'FG3_points']
        for col in ["FG2_points","FG3_points", "FG_points"]:
            team_game_log[prefix + col+"_share_FG"]=team_game_log[prefix + col]/team_game_log[prefix + "FG_points"]
            team_game_log[prefix + col + "_share_points"] = team_game_log[prefix + col] / team_game_log[
                prefix + "Score"]
        team_game_log[prefix+"FG_points/FT_points"]=team_game_log[prefix+"FG_points"]/team_game_log[prefix+"FTM"]
        team_game_log[prefix+"FT_point_share"]=team_game_log[prefix+"FTM"]/team_game_log[prefix + "Score"]
        team_game_log[prefix + 'eFGP'] = (team_game_log[prefix + 'FGM'] + .5 * team_game_log[prefix + 'FGM3'])\
                                         /team_game_log[prefix+'FGA']

        team_game_log[prefix+'scoring_eff']=team_game_log[prefix+'Score']/(team_game_log[prefix+'FGA']+.475*team_game_log[prefix+'FTA'])
        team_game_log[prefix+'score_op']=(team_game_log[prefix+'FGA']+.475*team_game_log[prefix+'FTA'])/team_game_log[prefix+"poss"]
        team_game_log[prefix+'off_rtg']=team_game_log[prefix+'Score']/team_game_log[prefix+'poss']*100
        team_game_log[prefix+'IE_numerator']=team_game_log[prefix+'Score']+team_game_log[prefix+'FGM']+team_game_log[prefix+'FTM']\
                                      -team_game_log[prefix+'FGA']-team_game_log[prefix+'FTA']+team_game_log[prefix+'DR']\
                                    +.5*team_game_log[prefix+'OR']+team_game_log[prefix+'Ast']+team_game_log[prefix+'Stl']\
                                    +.5*team_game_log[prefix+'Blk']-team_game_log[prefix+'PF']-team_game_log[prefix+'TO']


        for f, new_col in attempt_fields:
            col1 = prefix + f
            col2 = prefix + f.replace("M", "A")
            team_game_log[prefix + new_col] = team_game_log[col1] / team_game_log[col2]

    #for defensive
    for prefix,other_prefix in [("opp_","team_"), ("team_","opp_")]:
        team_game_log[prefix + 'def_rtg']=team_game_log[other_prefix+'off_rtg']
        team_game_log[prefix+'sos']=team_game_log[prefix+'off_rtg']-team_game_log[prefix+'def_rtg']

        #Add Four Factors https://www.basketball-reference.com/about/factors.html
        team_game_log[prefix+'to_poss']=team_game_log[prefix+'TO']/team_game_log[prefix+'poss']
        team_game_log[prefix+'orb_pct']=team_game_log[prefix+'OR']/(team_game_log[prefix+'OR']+team_game_log[other_prefix+'DR'])
        team_game_log[prefix + 'drb_pct'] = team_game_log[prefix + 'DR'] / (team_game_log[prefix + 'DR'] + team_game_log[other_prefix + 'OR'])
        team_game_log[prefix+'reb_pct']=(team_game_log[prefix+'orb_pct']+team_game_log[prefix+'drb_pct'])/2
        team_game_log[prefix+'IE']=team_game_log[prefix+'IE_numerator']/(team_game_log[prefix+'IE_numerator']+team_game_log[other_prefix+'IE_numerator'])
        team_game_log[prefix+'ast_rtio']=team_game_log[prefix+'Ast']/(team_game_log[prefix+'FGA']+.475*team_game_log[prefix+'FTA']+team_game_log[prefix+'TO']+team_game_log[prefix+'Ast'])*100
        team_game_log[prefix + 'Blk_pct']=team_game_log[prefix+'Blk']/team_game_log[other_prefix+'FGA2']*100
        team_game_log[prefix + 'Stl_pct'] = team_game_log[prefix + 'Stl'] / team_game_log[other_prefix + 'poss'] * 100

    tourney_seeds = build_tourney_seeds()
    merge_cols = ['season_id', 'region', 'seed']
    tourney_seeds = tourney_seeds.rename(columns={"team_season_id": "season_id"})
    team_game_log=team_game_log.merge(
        tourney_seeds[merge_cols].add_prefix("team_"),on="team_season_id", how="left").merge(
        tourney_seeds[merge_cols].add_prefix("opp_"), on="opp_season_id", how="left")
    team_game_log['team_is_higher_seed'] = team_game_log['team_seed'] > team_game_log['opp_seed']
    return team_game_log

def build_matchup_log(team_game_log):
    # we want team_1 to always be the same team based on matchup_id so the stats are computed for the same team
    matchup_log = team_game_log.sort_values(by="team_game_index").groupby("game_index", as_index=False).first()
    rename_dict = {}
    for c in matchup_log.columns:
        rename_dict[c] = c.replace("team_", "team_1_").replace("opp_", "team_2_")
    matchup_log = matchup_log.rename(columns=rename_dict)
    matchup_log = matchup_log[sorted(matchup_log.columns)].sort_values(by=["game_datetime"])

    # filter out irrelevant indices and duplicate information
    # example
    #   (team_1_margin and team_2_margin are redundant in the case of head to head matchups)
    #       -- in team_game_logs this won't be the case
    matchup_log.drop(["team_2_wins","team_2_margin"],axis=1, inplace=True)

    return matchup_log

def load_matchups_entityset(team_game_log=None):
    if team_game_log is None:
        team_game_log = build_team_game_log()

    possible_matchups_df = build_possible_matchups_df()
    matchup_log = build_matchup_log(team_game_log)


    matchups_es = ft.EntitySet()
    matchups_es = matchups_es.entity_from_dataframe(entity_id="possible_matchups",
                                                    dataframe=possible_matchups_df,
                                                    index="matchup_id"
                                                    )
    primary_cols=['game_index', 'DayNum', 'game_datetime', 'game_end', 'game_type',
       'matchup_id', 'season', 'team_1_game_index', 'team_1_game_loc',
       'team_1_is_higher_seed', 'team_1_region', 'team_1_season_id',
       'team_1_seed', 'team_2_game_index', 'team_2_region', 'team_2_season_id',
       'team_2_seed', 'team_1_TeamID', 'team_2_TeamID']
    secondary_cols=[col for col in matchup_log.columns if col not in primary_cols]
    matchups_es = matchups_es.entity_from_dataframe(entity_id="matchup_log",
                                                    dataframe=matchup_log,
                                                    index="game_index",
                                                    time_index="game_datetime",
                                                    variable_types={
                                                        "team_1_TeamID": ft.variable_types.Categorical,
                                                        "team_2_TeamID": ft.variable_types.Categorical
                                                    },
                                                    secondary_time_index={
                                                        "game_end": secondary_cols
                                                            }
                                                    )
    matchups_es.add_relationship(ft.Relationship(
        matchups_es['possible_matchups']['matchup_id'],
        matchups_es['matchup_log']['matchup_id']

         )
    )
    #I'm a little confused by how the automatic add_last_time_indexes works and some of the more advanced time features
    #I did this because it ran faster and is more explicit than the automatic behavior
    matchups_es['matchup_log'].last_time_index = matchups_es['matchup_log'].df['game_end']
    return matchups_es

def load_team_game_logs_entityset(team_game_log=None):
    if team_game_log is None:
        team_game_log=build_team_game_log()
    team_game_logs_es = ft.EntitySet()
    team_game_logs_es = team_game_logs_es.entity_from_dataframe(entity_id="team",
                                                                dataframe=teams,
                                                                index="TeamID"
                                                                )

    primary_cols=['team_game_index', 'DayNum', 'game_type', 'game_index', 'game_datetime',
     'team_game_loc', 'matchup_id', 'team_season_id', 'opp_season_id',
     'opp_game_index', 'game_end', 'team_region', 'team_seed', 'opp_region',
     'opp_seed', 'team_is_higher_seed', 'season', 'team_TeamID',
     'opp_TeamID']
    secondary_cols = [col for col in team_game_log.columns if col not in primary_cols]
    team_game_logs_es = team_game_logs_es.entity_from_dataframe(entity_id="team_game_log",
                                                                dataframe=team_game_log,
                                                                index="team_game_index",
                                                                time_index="game_datetime",
                                                                variable_types={
                                                                    "season": ft.variable_types.Categorical,
                                                                    "team_TeamID": ft.variable_types.Categorical,
                                                                    "opp_TeamID": ft.variable_types.Categorical,
                                                                    'NumOT': ft.variable_types.Categorical
                                                                },
                                                                secondary_time_index={
                                                                    "game_end": secondary_cols

                                                                }
                                                                )
    team_game_logs_es.add_relationship(ft.Relationship(
        team_game_logs_es['team']['TeamID'],
        team_game_logs_es['team_game_log']['team_TeamID']

    )
    )
    team_game_logs_es['team_game_log'].last_time_index = team_game_logs_es['team_game_log'].df['game_end']
    return team_game_logs_es

def make_tourney_labels(es, tourney_starts, only_matchups_with_history=True):
    '''
    Generates labels using tourney start information
    All labels are using game_datetime,
    since this is a single elimination tourney there shouldn't be any leakage for matchup based features
    :param only_matchups_with_history: restricts prediction data to those games with available head to head history
    '''

    #matchup_log = matchups_es["matchup_log"].df
    if "matchup_log" in es.entity_dict.keys():
        matchup_log=es['matchup_log'].df
        label_cols=['team_1_wins']
        index_cols=["game_index"]
    elif "team_game_log" in es.entity_dict.keys():
        matchup_log=es['team_game_log'].df
        label_cols=['team_wins']
        index_cols=["team_game_index"]
    full_training_data = pd.DataFrame()
    valid_pred_data = pd.DataFrame()
    # Cumulative appends so we can compute the history flag appropriately
    for season, tourney_start in tourney_starts.iteritems():
        training_data = matchup_log[(matchup_log["game_end"] <= tourney_start) & (matchup_log["season"] == season)]
        full_training_data = full_training_data.append(training_data)

        prediction_data = matchup_log[(matchup_log["game_end"] > tourney_start) & (matchup_log["season"] == season)]
        prediction_data['tourney_start'] = tourney_start
        if only_matchups_with_history and "matchup_log" in es.entity_dict.keys():
            valid_prediction_data = prediction_data[
                prediction_data['matchup_id'].isin(full_training_data['matchup_id'])]
        else:
            valid_prediction_data = prediction_data
        valid_pred_data = valid_pred_data.append(valid_prediction_data)

    labels = valid_pred_data
    labels["time"] = labels['game_datetime']  # TODO: change to work with tourney_start, I had issues when I tried this
    labels = labels[index_cols+["time"] + label_cols]

    return labels.reset_index(drop=True)

def make_day_of_labels(es, training_window, cutoff_time,
                       prediction_window):
    #makes labels computed on the day of games
    #based on https://github.com/Featuretools/predict-next-purchase/blob/master/utils.py
    prediction_window_end = cutoff_time + prediction_window
    t_start = cutoff_time - training_window

    matchup_logs = es["matchup_logs"].df

    time_index = "game_datetime"
    training_data = matchup_logs[(matchup_logs[time_index] <= cutoff_time) & (matchup_logs[time_index] > t_start)]
    prediction_data = matchup_logs[
        (matchup_logs[time_index] > cutoff_time) & (matchup_logs[time_index] < prediction_window_end)]

    matchup_ids_in_training = training_data['matchup_id'].unique()
    valid_pred_data = prediction_data[prediction_data['matchup_id'].isin(matchup_ids_in_training)]
    extra_cols = ['team_1_wins']
    labels = valid_pred_data[["game_index", "game_datetime", ] + extra_cols]
    labels["time"] = labels['game_datetime']
    del labels['game_datetime']
    labels = labels[["game_index", "time", ] + extra_cols]

    return labels.reset_index(drop=True)


def build_possible_tourney_games():

    def get_season_game_index(row):
        team_1_id, team_2_id = sorted([row['Team1_ID'], row['Team2_ID']])
        return "%s_%s_%s" % (row['Season'], team_1_id, team_2_id)

    possible_games_df = pd.DataFrame()
    for season, seeds in tourney_seeds.groupby("Season"):
        possible_games = list(combinations(sorted(tourney_seeds[tourney_seeds['Season'] == season]['TeamID']), 2))
        df = pd.DataFrame(possible_games, columns=["Team1_ID", "Team2_ID"])
        df['Season'] = season
        possible_games_df = possible_games_df.append(df)
    possible_games_df = possible_games_df[possible_games_df['Season'] >= 2014]

    possible_games_df['TeamID'] = possible_games_df.apply(lambda row: min([row["Team1_ID"], row["Team2_ID"]]), axis=1)
    possible_games_df['ID'] = possible_games_df.apply(lambda row: "%s_%s_%s" % (
    row['Season'], min([row["Team1_ID"], row["Team2_ID"]]), max([row["Team1_ID"], row["Team2_ID"]])), axis=1)
    return possible_games_df

def transform_to_Xy(fm_encoded, labels, restrict_to_features=[]):
    remove_cols=[col for col in fm_encoded.columns if "wins" in col ]
    X = fm_encoded.drop(remove_cols, axis=1).reset_index().merge(labels, on=["team_game_index", "time"])

    if "index" in X:
        del X['index']
    X.drop(["team_game_index", "time"], axis=1, inplace=True)
    X = X.replace([np.inf, -np.inf],np.nan).fillna(0)
    y = X.pop("team_wins")
    if restrict_to_features:
        X=X[restrict_to_features]

    return X,y

def build_played_kaggle_labels(matchups_es, tourney_starts):
    all_tourney_labels = make_tourney_labels(matchups_es, tourney_starts, only_matchups_with_history=False)
    all_tourney_labels['season'] = all_tourney_labels['game_index'].apply(lambda x: int(x.split("_")[0]))
    kaggle_played_tourney_labels=all_tourney_labels[all_tourney_labels['season'] >= 2014]#kaggle region of interest
    return kaggle_played_tourney_labels

def merge_features(team_game_log_fm_encoded):
    team_game_features = team_game_log_fm_encoded.reset_index().sort_values(by="team_game_index")
    team_game_features['game_index'] = team_game_features['team_game_index'].apply(
        lambda x: "_".join(x.split("_")[:-1]))
    merged_features = pd.DataFrame()
    for game_index, game in team_game_features.groupby("game_index"):
        assert (len(game) == 2)
        merged = pd.DataFrame([game.iloc[0]]).merge(pd.DataFrame([game.iloc[1]]), on="game_index",
                                                    suffixes=["_team1", "_team2"])
        merged2 = pd.DataFrame([game.iloc[0]]).merge(pd.DataFrame([game.iloc[1]]), on="game_index",
                                                     suffixes=["_team2", "_team1"])
        merged_features = merged_features.append(merged, sort=False).append(merged2, sort=False)
    merged_features=merged_features.rename(columns={

        "team_game_index_team1": "team_game_index",
        "time_team1": "time",
        "team_wins_team1": "team_1_wins"
    })
    merged_features.drop(["time_team2"],axis=1,inplace=True)
    return merged_features

def merge_matchups_and_team_game_log_features(matchups_fm_encoded,team_game_log_fm_encoded):
    team_game_features = team_game_log_fm_encoded.reset_index().sort_values(by="team_game_index")
    team_game_features['game_index'] = team_game_features['team_game_index'].apply(
        lambda x: "_".join(x.split("_")[:-1]))
    team_1_game_features = team_game_features.groupby("game_index", as_index=False).first()
    team_2_game_features = team_game_features.groupby("game_index", as_index=False).last()
    matchup_features = matchups_fm_encoded.reset_index()
    join_cols = ["game_index", "time"]
    merge_cols = list(team_1_game_features.columns.difference(matchup_features.columns)) + join_cols

    fm_encoded = matchup_features.merge(team_1_game_features[merge_cols], on=join_cols).merge(
        team_2_game_features[merge_cols], on=join_cols, suffixes=["_tgm_team", "_tgm_opp"])



    fm_encoded = fm_encoded.rename(columns={
            "team_1_is_higher_seed":"team_1_is_higher_seed_matchup",
            "team_1_seed": "team_1_seed_matchup",
            "team_2_seed": "team_2_seed_matchup",
                                            })

    #these are duplicated for matchups and team_game_logs
    # baseline_features = ["team_1_seed_matchup", "team_2_seed_matchup", "team_1_is_higher_seed_matchup"]
    # baseline_features2 = ["team_seed_tgm_team", "opp_seed_tgm_team", "team_is_higher_seed_tgm_team", ]
    # for col1, col2 in zip(baseline_features, baseline_features2):
    #     print((fm_encoded[col1] == fm_encoded[col2]).all())
    # fm_encoded.drop(["team_is_higher_seed_tgm_team", "opp_seed_tgm_team","team_seed_tgm_team"],axis=1,inplace=True)
    return fm_encoded

def game_index_to_kaggle_id(game_index):
    vals=game_index.split("_")
    vals.pop(1)
    return "_".join(vals)

def evaluate_features(X, y, clf=RandomForestClassifier(n_estimators=400, n_jobs=-1), metric="roc_auc"):
    scores = cross_val_score(estimator=clf, X=X, y=y, cv=3,
                             scoring=metric, verbose=True)

    print("AUC %.2f +/- %.2f" % (scores.mean(), scores.std()))

def evaluate_model(X, y, TEST_SIZE=.5, clf=RandomForestClassifier(n_estimators=100, n_jobs=-1, max_depth=3)):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=23
    )

    #
    clf.fit(X_train, y_train, )

    y_hat_train = clf.predict_proba(X_train)[:, 1]
    y_hat_test = clf.predict_proba(X_test)[:, 1]
    try:
        print("Log Loss")
        print("TRAIN:", log_loss(y_train, y_hat_train))
        print("TEST:", log_loss(y_test, y_hat_test))
        print()
    except:
        pass
    print("Accuracy")
    print("TRAIN:", accuracy_score(y_train, clf.predict(X_train)))
    print("TEST:", accuracy_score(y_test, clf.predict(X_test)))

    print()
    return clf

def feature_selection(feature_matrix, missing_threshold=90, correlation_threshold=0.95):
    """Feature selection for a dataframe."""

    n_features_start = feature_matrix.shape[1]
    print('Original shape: ', feature_matrix.shape)

    # Find missing and percentage
    missing = pd.DataFrame(feature_matrix.isnull().sum())
    missing['percent'] = 100 * (missing[0] / feature_matrix.shape[0])
    missing.sort_values('percent', ascending=False, inplace=True)

    # Missing above threshold
    missing_cols = list(missing[missing['percent'] > missing_threshold].index)
    n_missing_cols = len(missing_cols)

    # Remove missing columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in missing_cols]]
    print('{} missing columns with threshold: {}.'.format(n_missing_cols,
                                                          missing_threshold))

    # Zero variance
    unique_counts = pd.DataFrame(feature_matrix.nunique()).sort_values(0, ascending=True)
    zero_variance_cols = list(unique_counts[unique_counts[0] == 1].index)
    n_zero_variance_cols = len(zero_variance_cols)

    # Remove zero variance columns
    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in zero_variance_cols]]
    print('{} zero variance columns.'.format(n_zero_variance_cols))

    # Correlations
    corr_matrix = feature_matrix.corr()

    # Extract the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Select the features with correlations above the threshold
    # Need to use the absolute value
    to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

    n_collinear = len(to_drop)

    feature_matrix = feature_matrix[[x for x in feature_matrix if x not in to_drop]]
    print('{} collinear columns removed with threshold: {}.'.format(n_collinear,
                                                                    correlation_threshold))

    total_removed = n_missing_cols + n_zero_variance_cols + n_collinear

    print('Total columns removed: ', total_removed)
    print('Shape after feature selection: {}.'.format(feature_matrix.shape))
    return feature_matrix