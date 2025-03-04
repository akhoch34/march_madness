from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .DataManager import MarchMadnessDataManager
from .EloRatingSystem import EloRatingSystem
from .TeamStatsCalculator import TeamStatsCalculator


class MarchMadnessMLModel:
    def __init__(
        self,
        data_manager: MarchMadnessDataManager,
        elo_system: EloRatingSystem,
        stats_calculator: TeamStatsCalculator,
    ):
        self.data_manager = data_manager
        self.elo_system = elo_system
        self.stats_calculator = stats_calculator
        self.model = None
        self.feature_df: pd.DataFrame = None
        self.feature_columns = None
        self.exclude_columns = ['Result', 'Season', 'Team1ID', 'Team2ID']
        self.seed_lookup: dict = None
        
    def create_feature_dataset(self, train_years_range=(2010, 2024), include_elo=True, include_advanced_stats=True, include_all_matchups=False):
        """
        Create a dataset with features for training and prediction.
        
        Parameters:
        train_years_range (tuple): Range of years to use for training (inclusive)
        include_elo (bool): Whether to include ELO rating features
        include_advanced_stats (bool): Whether to include advanced box score stats
        """
        if self.feature_df is not None:
            print('Skipping feature creation, we already have a dataframe')
            return self.feature_df
        
        print("Creating feature dataset...")
        # Process seeds first
        self.data_manager.preprocess_seeds()
        
        # Calculate ELO ratings if needed
        if include_elo and not self.elo_system.team_elo_ratings:
            self.elo_system.calculate_elo_ratings(start_year=min(train_years_range[0] - 2, 2003))
            
        # Calculate advanced stats if needed
        if include_advanced_stats and (not hasattr(self, 'advanced_team_stats') or not self.stats_calculator.advanced_team_stats):
            self.stats_calculator.calculate_advanced_team_stats(start_season=min(train_years_range[0], 2003))
        
        # Get all possible tournament matchups from historical data
        tourney_games = self.data_manager.data['tourney_results'].copy()
        
        # Create features for each historical matchup
        features = []
        
        for _, game in tourney_games.iterrows():
            season = game['Season']
            
            # Skip if outside our training range
            if season < train_years_range[0] or season > train_years_range[1]:
                continue
                
            team1_id = game['WTeamID']  # Winner
            team2_id = game['LTeamID']  # Loser
            day_num = game['DayNum']
            
            # Get seed information
            team1_seed = self.seed_lookup.get((season, team1_id), 16)  # Default to 16 if not found
            team2_seed = self.seed_lookup.get((season, team2_id), 16)
            
            # Basic features
            game_features = {
                'Season': season,
                'Team1ID': team1_id,
                'Team2ID': team2_id,
                'Team1Seed': team1_seed,
                'Team2Seed': team2_seed,
                'SeedDiff': team2_seed - team1_seed,
                'Result': 1  # Team1 won
            }
            
            # Add season performance metrics
            game_features.update(self._get_season_stats(season, team1_id, team2_id))
            
            # Add ranking features if available
            if self.data_manager.rankings_available:
                game_features.update(self._get_ranking_features(season, team1_id, team2_id))
                
            # Add ELO rating features if available
            if include_elo and self.elo_system.team_elo_ratings:
                # Get ELO ratings just before this tournament game
                # We use day_num - 1 to ensure we don't leak future information
                team1_elo = self.elo_system.get_team_elo(season, team1_id, day_num - 1)
                team2_elo = self.elo_system.get_team_elo(season, team2_id, day_num - 1)
                
                # Calculate win probability
                elo_win_prob = self.elo_system.elo_win_probability(team1_elo, team2_elo)
                
                game_features.update({
                    'Team1ELO': team1_elo,
                    'Team2ELO': team2_elo,
                    'ELODiff': team1_elo - team2_elo,
                    'ELOWinProb': elo_win_prob
                })
                
            # Add advanced stats features if available
            if include_advanced_stats and hasattr(self, 'advanced_team_stats') and self.stats_calculator.advanced_team_stats:
                game_features.update(self._get_advanced_stats_features(season, team1_id, team2_id))
            
            features.append(game_features)
            
            # Also add the reversed matchup (with opposite result)
            reversed_features = game_features.copy()
            reversed_features['Team1ID'] = team2_id
            reversed_features['Team2ID'] = team1_id
            reversed_features['Team1Seed'] = team2_seed
            reversed_features['Team2Seed'] = team1_seed
            reversed_features['SeedDiff'] = team1_seed - team2_seed
            reversed_features['Result'] = 0  # Team1 lost
            
            # Reverse any asymmetric stat features
            if 'Team1WinPct' in reversed_features:
                reversed_features['Team1WinPct'] = game_features['Team2WinPct']
                reversed_features['Team2WinPct'] = game_features['Team1WinPct']
                
            # Reverse strength of schedule features if present
            if 'Team1SOS' in reversed_features:
                reversed_features['Team1SOS'] = game_features['Team2SOS'] 
                reversed_features['Team2SOS'] = game_features['Team1SOS']
                reversed_features['SOSDiff'] = -game_features['SOSDiff']
                
            # Reverse last 10 features if present
            if 'Team1Last10' in reversed_features:
                reversed_features['Team1Last10'] = game_features['Team2Last10']
                reversed_features['Team2Last10'] = game_features['Team1Last10']
                reversed_features['Last10Diff'] = -game_features['Last10Diff']
                
            # Reverse ELO features if present
            if 'Team1ELO' in reversed_features:
                reversed_features['Team1ELO'] = game_features['Team2ELO']
                reversed_features['Team2ELO'] = game_features['Team1ELO']
                reversed_features['ELODiff'] = -game_features['ELODiff']
                reversed_features['ELOWinProb'] = 1.0 - game_features['ELOWinProb']
                
            # Reverse advanced stats features if present
            for key in list(reversed_features.keys()):
                # Look for keys with Team1_ prefix that need to be swapped
                if key.startswith('Team1_') and key.replace('Team1_', 'Team2_') in reversed_features:
                    team1_key = key
                    team2_key = key.replace('Team1_', 'Team2_')
                    reversed_features[team1_key] = game_features[team2_key]
                    reversed_features[team2_key] = game_features[team1_key]
                    
                # Flip the sign of all difference features
                if key.endswith('_Diff') and key not in ['SeedDiff', 'ELODiff', 'WinPctDiff', 'SOSDiff', 'Last10Diff']:
                    reversed_features[key] = -game_features[key]
            
            features.append(reversed_features)
        
        # Create DataFrame with all features
        self.feature_df = pd.DataFrame(features)
        print(f"Created feature dataset with {len(self.feature_df)} samples")
        
        # Define feature columns (excluding outcome and identifiers)
        self.feature_columns = [col for col in self.feature_df.columns 
                            if col not in ['Result', 'Season', 'Team1ID', 'Team2ID']]
        
        return self.feature_df
    
    def _get_season_stats(self, season, team1_id, team2_id):
        """Get season performance stats for both teams"""
        # Filter regular season games for this season
        season_games = self.data_manager.data['regular_season'][self.data_manager.data['regular_season']['Season'] == season]
        
        # Team1 stats
        team1_wins = season_games[season_games['WTeamID'] == team1_id].shape[0]
        team1_losses = season_games[season_games['LTeamID'] == team1_id].shape[0]
        team1_win_pct = team1_wins / (team1_wins + team1_losses) if (team1_wins + team1_losses) > 0 else 0
        
        # Team2 stats
        team2_wins = season_games[season_games['WTeamID'] == team2_id].shape[0]
        team2_losses = season_games[season_games['LTeamID'] == team2_id].shape[0]
        team2_win_pct = team2_wins / (team2_wins + team2_losses) if (team2_wins + team2_losses) > 0 else 0
        
        # Calculate strength of schedule
        if hasattr(self, 'advanced_team_stats') and self.stats_calculator.advanced_team_stats and season in self.stats_calculator.advanced_team_stats:
            # Get list of opponents and their net efficiency
            team1_opponents = []
            team2_opponents = []
            
            # Get opponents from wins
            for _, game in season_games[season_games['WTeamID'] == team1_id].iterrows():
                team1_opponents.append(game['LTeamID'])
            
            for _, game in season_games[season_games['WTeamID'] == team2_id].iterrows():
                team2_opponents.append(game['LTeamID'])
            
            # Get opponents from losses
            for _, game in season_games[season_games['LTeamID'] == team1_id].iterrows():
                team1_opponents.append(game['WTeamID'])
            
            for _, game in season_games[season_games['LTeamID'] == team2_id].iterrows():
                team2_opponents.append(game['WTeamID'])
            
            # Calculate average opponent net efficiency
            season_stats = self.stats_calculator.advanced_team_stats[season]
            
            team1_opp_net_eff = [season_stats.get(opp, {}).get('NetEff', 0) for opp in team1_opponents]
            team2_opp_net_eff = [season_stats.get(opp, {}).get('NetEff', 0) for opp in team2_opponents]
            
            team1_sos = np.mean(team1_opp_net_eff) if team1_opp_net_eff else 0
            team2_sos = np.mean(team2_opp_net_eff) if team2_opp_net_eff else 0
            
            # Get all games for each team sorted by day
            team1_games = pd.concat([
                season_games[season_games['WTeamID'] == team1_id].assign(Result=1),
                season_games[season_games['LTeamID'] == team1_id].assign(Result=0)
            ]).sort_values('DayNum', ascending=False).head(10)
            
            team2_games = pd.concat([
                season_games[season_games['WTeamID'] == team2_id].assign(Result=1),
                season_games[season_games['LTeamID'] == team2_id].assign(Result=0)
            ]).sort_values('DayNum', ascending=False).head(10)
            
            team1_last_10_win_pct = team1_games['Result'].mean() if len(team1_games) > 0 else team1_win_pct
            team2_last_10_win_pct = team2_games['Result'].mean() if len(team2_games) > 0 else team2_win_pct
            
            return {
                'Team1WinPct': team1_win_pct,
                'Team2WinPct': team2_win_pct,
                'WinPctDiff': team1_win_pct - team2_win_pct,
                'Team1SOS': team1_sos,
                'Team2SOS': team2_sos,
                'SOSDiff': team1_sos - team2_sos,
                'Team1Last10': team1_last_10_win_pct,
                'Team2Last10': team2_last_10_win_pct,
                'Last10Diff': team1_last_10_win_pct - team2_last_10_win_pct
            }
        else:
            # Basic stats if advanced stats aren't available
            return {
                'Team1WinPct': team1_win_pct,
                'Team2WinPct': team2_win_pct,
                'WinPctDiff': team1_win_pct - team2_win_pct
            }

    def _get_ranking_features(self, season, team1_id, team2_id):
        """Get pre-tournament ranking features for both teams"""
        if not self.data_manager.rankings_available:
            return {}

        # Get final rankings before tournament (RankingDayNum = 133)
        rankings = self.data_manager.data["rankings"]
        pre_tourney_rankings = rankings[
            (rankings["Season"] == season) & (rankings["RankingDayNum"] == 133)
        ]

        # Aggregate rankings across systems (use mean)
        team_ranks = {}
        for _, row in pre_tourney_rankings.iterrows():
            team_id = row["TeamID"]
            if team_id not in team_ranks:
                team_ranks[team_id] = []
            team_ranks[team_id].append(row["OrdinalRank"])

        # Calculate average ranking for each team
        team1_avg_rank = (
            np.mean(team_ranks.get(team1_id, [353])) if team1_id in team_ranks else 353
        )
        team2_avg_rank = (
            np.mean(team_ranks.get(team2_id, [353])) if team2_id in team_ranks else 353
        )

        return {
            "Team1AvgRank": team1_avg_rank,
            "Team2AvgRank": team2_avg_rank,
            "RankDiff": team2_avg_rank
            - team1_avg_rank,  # Positive if team1 is ranked better
        }
    
    def _get_advanced_stats_features(self, season, team1_id, team2_id):
        """Get advanced stats features for both teams"""
        if not hasattr(self, 'advanced_team_stats') or not self.stats_calculator.advanced_team_stats:
            self.stats_calculator.calculate_advanced_team_stats()
            
        # Get stats for this season
        if season not in self.stats_calculator.advanced_team_stats:
            # If season not available, return empty dict
            return {}
            
        season_stats = self.stats_calculator.advanced_team_stats[season]
        
        # Get stats for both teams
        team1_stats = season_stats.get(team1_id, {})
        team2_stats = season_stats.get(team2_id, {})
        
        # Skip if either team doesn't have stats
        if not team1_stats or not team2_stats:
            return {}
            
        # Create features dictionary
        features = {}
        
        # Four Factors - the most predictive advanced metrics
        for factor in ['eFG%', 'TOV%', 'ORB%', 'FTRate']:
            # Offensive factors
            features[f'Team1_{factor}'] = team1_stats.get(factor, 0)
            features[f'Team2_{factor}'] = team2_stats.get(factor, 0)
            features[f'{factor}_Diff'] = team1_stats.get(factor, 0) - team2_stats.get(factor, 0)
            
            # Defensive factors (opponent's numbers)
            opp_factor = f'Opp{factor}'
            if opp_factor in team1_stats:
                features[f'Team1_Def_{factor}'] = team1_stats.get(opp_factor, 0)
                features[f'Team2_Def_{factor}'] = team2_stats.get(opp_factor, 0)
                features[f'Def_{factor}_Diff'] = team1_stats.get(opp_factor, 0) - team2_stats.get(opp_factor, 0)
        
        # Efficiency metrics
        for metric in ['OffEff', 'DefEff', 'NetEff']:
            features[f'Team1_{metric}'] = team1_stats.get(metric, 0)
            features[f'Team2_{metric}'] = team2_stats.get(metric, 0)
            features[f'{metric}_Diff'] = team1_stats.get(metric, 0) - team2_stats.get(metric, 0)
        
        # Tempo/Pace
        features['Team1_Pace'] = team1_stats.get('Pace', 0)
        features['Team2_Pace'] = team2_stats.get('Pace', 0)
        features['Pace_Diff'] = team1_stats.get('Pace', 0) - team2_stats.get('Pace', 0)
        
        # Shooting percentages
        for pct in ['FG%', '3P%', 'FT%']:
            features[f'Team1_{pct}'] = team1_stats.get(pct, 0)
            features[f'Team2_{pct}'] = team2_stats.get(pct, 0)
            features[f'{pct}_Diff'] = team1_stats.get(pct, 0) - team2_stats.get(pct, 0)
            
            # Defensive (opponent shooting percentages)
            opp_pct = f'Opp{pct}'
            features[f'Team1_Def_{pct}'] = team1_stats.get(opp_pct, 0)
            features[f'Team2_Def_{pct}'] = team2_stats.get(opp_pct, 0)
            features[f'Def_{pct}_Diff'] = team1_stats.get(opp_pct, 0) - team2_stats.get(opp_pct, 0)
        
        # Other key stats per game
        for stat in ['PointsPerGame', 'PointsAllowedPerGame', 'AstRate', 'BlkRate', 'StlRate']:
            if stat in team1_stats:
                features[f'Team1_{stat}'] = team1_stats.get(stat, 0)
                features[f'Team2_{stat}'] = team2_stats.get(stat, 0)
                features[f'{stat}_Diff'] = team1_stats.get(stat, 0) - team2_stats.get(stat, 0)
        
        return features

    def train_model(self, model_type="randomforest", test_size=0.2, random_state=42):
        """Train the ML model on historical data"""
        # Create feature matrix
        # X, y = self.create_feature_matrix()
        print('Creating feature dataset...')
        self.create_feature_dataset()
        X = self.feature_df[self.feature_columns]
        print(X)
        y = self.feature_df['Result']

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Initialize model
        if model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                random_state=random_state,
            )
        elif model_type == 'randomforest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,  # Limit tree depth
                min_samples_leaf=5,  # Require at least 5 samples per leaf
                min_samples_split=10,  # Require at least 10 samples to split a node
                max_features='sqrt',  # Use sqrt(n_features) features per split
                random_state=42
            )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate model
        train_preds = self.model.predict_proba(X_train)[:, 1]
        test_preds = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        train_acc = accuracy_score(y_train, train_preds > 0.5)
        test_acc = accuracy_score(y_test, test_preds > 0.5)

        train_log_loss = log_loss(y_train, train_preds)
        test_log_loss = log_loss(y_test, test_preds)

        train_brier = brier_score_loss(y_train, train_preds)
        test_brier = brier_score_loss(y_test, test_preds)

        print("Model Training Results:")
        print(f"Training Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        print(
            f"Training Log Loss: {train_log_loss:.4f}, Test Log Loss: {test_log_loss:.4f}"
        )
        print(
            f"Training Brier Score: {train_brier:.4f}, Test Brier Score: {test_brier:.4f}"
        )

        # Feature importance
        if hasattr(self.model, "feature_importances_") and self.feature_columns:
            self._display_feature_importance()

        return self.model

    def _display_feature_importance(self):
        """Display feature importance from the model"""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(
            range(len(importances)),
            [self.feature_columns[i] for i in indices],
            rotation=90,
        )
        plt.tight_layout()
        plt.show()

    def _get_game_features(self, team1_id, team2_id, season) -> pd.DataFrame:
        return self.feature_df[(self.feature_df['Team1ID'] == team1_id) & (self.feature_df['Team2ID'] == team2_id) & (self.feature_df['Season'] == season)]
        
    def predict(self, team1_id, team2_id, season):
        """Make a prediction for a specific matchup using the ML model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Create features for this matchup
        all_features = self._get_game_features(team1_id, team2_id, season)
        features = all_features[[col for col in all_features.columns if col not in self.exclude_columns]]

        # Make prediction
        prediction = self.model.predict_proba(features)[0, 1]
        print(prediction)

        return prediction
    
    def analyze_feature_matrix(self, X, feature_names):
        """
        Analyze the feature matrix for issues that may affect model performance
        
        Parameters:
        X (numpy.ndarray): The feature matrix
        feature_names (list): List of feature names
        
        Returns:
        dict: Dictionary with analysis results
        """
        import numpy as np
        import pandas as pd
        from scipy import stats
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(X, columns=feature_names)
        
        # Basic statistics
        basic_stats = df.describe().T
        
        # Calculate additional metrics
        analysis = {
            'missing_values': df.isna().sum().to_dict(),
            'zero_values': (df == 0).sum().to_dict(),
            'zero_percentage': ((df == 0).sum() / len(df) * 100).to_dict(),
            'data_types': df.dtypes.to_dict(),
            'skewness': df.skew().to_dict(),
            'kurtosis': df.kurtosis().to_dict(),
        }
        
        # Check for highly correlated features
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(col1, col2, corr_matrix.loc[col1, col2]) 
                        for col1 in upper_tri.index 
                        for col2 in upper_tri.columns 
                        if upper_tri.loc[col1, col2] > 0.9]
        
        analysis['high_correlations'] = high_corr_pairs
        
        # Check for features with low variance (might not be useful for prediction)
        low_variance_cols = [col for col in df.columns if df[col].var() < 0.01]
        analysis['low_variance_features'] = low_variance_cols
        
        # Top 5 features with most zeros (might be problematic)
        zero_counts = (df == 0).sum()
        top_zero_cols = zero_counts.sort_values(ascending=False).head(5)
        analysis['top_zero_features'] = top_zero_cols.to_dict()
        
        # Check for dataset balance
        analysis['class_balance'] = "Not applicable - no target provided"
        
        return analysis, basic_stats

    # Add this to your create_feature_matrix method right after creating X and y:
    def diagnostic_print(self, X, y, feature_columns):
        print(f"\n--- FEATURE MATRIX DIAGNOSTICS ---")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples: {X.shape[0]}")
        print(f"Class balance: {np.sum(y == 1)}/{len(y)} winners ({np.mean(y)*100:.1f}% win rate)")
        
        # Analyze features
        analysis, basic_stats = self.analyze_feature_matrix(X, feature_columns)
        
        # Print key findings
        print("\n--- KEY ISSUES DETECTED ---")
        
        # Features with too many zeros
        print("\nFeatures with high percentage of zeros:")
        for feat, pct in sorted(analysis['zero_percentage'].items(), key=lambda x: x[1], reverse=True)[:5]:
            if pct > 50:  # Only show if more than 50% zeros
                print(f"  {feat}: {pct:.1f}% zeros")
        
        # Low variance features
        if analysis['low_variance_features']:
            print("\nFeatures with very low variance (might not be useful):")
            for feat in analysis['low_variance_features'][:5]:
                print(f"  {feat}")
        
        # Highly correlated features
        if analysis['high_correlations']:
            print("\nHighly correlated feature pairs (r > 0.9):")
            for col1, col2, corr in analysis['high_correlations'][:5]:
                print(f"  {col1} & {col2}: r={corr:.3f}")
        
        # Return the analysis for further use if needed
        return analysis, basic_stats
