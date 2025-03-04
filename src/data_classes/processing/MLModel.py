from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import train_test_split
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
        self.feature_columns = None

        self.data_manager.load_data()

    def create_feature_matrix(self, start_season=2003):
        """Create feature matrix from historical matchups for ML training with robust handling of inconsistent features"""
        print(f"Creating features from {start_season} to current season...")
        features = []
        outcomes = []
        all_feature_names = set()
        
        # Get all tournament games
        tourney_games = self.data_manager.data['tourney_results']
        tourney_games = tourney_games[(tourney_games['Season'] >= start_season) & (tourney_games['Season'] < self.data_manager.current_season)]
        
        # First pass: identify all possible features
        print("First pass: identifying all possible features...")
        for _, game in tourney_games.iterrows():
            season = game['Season']
            day_num = game['DayNum']
            team1_id = game['WTeamID']  # Winner
            team2_id = game['LTeamID']  # Loser
            
            # Skip if we don't have necessary data
            if not self._has_required_data(season, team1_id, team2_id, 106):
                continue
                
            # Create feature vector for this matchup
            _, column_names = self._create_game_features(team1_id, team2_id, season, day_num)
            
            # Add to our set of all possible feature names
            if column_names:
                all_feature_names.update(column_names)
        
        # Convert to a sorted list to ensure consistent order
        all_feature_names = sorted(list(all_feature_names))
        print(f"Identified {len(all_feature_names)} unique features")
        
        # Second pass: create feature vectors with consistent dimensions
        print("Second pass: creating consistent feature vectors...")
        for _, game in tourney_games.iterrows():
            season = game['Season']
            day_num = game['DayNum']
            team1_id = game['WTeamID']  # Winner
            team2_id = game['LTeamID']  # Loser
            
            # Skip if we don't have necessary data

            if not self._has_required_data(season, team1_id, team2_id, 106):
                print('missing data')
                continue
                
            # Create feature vector for this matchup
            feature_vector, column_names = self._create_game_features(team1_id, team2_id, season, day_num)
            
            # Create a standardized feature vector with all possible features
            # Fill with zeros for any missing features
            standardized_vector = [0] * len(all_feature_names)
            
            # Fill in the values we have
            for i, name in enumerate(column_names):
                if name in all_feature_names:
                    idx = all_feature_names.index(name)
                    standardized_vector[idx] = feature_vector[i]
            
            # Add to our dataset
            features.append(standardized_vector)
            outcomes.append(1)  # Team1 won
            
            # Also add reversed matchup with flipped outcome
            reversed_features, reversed_names = self._create_game_features(team2_id, team1_id, season, day_num)
            
            # Create standardized vector for reversed matchup
            reversed_std_vector = [0] * len(all_feature_names)
            for i, name in enumerate(reversed_names):
                if name in all_feature_names:
                    idx = all_feature_names.index(name)
                    reversed_std_vector[idx] = reversed_features[i]
                    
            features.append(reversed_std_vector)
            outcomes.append(0)  # Team2 lost
        
        # Check for feature vector consistency
        feature_lengths = [len(f) for f in features]
        unique_lengths = set(feature_lengths)
        if len(unique_lengths) > 1:
            print(f"WARNING: Inconsistent feature lengths detected: {unique_lengths}")
            
            # Force consistency by padding or truncating
            max_length = max(unique_lengths)
            features = [f + [0] * (max_length - len(f)) if len(f) < max_length else f[:max_length] for f in features]
            print(f"Forced all feature vectors to length {max_length}")
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(outcomes)
        
        self.feature_columns = all_feature_names
        
        print(f"Created dataset with {X.shape[0]} samples and {X.shape[1]} features")
        return X, y

    def _create_game_features(self, team1_id, team2_id, season, day_num):
        """Create feature vector for a single game with consistent feature naming"""
        features = []
        feature_names = []
        
        try:
            # Team seed features
            team1_seed = self.data_manager.seed_lookup.get((season, team1_id), 16)
            team2_seed = self.data_manager.seed_lookup.get((season, team2_id), 16)
            
            features.append(team1_seed)
            feature_names.append('Team1Seed')
            
            features.append(team2_seed)
            feature_names.append('Team2Seed')
            
            features.append(team1_seed - team2_seed)
            feature_names.append('SeedDiff')
            
            # ELO ratings
            team1_elo = self.elo_system.get_team_elo(season, team1_id, day_num-1)
            team2_elo = self.elo_system.get_team_elo(season, team2_id, day_num-1)
            
            features.append(team1_elo)
            feature_names.append('Team1Elo')
            
            features.append(team2_elo)
            feature_names.append('Team2Elo')
            
            features.append(team1_elo - team2_elo)
            feature_names.append('EloDiff')
            
            # Include raw ELO win probability as a feature
            elo_prob = self.elo_system.elo_win_probability(team1_elo, team2_elo)
            features.append(elo_prob)
            feature_names.append('EloProb')
            
            # Advanced stats features (if available)
            if hasattr(self.stats_calculator, 'advanced_team_stats') and season in self.stats_calculator.advanced_team_stats:
                team1_stats = self.stats_calculator.advanced_team_stats[season].get(team1_id, {})
                team2_stats = self.stats_calculator.advanced_team_stats[season].get(team2_id, {})
                
                # Add key stats as features - safely handle missing stats
                for stat in ['OffEff', 'DefEff', 'NetEff', 'Pace', 'eFG%', 'TOV%', 'ORB%', 'FTRate']:
                    # Team 1 stats
                    if stat in team1_stats:
                        features.append(team1_stats[stat])
                        feature_names.append(f'Team1_{stat}')
                    else:
                        # Use 0 as placeholder for missing stat
                        features.append(0)
                        feature_names.append(f'Team1_{stat}')
                    
                    # Team 2 stats
                    if stat in team2_stats:
                        features.append(team2_stats[stat])
                        feature_names.append(f'Team2_{stat}')
                    else:
                        # Use 0 as placeholder for missing stat
                        features.append(0)
                        feature_names.append(f'Team2_{stat}')
                    
                    # Add difference feature if both stats are available
                    if stat in team1_stats and stat in team2_stats:
                        features.append(team1_stats[stat] - team2_stats[stat])
                        feature_names.append(f'{stat}_Diff')
                    else:
                        # Use 0 as placeholder
                        features.append(0)
                        feature_names.append(f'{stat}_Diff')
        except Exception as e:
            print(f"Error creating features for {team1_id} vs {team2_id} in {season}: {str(e)}")
            # Return minimal features to avoid errors
            return [16, 16, 0, 1500, 1500, 0, 0.5], ['Team1Seed', 'Team2Seed', 'SeedDiff', 'Team1Elo', 'Team2Elo', 'EloDiff', 'EloProb']
        
        return features, feature_names

    def _has_required_data(self, season, team1_id, team2_id, max_days):
        """Check if we have all required data for a matchup - now more lenient for women's tournament"""
        try:
            # Check if we have seed data (most critical)
            has_seeds = (season, team1_id) in self.data_manager.seed_lookup and \
                        (season, team2_id) in self.data_manager.seed_lookup
            
            # For women's tournament, be more lenient with ELO requirements
            if self.data_manager.gender == 'W':
                # Just require seed data
                return has_seeds
            else:
                # For men's tournament, require both seeds and ELO
                has_elo = hasattr(self.elo_system, 'team_elo_ratings') and \
                        (season, team1_id, max_days) in self.elo_system.team_elo_ratings and \
                        (season, team2_id, max_days) in self.elo_system.team_elo_ratings
                
                return has_seeds and has_elo
        except Exception as e:
            print(f"Error checking required data: {str(e)}")
            return False

    def train_model(self, model_type="xgboost", test_size=0.2, random_state=42):
        """Train the ML model on historical data"""
        # Create feature matrix
        X, y = self.create_feature_matrix()

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

    def predict(self, team1_id, team2_id, day_num, season):
        """Make a prediction for a specific matchup using the ML model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Create features for this matchup
        features, _ = self._create_game_features(team1_id, team2_id, season, day_num)
        features = np.array([features])

        # Make prediction
        prediction = self.model.predict_proba(features)[0, 1]

        return prediction
