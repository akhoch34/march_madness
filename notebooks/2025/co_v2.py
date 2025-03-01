import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from collections import defaultdict


class MarchMadnessPredictor:
    def __init__(self, data_dir, gender='M', current_season=2025):
        """
        Initialize the March Madness predictor.
        
        Parameters:
        data_dir (str): Directory containing the data files
        gender (str): 'M' for men's tournament, 'W' for women's
        current_season (int): The current season year (for prediction)
        """
        self.data_dir = data_dir
        self.gender = gender
        self.current_season = current_season
        self.data = {}
        self.model = None
        self.feature_cols = []
        self.team_elo_ratings = {}  # Store ELO ratings by (season, team_id)
        self.advanced_team_stats = {}  # Store advanced team stats by season
    
    def load_data(self):
        """Load all necessary data files"""
        # Teams data
        self.data['teams'] = pd.read_csv(f"{self.data_dir}/{self.gender}Teams.csv")
        
        # Regular season results
        self.data['regular_season'] = pd.read_csv(
            f"{self.data_dir}/{self.gender}RegularSeasonCompactResults.csv"
        )
        
        # Tournament results (for training)
        self.data['tourney_results'] = pd.read_csv(
            f"{self.data_dir}/{self.gender}NCAATourneyCompactResults.csv"
        )
        
        # Tournament seeds
        self.data['tourney_seeds'] = pd.read_csv(
            f"{self.data_dir}/{self.gender}NCAATourneySeeds.csv"
        )
        
        # Try to load detailed results if available (for advanced features)
        try:
            self.data['regular_season_detailed'] = pd.read_csv(
                f"{self.data_dir}/{self.gender}RegularSeasonDetailedResults.csv"
            )
            self.detailed_stats_available = True
            
            # Also load tournament detailed results
            self.data['tourney_detailed'] = pd.read_csv(
                f"{self.data_dir}/{self.gender}NCAATourneyDetailedResults.csv"
            )
        except FileNotFoundError:
            self.detailed_stats_available = False
            
        # Try to load rankings data if available (men's only)
        if self.gender == 'M':
            try:
                self.data['rankings'] = pd.read_csv(f"{self.data_dir}/MMasseyOrdinals.csv")
                self.rankings_available = True
            except FileNotFoundError:
                self.rankings_available = False
        else:
            self.rankings_available = False
            
        # Load secondary tournament results if available (for more games to train ELO)
        try:
            self.data['secondary_tourney'] = pd.read_csv(
                f"{self.data_dir}/{self.gender}SecondaryTourneyCompactResults.csv"
            )
            self.secondary_tourney_available = True
        except FileNotFoundError:
            self.secondary_tourney_available = False
        
        print(f"Loaded {len(self.data)} datasets")

    def preprocess_seeds(self):
        """Process tournament seeds to extract region and numeric seed value"""
        # Extract numeric seed from seed string
        def extract_seed_number(seed_str):
            # Remove region identifier and possible play-in indicator
            return int(seed_str[1:3])
        
        # Process seeds for easier use
        seeds_df = self.data['tourney_seeds'].copy()
        seeds_df['SeedNumber'] = seeds_df['Seed'].apply(extract_seed_number)
        seeds_df['SeedRegion'] = seeds_df['Seed'].str[0]
        
        # Create a dictionary for quick seed lookup
        seed_dict = {}
        for _, row in seeds_df.iterrows():
            key = (row['Season'], row['TeamID'])
            seed_dict[key] = row['SeedNumber']
        
        self.data['processed_seeds'] = seeds_df
        self.seed_lookup = seed_dict
        
    def _get_ranking_features(self, season, team1_id, team2_id):
        """Get pre-tournament ranking features for both teams"""
        if not self.rankings_available:
            return {}
            
        # Get final rankings before tournament (RankingDayNum = 133)
        rankings = self.data['rankings']
        pre_tourney_rankings = rankings[(rankings['Season'] == season) & 
                                       (rankings['RankingDayNum'] == 133)]
        
        # Aggregate rankings across systems (use mean)
        team_ranks = {}
        for _, row in pre_tourney_rankings.iterrows():
            team_id = row['TeamID']
            if team_id not in team_ranks:
                team_ranks[team_id] = []
            team_ranks[team_id].append(row['OrdinalRank'])
        
        # Calculate average ranking for each team
        team1_avg_rank = np.mean(team_ranks.get(team1_id, [353])) if team1_id in team_ranks else 353
        team2_avg_rank = np.mean(team_ranks.get(team2_id, [353])) if team2_id in team_ranks else 353
        
        return {
            'Team1AvgRank': team1_avg_rank,
            'Team2AvgRank': team2_avg_rank,
            'RankDiff': team2_avg_rank - team1_avg_rank  # Positive if team1 is ranked better
        }
        
    def _get_early_season_rankings(self, season):
        """Get early season rankings (as a proxy for preseason rankings)"""
        if not self.rankings_available:
            return {}
            
        # Get rankings from early in the season (typically first 2-3 weeks)
        # Using RankingDayNum = 45 (roughly mid-December)
        rankings = self.data['rankings']
        early_rankings = rankings[(rankings['Season'] == season) & 
                                 (rankings['RankingDayNum'] <= 45)]
        
        # Take the earliest available ranking for each system and team
        early_rankings = early_rankings.sort_values('RankingDayNum')
        early_rankings = early_rankings.drop_duplicates(subset=['Season', 'SystemName', 'TeamID'], keep='first')
        
        # Aggregate across systems
        team_ranks = defaultdict(list)
        for _, row in early_rankings.iterrows():
            team_ranks[row['TeamID']].append(row['OrdinalRank'])
            
        # Calculate average early ranking for each team
        avg_ranks = {team_id: np.mean(ranks) for team_id, ranks in team_ranks.items()}
        
        return avg_ranks
    
    def calculate_elo_ratings(self, start_year=2003, k_factor=20, home_advantage=100, 
                          carry_over_factor=0.75, new_team_rating=1500, reset_each_year=False):
        """
        Calculate ELO ratings for all teams across multiple seasons.
        
        Parameters:
        start_year (int): First year to calculate ELO ratings for
        k_factor (float): How much each game impacts ELO (higher = more impact)
        home_advantage (float): ELO points added for home court advantage
        carry_over_factor (float): How much of previous season's rating carries over (0-1)
        new_team_rating (float): Default rating for new teams
        reset_each_year (bool): Whether to reset ratings each season (if True, carry_over_factor is ignored)
        """
        print(f"Calculating ELO ratings from {start_year} to {self.current_season}...")
        
        # Initialize with default ratings
        self.team_elo_ratings = {}
        
        # Get all regular season games, sorted by season and day
        all_games = pd.concat([
            self.data['regular_season'],
            self.data['tourney_results']
        ])
        
        # Add secondary tournament games if available
        if hasattr(self, 'secondary_tourney_available') and self.secondary_tourney_available:
            all_games = pd.concat([all_games, self.data['secondary_tourney']])
            
        # Sort by season and day
        all_games = all_games.sort_values(['Season', 'DayNum'])
        
        # Get list of seasons and all teams
        seasons = all_games['Season'].unique()
        seasons.sort()
        seasons = seasons[seasons >= start_year]
        
        all_teams = set(self.data['teams']['TeamID'].unique())
        
        # Process each season
        current_ratings = {team_id: new_team_rating for team_id in all_teams}
        
        for i, season in enumerate(seasons):
            # Apply carry-over from previous season (or reset)
            if i > 0 and not reset_each_year:
                for team_id in current_ratings:
                    # Regress toward the mean
                    current_ratings[team_id] = new_team_rating + carry_over_factor * (current_ratings[team_id] - new_team_rating)
            else:
                # Reset all ratings
                current_ratings = {team_id: new_team_rating for team_id in all_teams}
                
            # Apply preseason adjustments based on early rankings if available
            if self.rankings_available:
                early_ranks = self._get_early_season_rankings(season)
                
                # Map rankings to ELO adjustments
                # Teams in top 25 get a boost, lower ranked teams get smaller adjustment
                for team_id, rank in early_ranks.items():
                    if rank <= 25:
                        # Top 25 teams get bigger boost
                        adjustment = 100 - (rank - 1) * 4  # #1 gets +100, #25 gets +4
                    elif rank <= 100:
                        # Teams 26-100 get small boost
                        adjustment = max(0, 10 - (rank - 25) * 0.1)  # Linear decrease from +10 to 0
                    else:
                        # Teams outside top 100 get small penalty
                        adjustment = min(0, -((rank - 100) * 0.05))  # Small penalty for very low ranked teams
                        
                    if team_id in current_ratings:
                        current_ratings[team_id] += adjustment
            
            # Store initial season ratings
            for team_id, rating in current_ratings.items():
                self.team_elo_ratings[(season, team_id, 0)] = rating
            
            # Process each game in the season
            season_games = all_games[all_games['Season'] == season]
            
            for _, game in season_games.iterrows():
                w_team = game['WTeamID']
                l_team = game['LTeamID']
                day_num = game['DayNum']
                w_loc = game['WLoc']
                
                # Get current ratings
                w_rating = current_ratings.get(w_team, new_team_rating)
                l_rating = current_ratings.get(l_team, new_team_rating)
                
                # Adjust for home court advantage
                if w_loc == 'H':
                    # Winner at home
                    adjusted_w_rating = w_rating + home_advantage
                    adjusted_l_rating = l_rating
                elif w_loc == 'A':
                    # Winner away
                    adjusted_w_rating = w_rating
                    adjusted_l_rating = l_rating + home_advantage
                else:
                    # Neutral court
                    adjusted_w_rating = w_rating
                    adjusted_l_rating = l_rating
                
                # Calculate win probability based on ELO
                win_prob = 1.0 / (1.0 + math.pow(10, (adjusted_l_rating - adjusted_w_rating) / 400.0))
                
                # Update ratings
                rating_change = k_factor * (1.0 - win_prob)
                current_ratings[w_team] = w_rating + rating_change
                current_ratings[l_team] = l_rating - rating_change
                
                # Store updated ratings after each game
                self.team_elo_ratings[(season, w_team, day_num)] = current_ratings[w_team]
                self.team_elo_ratings[(season, l_team, day_num)] = current_ratings[l_team]
                
        print(f"Calculated ELO ratings for {len(seasons)} seasons")
        return self.team_elo_ratings
    
    def get_team_elo(self, season, team_id, day_num=None):
        """Get a team's ELO rating for a specific season and day"""
        if day_num is None:
            # If no day specified, get rating before tournament (day 132)
            day_num = 132
            
        # Find the most recent day with a rating
        while day_num >= 0:
            if (season, team_id, day_num) in self.team_elo_ratings:
                return self.team_elo_ratings[(season, team_id, day_num)]
            day_num -= 1
            
        # If no rating found, return default
        return 1500
    
    def elo_win_probability(self, team1_elo, team2_elo, home_advantage=100, location=None):
        """Calculate win probability based on ELO ratings"""
        # Adjust for home court if specified
        if location == 'H':  # Team1 at home
            team1_elo += home_advantage
        elif location == 'A':  # Team1 away
            team2_elo += home_advantage
            
        # Calculate win probability
        return 1.0 / (1.0 + math.pow(10, (team2_elo - team1_elo) / 400.0))
    
    def calculate_advanced_team_stats(self, start_season=2003):
        """
        Calculate advanced team statistics for all seasons where detailed data is available.
        These include:
        - Offensive/Defensive Efficiency
        - Four Factors (eFG%, TOV%, ORB%, FT Rate)
        - Pace
        - Shooting percentages
        - Advanced possession-based metrics
        
        Parameters:
        start_season (int): First season to calculate advanced stats for
        
        Returns:
        dict: Dictionary with team advanced stats by season
        """
        if not self.detailed_stats_available:
            raise ValueError("Detailed stats not available. Cannot calculate advanced metrics.")
            
        print(f"Calculating advanced team stats from {start_season} to {self.current_season}...")
        
        # Get all detailed game results
        all_detailed_games = pd.concat([
            self.data['regular_season_detailed'],
            self.data['tourney_detailed'] if 'tourney_detailed' in self.data else pd.DataFrame()
        ])
        
        # Filter for seasons we want
        all_detailed_games = all_detailed_games[all_detailed_games['Season'] >= start_season]
        
        # Sort by season and day
        all_detailed_games = all_detailed_games.sort_values(['Season', 'DayNum'])
        
        # Get list of seasons
        seasons = all_detailed_games['Season'].unique()
        
        # Get list of teams
        all_teams = self.data['teams']['TeamID'].unique()
        
        # Dictionary to store advanced stats by season and team
        advanced_stats = {}

        # Process each season
        for season in seasons:
            # Get games for this season
            season_games = all_detailed_games[all_detailed_games['Season'] == season]
            
            # Dictionary to store team stats for this season
            season_stats = {}
            
            # Initialize stats for each team
            for team_id in all_teams:
                # Basic counting stats
                season_stats[team_id] = {
                    'Games': 0,
                    'Wins': 0,
                    'Losses': 0,
                    'Points': 0,
                    'PointsAllowed': 0,
                    
                    # Shooting stats
                    'FGM': 0,
                    'FGA': 0,
                    'FGM3': 0,
                    'FGA3': 0,
                    'FTM': 0,
                    'FTA': 0,
                    
                    # Rebounding
                    'OR': 0,
                    'DR': 0,
                    
                    # Other stats
                    'Ast': 0,
                    'TO': 0,
                    'Stl': 0,
                    'Blk': 0,
                    'PF': 0,
                    
                    # Opponent stats
                    'OppFGM': 0,
                    'OppFGA': 0,
                    'OppFGM3': 0,
                    'OppFGA3': 0,
                    'OppFTM': 0,
                    'OppFTA': 0,
                    'OppOR': 0,
                    'OppDR': 0,
                    'OppAst': 0,
                    'OppTO': 0,
                    'OppStl': 0,
                    'OppBlk': 0,
                    'OppPF': 0
                }

            # Process each game
            for _, game in season_games.iterrows():
                win_team_id = game['WTeamID']
                loss_team_id = game['LTeamID']
                
                # Skip if teams not in our list
                if win_team_id not in season_stats or loss_team_id not in season_stats:
                    continue
                
                # Update winning team stats
                season_stats[win_team_id]['Games'] += 1
                season_stats[win_team_id]['Wins'] += 1
                season_stats[win_team_id]['Points'] += game['WScore']
                season_stats[win_team_id]['PointsAllowed'] += game['LScore']
                
                season_stats[win_team_id]['FGM'] += game['WFGM']
                season_stats[win_team_id]['FGA'] += game['WFGA']
                season_stats[win_team_id]['FGM3'] += game['WFGM3']
                season_stats[win_team_id]['FGA3'] += game['WFGA3']
                season_stats[win_team_id]['FTM'] += game['WFTM']
                season_stats[win_team_id]['FTA'] += game['WFTA']
                
                season_stats[win_team_id]['OR'] += game['WOR']
                season_stats[win_team_id]['DR'] += game['WDR']
                
                season_stats[win_team_id]['Ast'] += game['WAst']
                season_stats[win_team_id]['TO'] += game['WTO']
                season_stats[win_team_id]['Stl'] += game['WStl']
                season_stats[win_team_id]['Blk'] += game['WBlk']
                season_stats[win_team_id]['PF'] += game['WPF']
                
                # Opponent stats for winning team
                season_stats[win_team_id]['OppFGM'] += game['LFGM']
                season_stats[win_team_id]['OppFGA'] += game['LFGA']
                season_stats[win_team_id]['OppFGM3'] += game['LFGM3']
                season_stats[win_team_id]['OppFGA3'] += game['LFGA3']
                season_stats[win_team_id]['OppFTM'] += game['LFTM']
                season_stats[win_team_id]['OppFTA'] += game['LFTA']
                season_stats[win_team_id]['OppOR'] += game['LOR']
                season_stats[win_team_id]['OppDR'] += game['LDR']
                season_stats[win_team_id]['OppAst'] += game['LAst']
                season_stats[win_team_id]['OppTO'] += game['LTO']
                season_stats[win_team_id]['OppStl'] += game['LStl']
                season_stats[win_team_id]['OppBlk'] += game['LBlk']
                season_stats[win_team_id]['OppPF'] += game['LPF']
                
                # Update losing team stats
                season_stats[loss_team_id]['Games'] += 1
                season_stats[loss_team_id]['Losses'] += 1
                season_stats[loss_team_id]['Points'] += game['LScore']
                season_stats[loss_team_id]['PointsAllowed'] += game['WScore']
                
                season_stats[loss_team_id]['FGM'] += game['LFGM']
                season_stats[loss_team_id]['FGA'] += game['LFGA']
                season_stats[loss_team_id]['FGM3'] += game['LFGM3']
                season_stats[loss_team_id]['FGA3'] += game['LFGA3']
                season_stats[loss_team_id]['FTM'] += game['LFTM']
                season_stats[loss_team_id]['FTA'] += game['LFTA']
                
                season_stats[loss_team_id]['OR'] += game['LOR']
                season_stats[loss_team_id]['DR'] += game['LDR']
                
                season_stats[loss_team_id]['Ast'] += game['LAst']
                season_stats[loss_team_id]['TO'] += game['LTO']
                season_stats[loss_team_id]['Stl'] += game['LStl']
                season_stats[loss_team_id]['Blk'] += game['LBlk']
                season_stats[loss_team_id]['PF'] += game['LPF']
                
                # Opponent stats for losing team
                season_stats[loss_team_id]['OppFGM'] += game['WFGM']
                season_stats[loss_team_id]['OppFGA'] += game['WFGA']
                season_stats[loss_team_id]['OppFGM3'] += game['WFGM3']
                season_stats[loss_team_id]['OppFGA3'] += game['WFGA3']
                season_stats[loss_team_id]['OppFTM'] += game['WFTM']
                season_stats[loss_team_id]['OppFTA'] += game['WFTA']
                season_stats[loss_team_id]['OppOR'] += game['WOR']
                season_stats[loss_team_id]['OppDR'] += game['WDR']
                season_stats[loss_team_id]['OppAst'] += game['WAst']
                season_stats[loss_team_id]['OppTO'] += game['WTO']
                season_stats[loss_team_id]['OppStl'] += game['WStl']
                season_stats[loss_team_id]['OppBlk'] += game['WBlk']
                season_stats[loss_team_id]['OppPF'] += game['WPF']

            # Calculate advanced stats for each team
            for team_id, stats in season_stats.items():
                # Skip teams with no games
                if stats['Games'] == 0:
                    continue
                    
                # Calculate shooting percentages
                stats['FG%'] = stats['FGM'] / stats['FGA'] if stats['FGA'] > 0 else 0
                stats['3P%'] = stats['FGM3'] / stats['FGA3'] if stats['FGA3'] > 0 else 0
                stats['FT%'] = stats['FTM'] / stats['FTA'] if stats['FTA'] > 0 else 0
                
                # Calculate opponent shooting percentages
                stats['OppFG%'] = stats['OppFGM'] / stats['OppFGA'] if stats['OppFGA'] > 0 else 0
                stats['Opp3P%'] = stats['OppFGM3'] / stats['OppFGA3'] if stats['OppFGA3'] > 0 else 0
                stats['OppFT%'] = stats['OppFTM'] / stats['OppFTA'] if stats['OppFTA'] > 0 else 0
                
                # Calculate effective field goal percentage (eFG%)
                # eFG% = (FGM + 0.5 * FGM3) / FGA
                stats['eFG%'] = (stats['FGM'] + 0.5 * stats['FGM3']) / stats['FGA'] if stats['FGA'] > 0 else 0
                stats['OppeFG%'] = (stats['OppFGM'] + 0.5 * stats['OppFGM3']) / stats['OppFGA'] if stats['OppFGA'] > 0 else 0
                
                # Estimate possessions (Pace)
                # Possessions = FGA - OR + TO + (0.44 * FTA)
                stats['Poss'] = stats['FGA'] - stats['OR'] + stats['TO'] + (0.44 * stats['FTA'])
                stats['OppPoss'] = stats['OppFGA'] - stats['OppOR'] + stats['OppTO'] + (0.44 * stats['OppFTA'])
                
                # Average possessions per game (Pace)
                stats['Pace'] = (stats['Poss'] + stats['OppPoss']) / (2 * stats['Games'])
                
                # Offensive and Defensive Efficiency (points per 100 possessions)
                stats['OffEff'] = 100 * stats['Points'] / stats['Poss'] if stats['Poss'] > 0 else 0
                stats['DefEff'] = 100 * stats['PointsAllowed'] / stats['OppPoss'] if stats['OppPoss'] > 0 else 0
                
                # Net Efficiency
                stats['NetEff'] = stats['OffEff'] - stats['DefEff']

                # Four Factors
                # 1. Shooting - eFG% (already calculated)
                # 2. Turnovers - Turnover Rate
                stats['TOV%'] = stats['TO'] / stats['Poss'] if stats['Poss'] > 0 else 0
                stats['OppTOV%'] = stats['OppTO'] / stats['OppPoss'] if stats['OppPoss'] > 0 else 0
                
                # 3. Rebounding - Offensive Rebounding Percentage
                total_rebounds = stats['OR'] + stats['OppDR']
                opp_total_rebounds = stats['OppOR'] + stats['DR']
                
                stats['ORB%'] = stats['OR'] / total_rebounds if total_rebounds > 0 else 0
                stats['DRB%'] = stats['DR'] / opp_total_rebounds if opp_total_rebounds > 0 else 0
                
                # 4. Free Throws - Free Throw Rate (FTA/FGA)
                stats['FTRate'] = stats['FTA'] / stats['FGA'] if stats['FGA'] > 0 else 0
                stats['OppFTRate'] = stats['OppFTA'] / stats['OppFGA'] if stats['OppFGA'] > 0 else 0
                
                # Additional metrics
                # Assist Rate (percentage of made field goals that are assisted)
                stats['AstRate'] = stats['Ast'] / stats['FGM'] if stats['FGM'] > 0 else 0
                
                # Block Rate (percentage of opponent 2-point attempts that are blocked)
                opp_2pa = stats['OppFGA'] - stats['OppFGA3']
                stats['BlkRate'] = stats['Blk'] / opp_2pa if opp_2pa > 0 else 0
                
                # Steal Rate (percentage of opponent possessions that end in a steal)
                stats['StlRate'] = stats['Stl'] / stats['OppPoss'] if stats['OppPoss'] > 0 else 0
                
                # Per-game averages
                for key in ['Points', 'PointsAllowed', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
                           'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']:
                    stats[key + 'PerGame'] = stats[key] / stats['Games'] if stats['Games'] > 0 else 0
            
            # Store stats for this season
            advanced_stats[season] = season_stats
            
        # Store in class instance
        self.advanced_team_stats = advanced_stats
        
        print(f"Calculated advanced stats for {len(advanced_stats)} seasons")
        return advanced_stats
    
    def _get_advanced_stats_features(self, season, team1_id, team2_id):
        """Get advanced stats features for both teams"""
        if not hasattr(self, 'advanced_team_stats') or not self.advanced_team_stats:
            self.calculate_advanced_team_stats()
            
        # Get stats for this season
        if season not in self.advanced_team_stats:
            # If season not available, return empty dict
            return {}
            
        season_stats = self.advanced_team_stats[season]
        
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
    
    def _get_season_stats(self, season, team1_id, team2_id):
        """Get season performance stats for both teams"""
        # Filter regular season games for this season
        season_games = self.data['regular_season'][self.data['regular_season']['Season'] == season]
        
        # Team1 stats
        team1_wins = season_games[season_games['WTeamID'] == team1_id].shape[0]
        team1_losses = season_games[season_games['LTeamID'] == team1_id].shape[0]
        team1_win_pct = team1_wins / (team1_wins + team1_losses) if (team1_wins + team1_losses) > 0 else 0
        
        # Team2 stats
        team2_wins = season_games[season_games['WTeamID'] == team2_id].shape[0]
        team2_losses = season_games[season_games['LTeamID'] == team2_id].shape[0]
        team2_win_pct = team2_wins / (team2_wins + team2_losses) if (team2_wins + team2_losses) > 0 else 0
        
        # Calculate strength of schedule
        if hasattr(self, 'advanced_team_stats') and self.advanced_team_stats and season in self.advanced_team_stats:
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
            season_stats = self.advanced_team_stats[season]
            
            team1_opp_net_eff = [season_stats.get(opp, {}).get('NetEff', 0) for opp in team1_opponents]
            team2_opp_net_eff = [season_stats.get(opp, {}).get('NetEff', 0) for opp in team2_opponents]
            
            team1_sos = np.mean(team1_opp_net_eff) if team1_opp_net_eff else 0
            team2_sos = np.mean(team2_opp_net_eff) if team2_opp_net_eff else 0
            
            # Get last 10 games performance
            team1_last_10 = []
            team2_last_10 = []
            
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
        
    def create_feature_dataset(self, train_years_range=(2010, 2024), include_elo=True, include_advanced_stats=True):
        """
        Create a dataset with features for training and prediction.
        
        Parameters:
        train_years_range (tuple): Range of years to use for training (inclusive)
        include_elo (bool): Whether to include ELO rating features
        include_advanced_stats (bool): Whether to include advanced box score stats
        """
        print("Creating feature dataset...")
        
        # Process seeds first
        self.preprocess_seeds()
        
        # Calculate ELO ratings if needed
        if include_elo and not self.team_elo_ratings:
            self.calculate_elo_ratings(start_year=min(train_years_range[0] - 2, 2003))
            
        # Calculate advanced stats if needed
        if include_advanced_stats and (not hasattr(self, 'advanced_team_stats') or not self.advanced_team_stats):
            self.calculate_advanced_team_stats(start_season=min(train_years_range[0], 2003))
        
        # Get all possible tournament matchups from historical data
        tourney_games = self.data['tourney_results'].copy()
        
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
            if self.rankings_available:
                game_features.update(self._get_ranking_features(season, team1_id, team2_id))
                
            # Add ELO rating features if available
            if include_elo and self.team_elo_ratings:
                # Get ELO ratings just before this tournament game
                # We use day_num - 1 to ensure we don't leak future information
                team1_elo = self.get_team_elo(season, team1_id, day_num - 1)
                team2_elo = self.get_team_elo(season, team2_id, day_num - 1)
                
                # Calculate win probability
                elo_win_prob = self.elo_win_probability(team1_elo, team2_elo)
                
                game_features.update({
                    'Team1ELO': team1_elo,
                    'Team2ELO': team2_elo,
                    'ELODiff': team1_elo - team2_elo,
                    'ELOWinProb': elo_win_prob
                })
                
            # Add advanced stats features if available
            if include_advanced_stats and hasattr(self, 'advanced_team_stats') and self.advanced_team_stats:
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
        self.feature_cols = [col for col in self.feature_df.columns 
                            if col not in ['Result', 'Season', 'Team1ID', 'Team2ID']]
        
        return self.feature_df
    
    def train_model(self):
        """Train a model using the feature dataset"""
        if not hasattr(self, 'feature_df'):
            raise ValueError("Feature dataset not created. Call create_feature_dataset() first.")
            
        # Split features and target
        X = self.feature_df[self.feature_cols]
        y = self.feature_df['Result']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,  # Limit tree depth
            min_samples_leaf=5,  # Require at least 5 samples per leaf
            min_samples_split=10,  # Require at least 10 samples to split a node
            max_features='sqrt',  # Use sqrt(n_features) features per split
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_preds = self.model.predict_proba(X_train)[:, 1]
        test_preds = self.model.predict_proba(X_test)[:, 1]
        
        train_accuracy = accuracy_score(y_train, train_preds > 0.5)
        test_accuracy = accuracy_score(y_test, test_preds > 0.5)
        
        train_log_loss = log_loss(y_train, train_preds)
        test_log_loss = log_loss(y_test, test_preds)
        
        print(f"Train accuracy: {train_accuracy:.4f}, Log loss: {train_log_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}, Log loss: {test_log_loss:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 important features:")
        print(feature_importance.head(10))
        
        return self.model
    
    def generate_predictions(self, submission_file='submission.csv', blend_elo=True, elo_weight=0.3):
        """
        Generate predictions for the current tournament
        
        Parameters:
        submission_file (str): Path to save the submission file
        blend_elo (bool): Whether to blend the model predictions with ELO predictions
        elo_weight (float): Weight to give ELO predictions when blending (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        # Get current season seeds
        current_seeds = self.data['processed_seeds'][self.data['processed_seeds']['Season'] == self.current_season]
        
        if len(current_seeds) == 0:
            raise ValueError(f"No seed data found for season {self.current_season}")
            
        # Ensure we have ELO ratings if blending is requested
        if blend_elo and not self.team_elo_ratings:
            print("ELO ratings not found. Calculating now...")
            self.calculate_elo_ratings()
            
        # Make sure we have advanced stats if they were used in training
        if 'Team1_OffEff' in self.feature_cols and (not hasattr(self, 'advanced_team_stats') or not self.advanced_team_stats):
            print("Advanced stats not found. Calculating now...")
            self.calculate_advanced_team_stats()
            
        # Generate all possible matchups
        team_ids = current_seeds['TeamID'].unique()
        matchups = []
        
        for i, team1_id in enumerate(team_ids):
            team1_seed = self.seed_lookup.get((self.current_season, team1_id), 16)
            for team2_id in team_ids[i+1:]:
                team2_seed = self.seed_lookup.get((self.current_season, team2_id), 16)
                
                # Create ID in required format
                matchup_id = f"{self.current_season}_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"
                
                # Prepare features for prediction
                if team1_id < team2_id:
                    features = {
                        'Team1ID': team1_id,
                        'Team2ID': team2_id,
                        'Team1Seed': team1_seed,
                        'Team2Seed': team2_seed,
                        'SeedDiff': team2_seed - team1_seed
                    }
                    team1_is_first = True
                else:
                    features = {
                        'Team1ID': team2_id,
                        'Team2ID': team1_id,
                        'Team1Seed': team2_seed,
                        'Team2Seed': team1_seed, 
                        'SeedDiff': team1_seed - team2_seed
                    }
                    team1_is_first = False
                
                # Add season stats
                features.update(self._get_season_stats(self.current_season, 
                                                     features['Team1ID'], 
                                                     features['Team2ID']))
                
                # Add ranking features if available
                if self.rankings_available:
                    features.update(self._get_ranking_features(self.current_season, 
                                                             features['Team1ID'], 
                                                             features['Team2ID']))
                
                # Add ELO rating features if available and using ELO in model
                if self.team_elo_ratings:
                    team1_elo = self.get_team_elo(self.current_season, features['Team1ID'])
                    team2_elo = self.get_team_elo(self.current_season, features['Team2ID'])
                    elo_win_prob = self.elo_win_probability(team1_elo, team2_elo)
                    
                    if 'Team1ELO' in self.feature_cols:
                        features.update({
                            'Team1ELO': team1_elo,
                            'Team2ELO': team2_elo,
                            'ELODiff': team1_elo - team2_elo,
                            'ELOWinProb': elo_win_prob
                        })
                        
                # Add advanced stats features if available and using them in model
                if hasattr(self, 'advanced_team_stats') and self.advanced_team_stats:
                    adv_features = self._get_advanced_stats_features(self.current_season, 
                                                                  features['Team1ID'], 
                                                                  features['Team2ID'])
                    features.update(adv_features)
                
                # Extract just the model features
                X_pred = pd.DataFrame([{col: features.get(col, 0) for col in self.feature_cols}])
                
                # Get model prediction
                model_pred = self.model.predict_proba(X_pred)[0, 1]
                
                # Blend with ELO if requested
                if blend_elo and self.team_elo_ratings:
                    # Get ELO-based prediction
                    team1_elo = self.get_team_elo(self.current_season, features['Team1ID'])
                    team2_elo = self.get_team_elo(self.current_season, features['Team2ID'])
                    elo_pred = self.elo_win_probability(team1_elo, team2_elo)
                    
                    # Blend predictions
                    pred = (1 - elo_weight) * model_pred + elo_weight * elo_pred
                else:
                    pred = model_pred
                
                # If team1 is not the first ID in the matchup_id, flip the prediction
                if not team1_is_first:
                    pred = 1 - pred
                
                matchups.append({
                    'ID': matchup_id,
                    'Pred': pred
                })
        
        # Create submission DataFrame
        submission_df = pd.DataFrame(matchups)
        
        # Save to CSV
        submission_df.to_csv(submission_file, index=False)
        print(f"Saved {len(submission_df)} predictions to {submission_file}")
        
        return submission_df
    

    def analyze_team_chances(self, team_id):
        """Analyze a specific team's chances against all other tournament teams"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Get current season seeds
        current_seeds = self.data['processed_seeds'][self.data['processed_seeds']['Season'] == self.current_season]
        
        if len(current_seeds) == 0:
            raise ValueError(f"No seed data found for season {self.current_season}")
        
        # Get team info
        team_info = self.data['teams'][self.data['teams']['TeamID'] == team_id]
        if len(team_info) == 0:
            raise ValueError(f"Team ID {team_id} not found")
            
        team_name = team_info.iloc[0]['TeamName']
        team_seed = self.seed_lookup.get((self.current_season, team_id), "Unknown")
        
        print(f"Analyzing {team_name} (Seed: {team_seed})")
        
        # Get all other tournament teams
        other_teams = current_seeds[current_seeds['TeamID'] != team_id]
        
        results = []
        for _, other_team in other_teams.iterrows():
            other_id = other_team['TeamID']
            other_name = self.data['teams'][self.data['teams']['TeamID'] == other_id].iloc[0]['TeamName']
            other_seed = other_team['SeedNumber']
            
            # Prepare features for prediction (team_id as Team1)
            features = {
                'Team1ID': team_id,
                'Team2ID': other_id,
                'Team1Seed': team_seed if isinstance(team_seed, int) else 16,
                'Team2Seed': other_seed,
                'SeedDiff': other_seed - (team_seed if isinstance(team_seed, int) else 16)
            }
            
            # Add season stats
            features.update(self._get_season_stats(self.current_season, team_id, other_id))
            
            # Add ranking features if available
            if self.rankings_available:
                features.update(self._get_ranking_features(self.current_season, team_id, other_id))
                
            # Add ELO rating features if available
            if hasattr(self, 'team_elo_ratings') and self.team_elo_ratings:
                team1_elo = self.get_team_elo(self.current_season, team_id)
                team2_elo = self.get_team_elo(self.current_season, other_id)
                elo_win_prob = self.elo_win_probability(team1_elo, team2_elo)
                
                features.update({
                    'Team1ELO': team1_elo,
                    'Team2ELO': team2_elo,
                    'ELODiff': team1_elo - team2_elo,
                    'ELOWinProb': elo_win_prob
                })
                
            # Add advanced stats features if available
            if hasattr(self, 'advanced_team_stats') and self.advanced_team_stats:
                adv_features = self._get_advanced_stats_features(self.current_season, team_id, other_id)
                features.update(adv_features)
            
            # Extract just the model features
            X_pred = pd.DataFrame([{col: features.get(col, 0) for col in self.feature_cols}])
            
            # Get prediction
            win_prob = self.model.predict_proba(X_pred)[0, 1]
            
            results.append({
                'OpponentID': other_id,
                'OpponentName': other_name,
                'OpponentSeed': other_seed,
                'WinProbability': win_prob
            })
        
        # Create and sort DataFrame
        results_df = pd.DataFrame(results).sort_values('WinProbability', ascending=False)
        
        # Display results
        print(f"\nWin probabilities for {team_name}:")
        print(results_df[['OpponentName', 'OpponentSeed', 'WinProbability']].head(10))
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='WinProbability', y='OpponentName', 
                   hue='OpponentSeed', data=results_df.head(15), palette='viridis')
        plt.title(f"Win Probabilities for {team_name}")
        plt.xlabel('Probability')
        plt.ylabel('Opponent')
        plt.tight_layout()
        plt.show()
        
        return results_df

    def visualize_elo_history(self, team_ids, seasons=None, title="Team ELO Rating History"):
        """
        Visualize ELO rating history for selected teams.
        
        Parameters:
        team_ids (list): List of TeamIDs to visualize
        seasons (list, optional): List of seasons to include. If None, use all available.
        title (str): Plot title
        """
        if not self.team_elo_ratings:
            raise ValueError("ELO ratings not calculated. Call calculate_elo_ratings() first.")
            
        # Get team names
        team_names = {}
        for team_id in team_ids:
            team_info = self.data['teams'][self.data['teams']['TeamID'] == team_id]
            if len(team_info) > 0:
                team_names[team_id] = team_info.iloc[0]['TeamName']
            else:
                team_names[team_id] = f"Team {team_id}"
                
        # Extract ELO history
        elo_history = defaultdict(list)
        
        for (season, team_id, day_num), rating in self.team_elo_ratings.items():
            if team_id in team_ids:
                if seasons is None or season in seasons:
                    elo_history[(season, team_id)].append((day_num, rating))
        
        # Setup plot
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10.colors
        
        # Plot each team's rating over time
        for i, team_id in enumerate(team_ids):
            color = colors[i % len(colors)]
            team_name = team_names[team_id]
            
            for season in sorted(set(s for (s, t), _ in elo_history.items() if t == team_id)):
                # Get data for this team and season
                data = sorted(elo_history[(season, team_id)])
                if data:
                    days, ratings = zip(*data)
                    
                    # Plot with season label for first point only
                    if i == 0:  # Only label seasons for the first team to avoid clutter
                        plt.plot(days, ratings, '-', color=color, alpha=0.7, 
                                linewidth=2, label=f"{season}")
                    else:
                        plt.plot(days, ratings, '-', color=color, alpha=0.7, linewidth=2)
            
            # Add a dummy line for the team legend
            plt.plot([], [], '-', color=color, linewidth=3, label=team_name)
            
        # Add NCAA tournament markers
        plt.axvspan(132, 154, color='lightgray', alpha=0.3, label='NCAA Tournament')
        
        # Formatting
        plt.xlabel('Day Number')
        plt.ylabel('ELO Rating')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Create two legends
        handles, labels = plt.gca().get_legend_handles_labels()
        
        # Split into teams and seasons
        team_handles = [h for h, l in zip(handles, labels) if not l.isdigit() and l != 'NCAA Tournament']
        team_labels = [l for l in labels if not l.isdigit() and l != 'NCAA Tournament']
        
        season_handles = [h for h, l in zip(handles, labels) if l.isdigit()]
        season_labels = [l for l in labels if l.isdigit()]
        
        tournament_handles = [h for h, l in zip(handles, labels) if l == 'NCAA Tournament']
        tournament_labels = ['NCAA Tournament'] if tournament_handles else []
        
        # Place legends
        if team_handles:
            plt.legend(team_handles, team_labels, loc='upper left', title='Teams')
            
        if season_handles:
            plt.legend(season_handles + tournament_handles, 
                      season_labels + tournament_labels, 
                      loc='upper right', title='Seasons')
            
        plt.tight_layout()
        plt.show()
        
    def visualize_advanced_stats(self, season, teams=None, metric='OffEff', title=None):
        """
        Visualize advanced stats for teams in a specific season.
        
        Parameters:
        season (int): Season to analyze
        teams (list, optional): List of TeamIDs to highlight. If None, will show all tournament teams.
        metric (str): Metric to visualize. Options include 'OffEff', 'DefEff', 'NetEff', 'eFG%', etc.
        title (str, optional): Custom plot title. If None, will generate based on metric.
        """
        if not hasattr(self, 'advanced_team_stats') or not self.advanced_team_stats:
            self.calculate_advanced_team_stats()
            
        if season not in self.advanced_team_stats:
            raise ValueError(f"No advanced stats available for season {season}")
            
        # Get tournament teams for this season
        tourney_teams = set()
        if season in self.data['tourney_seeds']['Season'].values:
            tourney_teams = set(self.data['tourney_seeds'][self.data['tourney_seeds']['Season'] == season]['TeamID'])
            
        # Get team names
        team_names = {}
        for team_id in self.data['teams']['TeamID']:
            team_info = self.data['teams'][self.data['teams']['TeamID'] == team_id]
            if len(team_info) > 0:
                team_names[team_id] = team_info.iloc[0]['TeamName']
                
        # Prepare data for visualization
        season_stats = self.advanced_team_stats[season]
        
        # Filter teams to show
        if teams is None:
            # Show all tournament teams
            teams_to_show = tourney_teams
        else:
            teams_to_show = set(teams)
            
        # Gather data for selected teams
        viz_data = []
        
        # Also include top 25 teams by the metric if teams list is short
        if teams is not None and len(teams) < 10:
            # Get top teams by the metric
            all_teams_by_metric = [(team_id, stats.get(metric, 0)) 
                                   for team_id, stats in season_stats.items()]
            all_teams_by_metric.sort(key=lambda x: x[1], reverse=True)
            top_teams = [team_id for team_id, _ in all_teams_by_metric[:25]]
            teams_to_show = teams_to_show.union(set(top_teams))
        
        for team_id, stats in season_stats.items():
            if team_id in teams_to_show and metric in stats:
                team_name = team_names.get(team_id, f"Team {team_id}")
                seed = None
                if team_id in tourney_teams:
                    seed_info = self.data['tourney_seeds'][
                        (self.data['tourney_seeds']['Season'] == season) & 
                        (self.data['tourney_seeds']['TeamID'] == team_id)
                    ]
                    if len(seed_info) > 0:
                        seed = int(seed_info.iloc[0]['Seed'][1:3])
                
                viz_data.append({
                    'TeamID': team_id,
                    'TeamName': team_name,
                    'Seed': seed,
                    'Value': stats.get(metric, 0),
                    'InTourney': team_id in tourney_teams,
                    'Highlighted': teams is not None and team_id in teams
                })
        
        # Create DataFrame and sort by the metric value
        viz_df = pd.DataFrame(viz_data)
        viz_df = viz_df.sort_values('Value', ascending=False)
        
        # Set up plot
        plt.figure(figsize=(12, max(8, len(viz_df) * 0.25)))
        
        # Create bars
        bar_colors = []
        for _, row in viz_df.iterrows():
            if row['Highlighted']:
                bar_colors.append('gold')
            elif row['InTourney']:
                bar_colors.append('skyblue')
            else:
                bar_colors.append('lightgray')
                
        # Plot
        bars = plt.barh(viz_df['TeamName'], viz_df['Value'], color=bar_colors)
        
        # Add team seeds for tournament teams
        for i, (_, row) in enumerate(viz_df.iterrows()):
            if row['Seed'] is not None:
                plt.text(row['Value'] + 0.5, i, f"#{row['Seed']}", 
                        verticalalignment='center', fontsize=8)
        
        # Set title and labels
        if title is None:
            if metric == 'OffEff':
                title = f"{season} Offensive Efficiency (Points per 100 Possessions)"
            elif metric == 'DefEff':
                title = f"{season} Defensive Efficiency (Points Allowed per 100 Possessions)"
            elif metric == 'NetEff':
                title = f"{season} Net Efficiency (Off-Def per 100 Possessions)"
            else:
                title = f"{season} {metric} by Team"
                
        plt.title(title)
        plt.xlabel(metric)
        plt.ylabel('Team')
        plt.grid(axis='x', alpha=0.3)
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gold', label='Highlighted Teams'),
            Patch(facecolor='skyblue', label='Tournament Teams'),
            Patch(facecolor='lightgray', label='Other Teams')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.show()
        
        # Return the data for further analysis
        return viz_df

    def four_factors_breakdown(self, season, team_id):
        """
        Visualize the Four Factors breakdown for a specific team.
        
        Parameters:
        season (int): Season to analyze
        team_id (int): TeamID to analyze
        """
        if not hasattr(self, 'advanced_team_stats') or not self.advanced_team_stats:
            self.calculate_advanced_team_stats()
            
        if season not in self.advanced_team_stats:
            raise ValueError(f"No advanced stats available for season {season}")
            
        # Get team stats
        season_stats = self.advanced_team_stats[season]
        
        if team_id not in season_stats:
            raise ValueError(f"No stats for team {team_id} in season {season}")
            
        team_stats = season_stats[team_id]
        
        # Get team name
        team_info = self.data['teams'][self.data['teams']['TeamID'] == team_id]
        team_name = team_info.iloc[0]['TeamName'] if len(team_info) > 0 else f"Team {team_id}"
        
        # Calculate league averages
        league_averages = {}
        factors = ['eFG%', 'TOV%', 'ORB%', 'FTRate']
        
        for factor in factors:
            values = [stats.get(factor, 0) for stats in season_stats.values() if factor in stats]
            league_averages[factor] = np.mean(values) if values else 0
            
        # Prepare data for the radar chart
        categories = ['Shooting\n(eFG%)', 'Ball Control\n(1-TOV%)', 'Rebounding\n(ORB%)', 'Free Throws\n(FTRate)']
        
        # Note: We use 1-TOV% because lower turnover rates are better
        values = [
            team_stats.get('eFG%', 0),
            1 - team_stats.get('TOV%', 0),
            team_stats.get('ORB%', 0),
            team_stats.get('FTRate', 0)
        ]
        
        # League average values
        league_values = [
            league_averages.get('eFG%', 0),
            1 - league_averages.get('TOV%', 0),
            league_averages.get('ORB%', 0),
            league_averages.get('FTRate', 0)
        ]
        
        # Standardize values relative to league average
        # 1.0 means exactly league average
        relative_values = [v / l if l > 0 else 1.0 for v, l in zip(values, league_values)]
        
        # Set up the radar chart
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        relative_values += relative_values[:1]  # Close the loop
        league_values_norm = [1.0] * len(categories) + [1.0]  # League average reference (always 1.0)
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Plot team values
        ax.plot(angles, relative_values, 'o-', linewidth=2, label=f'{team_name}')
        # Plot league average reference
        ax.plot(angles, league_values_norm, '--', color='gray', alpha=0.7, linewidth=1, label='League Average')
        
        # Fill team area
        ax.fill(angles, relative_values, alpha=0.25)
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Set radial limits
        ax.set_ylim(0, max(max(relative_values), 1.5))
        
        # Add grid and legend
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title(f"{season} Four Factors: {team_name}\n(Relative to League Average)", size=15)
        
        # Add annotations for actual values
        for i, (angle, value, label, rel_value) in enumerate(zip(angles[:-1], values, categories, relative_values[:-1])):
            ax.annotate(f"{value:.3f}", 
                      xy=(angle, rel_value),
                      xytext=(angle, rel_value + 0.1),
                      ha='center',
                      va='center')
        
        plt.tight_layout()
        plt.show()
        
        # Return the raw data
        return {
            'team_name': team_name,
            'categories': categories,
            'team_values': values,
            'league_averages': league_values,
            'relative_values': relative_values[:-1]  # Exclude the repeated value
        }
        
    def analyze_elo_factors(self, k_values=[10, 20, 30], carry_over_values=[0.5, 0.75, 0.9]):
        """
        Analyze how different ELO parameters affect predictive performance.
        
        Parameters:
        k_values (list): Different k-factor values to test
        carry_over_values (list): Different season-to-season carryover factors to test
        """
        # Get tournament games for testing
        test_games = self.data['tourney_results'].copy()
        test_games = test_games[test_games['Season'] >= 2015]  # Use recent seasons
        
        results = []
        
        for k in k_values:
            for carry_over in carry_over_values:
                # Calculate ELO ratings with these parameters
                self.calculate_elo_ratings(k_factor=k, carry_over_factor=carry_over)
                
                # Test accuracy on tournament games
                correct = 0
                total = 0
                
                for _, game in test_games.iterrows():
                    season = game['Season']
                    w_team = game['WTeamID']
                    l_team = game['LTeamID']
                    day_num = game['DayNum']
                    
                    # Get ELO ratings before the game
                    w_elo = self.get_team_elo(season, w_team, day_num - 1)
                    l_elo = self.get_team_elo(season, l_team, day_num - 1)
                    
                    # Predict winner based on ELO
                    predicted_winner = w_team if w_elo > l_elo else l_team
                    
                    # Check if prediction was correct
                    if predicted_winner == w_team:
                        correct += 1
                    
                    total += 1
                
                # Calculate accuracy
                accuracy = correct / total if total > 0 else 0
                
                results.append({
                    'k_factor': k,
                    'carry_over': carry_over,
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total
                })
                
        # Convert to DataFrame for easy analysis
        results_df = pd.DataFrame(results)
        
        # Plot results
        plt.figure(figsize=(10, 6))
        
        # Create pivot table for heatmap
        pivot_data = results_df.pivot(index='k_factor', columns='carry_over', values='accuracy')
        
        # Plot heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis')
        plt.title('Tournament Prediction Accuracy by ELO Parameters')
        plt.xlabel('Carry Over Factor')
        plt.ylabel('K Factor')
        plt.tight_layout()
        plt.show()
        
        return results_df
    
    def backtest_tournament(self, test_season, train_seasons=None, include_elo=True, 
                          include_advanced_stats=True, blend_elo=True, elo_weight=0.3):
        """
        Backtest the model on a historical tournament.
        
        Parameters:
        test_season (int): Season to use as the test set
        train_seasons (list, optional): List of seasons to use for training. If None, use all seasons before test_season.
        include_elo (bool): Whether to include ELO features in the model
        include_advanced_stats (bool): Whether to include advanced stats in the model
        blend_elo (bool): Whether to blend model predictions with ELO predictions
        elo_weight (float): Weight to give ELO predictions when blending (0-1)
        
        Returns:
        dict: Evaluation metrics including Brier score, accuracy, and log loss
        """
        print(f"Backtesting on {test_season} tournament...")
        
        # Set up train seasons if not provided
        if train_seasons is None:
            # Use all seasons before test_season
            all_seasons = sorted(self.data['tourney_results']['Season'].unique())
            train_seasons = [s for s in all_seasons if s < test_season]
        
        # Temporary save original current_season
        original_season = self.current_season
        self.current_season = test_season
        
        # Calculate ELO ratings if needed (using all historical data up to test_season)
        if include_elo and not self.team_elo_ratings:
            self.calculate_elo_ratings(start_year=min(train_seasons) - 2)
            
        # Calculate advanced stats if needed
        if include_advanced_stats and (not hasattr(self, 'advanced_team_stats') or not self.advanced_team_stats):
            self.calculate_advanced_team_stats(start_season=min(train_seasons))
        
        # Create feature dataset for training (excluding test_season)
        train_years_range = (min(train_seasons), max(train_seasons))
        self.create_feature_dataset(train_years_range=train_years_range,
                                    include_elo=include_elo,
                                    include_advanced_stats=include_advanced_stats)
        
        # Train the model on historical data (excluding test_season)
        self.train_model()
        
        # Get the actual tournament games for test_season
        test_games = self.data['tourney_results'][self.data['tourney_results']['Season'] == test_season].copy()
        # Add this at the beginning of the backtest_tournament method
        if len(test_games) == 0:
            print(f"ERROR: No test games found for season {test_season}")
            return {
                'season': test_season,
                'train_seasons': train_seasons,
                'num_games': 0,
                'brier_score': float('nan'),
                'accuracy': float('nan'),
                'log_loss': float('nan'),
                'error': 'No test games found'
            }
        
        # Process seeds for test_season
        self.preprocess_seeds()
        
        # Create predictions for each game
        predictions = []
        actual_results = []
        
        for _, game in test_games.iterrows():
            team1_id = game['WTeamID']  # Winner
            team2_id = game['LTeamID']  # Loser
            day_num = game['DayNum']
            
            # Get seed information
            team1_seed = self.seed_lookup.get((test_season, team1_id), 16)
            team2_seed = self.seed_lookup.get((test_season, team2_id), 16)
            
            # Create features for prediction
            features = {
                'Team1ID': team1_id,
                'Team2ID': team2_id,
                'Team1Seed': team1_seed,
                'Team2Seed': team2_seed,
                'SeedDiff': team2_seed - team1_seed
            }
            
            # Add season stats
            features.update(self._get_season_stats(test_season, team1_id, team2_id))
            
            # Add ranking features if available
            if self.rankings_available:
                features.update(self._get_ranking_features(test_season, team1_id, team2_id))
                
            # Add ELO rating features if available
            # Important: Use day_num - 1 to avoid future information leakage
            if include_elo and self.team_elo_ratings:
                team1_elo = self.get_team_elo(test_season, team1_id, day_num - 1)
                team2_elo = self.get_team_elo(test_season, team2_id, day_num - 1)
                elo_win_prob = self.elo_win_probability(team1_elo, team2_elo)
                
                features.update({
                    'Team1ELO': team1_elo,
                    'Team2ELO': team2_elo,
                    'ELODiff': team1_elo - team2_elo,
                    'ELOWinProb': elo_win_prob
                })
                
            # Add advanced stats features if available
            if include_advanced_stats and hasattr(self, 'advanced_team_stats') and self.advanced_team_stats:
                features.update(self._get_advanced_stats_features(test_season, team1_id, team2_id))
            
            # Extract just the model features
            X_pred = pd.DataFrame([{col: features.get(col, 0) for col in self.feature_cols}])
            
            # Get model prediction
            model_pred = self.model.predict_proba(X_pred)[0, 1]
            
            # Blend with ELO if requested
            if blend_elo and include_elo and self.team_elo_ratings:
                elo_pred = features.get('ELOWinProb', 0.5)  # Use previously calculated ELO prediction
                pred = (1 - elo_weight) * model_pred + elo_weight * elo_pred
            else:
                pred = model_pred
                
            # Store prediction and actual result
            predictions.append(pred)
            actual_results.append(1)  # Team1 won (WTeamID is always the winner)
            
            # Also add the reversed matchup for evaluation
            features_reversed = {
                'Team1ID': team2_id,
                'Team2ID': team1_id,
                'Team1Seed': team2_seed,
                'Team2Seed': team1_seed,
                'SeedDiff': team1_seed - team2_seed
            }
            
            # Add season stats for reversed
            features_reversed.update(self._get_season_stats(test_season, team2_id, team1_id))
            
            # Add ranking features if available
            if self.rankings_available:
                features_reversed.update(self._get_ranking_features(test_season, team2_id, team1_id))
                
            # Add ELO rating features if available
            if include_elo and self.team_elo_ratings:
                team2_elo = self.get_team_elo(test_season, team2_id, day_num - 1)
                team1_elo = self.get_team_elo(test_season, team1_id, day_num - 1)
                elo_win_prob_reversed = self.elo_win_probability(team2_elo, team1_elo)
                
                features_reversed.update({
                    'Team1ELO': team2_elo,
                    'Team2ELO': team1_elo,
                    'ELODiff': team2_elo - team1_elo,
                    'ELOWinProb': elo_win_prob_reversed
                })
                
            # Add advanced stats features if available
            if include_advanced_stats and hasattr(self, 'advanced_team_stats') and self.advanced_team_stats:
                features_reversed.update(self._get_advanced_stats_features(test_season, team2_id, team1_id))
            
            # Extract just the model features
            X_pred_reversed = pd.DataFrame([{col: features_reversed.get(col, 0) for col in self.feature_cols}])
            
            # Get model prediction
            model_pred_reversed = self.model.predict_proba(X_pred_reversed)[0, 1]
            
            # Blend with ELO if requested
            if blend_elo and include_elo and self.team_elo_ratings:
                elo_pred_reversed = features_reversed.get('ELOWinProb', 0.5)
                pred_reversed = (1 - elo_weight) * model_pred_reversed + elo_weight * elo_pred_reversed
            else:
                pred_reversed = model_pred_reversed
                
            # Store prediction and actual result (reversed)
            predictions.append(pred_reversed)
            actual_results.append(0)  # Team1 lost (Team2 was the winner)
        
        if len(predictions) == 0:
            print(f"ERROR: No predictions were generated for season {test_season}")
            return {
                'season': test_season,
                'train_seasons': train_seasons,
                'num_games': len(test_games),
                'brier_score': float('nan'),
                'accuracy': float('nan'),
                'log_loss': float('nan'),
                'error': 'No predictions generated'
            }

        # Calculate evaluation metrics
        
        # Brier score (mean squared error)
        
        brier_score = float('inf')
        try:
            brier_score = np.mean((np.array(predictions) - np.array(actual_results)) ** 2)
            if np.isnan(brier_score):
                print(f"WARNING: Brier score calculation resulted in NaN")
                brier_score = float('inf')  # Use infinity instead of NaN
        except Exception as e:
            print(f"ERROR in Brier score calculation: {e}")
        
        # Accuracy
        predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]
        accuracy = np.mean([1 if p == a else 0 for p, a in zip(predicted_labels, actual_results)])
        
        # Log loss 
        epsilon = 1e-15  # To avoid log(0)
        predictions_clipped = [max(min(p, 1-epsilon), epsilon) for p in predictions]
        log_loss_value = -np.mean([a * np.log(p) + (1-a) * np.log(1-p) 
                                 for a, p in zip(actual_results, predictions_clipped)])
        
        # Create evaluation result
        eval_result = {
            'season': test_season,
            'train_seasons': train_seasons,
            'num_games': len(test_games),
            'brier_score': brier_score,
            'accuracy': accuracy,
            'log_loss': log_loss_value
        }
        
        # Print results
        print(f"Backtesting results for {test_season} tournament:")
        print(f"Brier Score: {brier_score:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Log Loss: {log_loss_value:.4f}")
        
        # Restore original current_season
        self.current_season = original_season
        
        return eval_result

    def backtest_multiple_seasons(self, test_seasons=None, lookback_years=5, 
                             include_elo=True, include_advanced_stats=True,
                             blend_elo=True, elo_weight=0.3, 
                             test_elo_weights=None, visualize=True):
        """
        Run backtesting on multiple tournament seasons to evaluate model stability.
        
        Parameters:
        test_seasons (list): List of seasons to backtest on. If None, use last 10 available seasons.
        lookback_years (int): Number of previous years to use for training for each test season
        include_elo (bool): Whether to include ELO features in the model
        include_advanced_stats (bool): Whether to include advanced stats in the model
        blend_elo (bool): Whether to blend model predictions with ELO predictions
        elo_weight (float): Default weight to give ELO predictions when blending
        test_elo_weights (list): If provided, test different ELO weights for each season
        visualize (bool): Whether to create visualization of results
        
        Returns:
        dict: Aggregated results and per-season metrics
        """
        # Get available seasons
        available_seasons = sorted(self.data['tourney_results']['Season'].unique())
        
        # Set test_seasons if not provided
        if test_seasons is None:
            # Use last 10 available tournament seasons
            test_seasons = available_seasons[-10:]
        
        # Initialize results
        results = []
        
        # If testing different ELO weights, prepare weight values
        if test_elo_weights is None:
            # Use same weight for all seasons
            test_weights = {season: elo_weight for season in test_seasons}
        else:
            # Create different configurations to test
            test_weights = {}
            for i, season in enumerate(test_seasons):
                # If we have fewer weights than seasons, cycle through the weights
                weight_idx = i % len(test_elo_weights)
                test_weights[season] = test_elo_weights[weight_idx]
        
        # Backtest each season
        for season in test_seasons:
            # Identify training seasons (lookback_years previous seasons)
            train_seasons = [s for s in available_seasons 
                             if s < season and s >= season - lookback_years]
            
            if len(train_seasons) == 0:
                print(f"Warning: No training data available for season {season}. Skipping.")
                continue
                
            # Get weight for this season
            season_elo_weight = test_weights[season]
            
            # Run backtest for this season
            season_result = self.backtest_tournament(
                test_season=season,
                train_seasons=train_seasons,
                include_elo=include_elo,
                include_advanced_stats=include_advanced_stats,
                blend_elo=blend_elo,
                elo_weight=season_elo_weight
            )
            
            # Add this season's weight to the results
            season_result['elo_weight'] = season_elo_weight
            
            # Store results
            results.append(season_result)
            
        # Calculate aggregate metrics
        if results:
            avg_brier = np.mean([r['brier_score'] for r in results if not np.isnan(r['brier_score'])])
            avg_accuracy = np.mean([r['accuracy'] for r in results if not np.isnan(r['accuracy'])])
            avg_log_loss = np.mean([r['log_loss'] for r in results if not np.isnan(r['log_loss'])])

            print(f"\nOut of {len(results)} test seasons, {len([r for r in results if np.isnan(r['brier_score'])])} have NaN scores.)")
            
            print("\nAggregate results across all test seasons:")
            print(f"Average Brier Score: {avg_brier:.4f}")
            print(f"Average Accuracy: {avg_accuracy:.4f}")
            print(f"Average Log Loss: {avg_log_loss:.4f}")
            
            # Create visualization if requested
            if visualize and results:
                self._visualize_backtest_results(results)
                
            return {
                'aggregate': {
                    'brier_score': avg_brier,
                    'accuracy': avg_accuracy,
                    'log_loss': avg_log_loss,
                    'num_seasons': len(results)
                },
                'per_season': results
            }
        else:
            print("No valid backtest results to report.")
            return None
    
    def _visualize_backtest_results(self, results):
        """
        Create visualizations of backtesting results.
        
        Parameters:
        results (list): List of dictionaries with backtesting results
        """
        # Sort results by season
        results = sorted(results, key=lambda x: x['season'])
        
        # Extract seasons and metrics
        seasons = [r['season'] for r in results]
        brier_scores = [r['brier_score'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        log_losses = [r['log_loss'] for r in results]
        
        # If we've tested different ELO weights, include that information
        elo_weights = [r.get('elo_weight', 'N/A') for r in results]
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Plot Brier Score
        axes[0].plot(seasons, brier_scores, 'o-', color='blue', linewidth=2)
        axes[0].set_ylabel('Brier Score')
        axes[0].set_title('Backtesting Results by Season')
        axes[0].grid(True, alpha=0.3)
        
        # Add ELO weight annotations if relevant
        if len(set(elo_weights)) > 1:
            for i, (season, score, weight) in enumerate(zip(seasons, brier_scores, elo_weights)):
                axes[0].annotate(f"w={weight}", 
                               (season, score),
                               textcoords="offset points",
                               xytext=(0, 10),
                               ha='center')
        
        # Plot Accuracy
        axes[1].plot(seasons, accuracies, 'o-', color='green', linewidth=2)
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True, alpha=0.3)
        
        # Plot Log Loss
        axes[2].plot(seasons, log_losses, 'o-', color='red', linewidth=2)
        axes[2].set_xlabel('Season')
        axes[2].set_ylabel('Log Loss')
        axes[2].grid(True, alpha=0.3)
        
        # Format x-axis to show years clearly
        axes[2].set_xticks(seasons)
        
        # Add aggregate metrics as horizontal lines
        avg_brier = np.mean(brier_scores)
        avg_accuracy = np.mean(accuracies)
        avg_log_loss = np.mean(log_losses)
        
        axes[0].axhline(y=avg_brier, color='blue', linestyle='--', alpha=0.5, 
                       label=f'Avg: {avg_brier:.4f}')
        axes[1].axhline(y=avg_accuracy, color='green', linestyle='--', alpha=0.5,
                       label=f'Avg: {avg_accuracy:.4f}')
        axes[2].axhline(y=avg_log_loss, color='red', linestyle='--', alpha=0.5,
                       label=f'Avg: {avg_log_loss:.4f}')
        
        # Add legends
        for ax in axes:
            ax.legend()
            
        plt.tight_layout()
        plt.show()
        
        # If we tested different ELO weights, create an ELO weight optimization chart
        if len(set(elo_weights)) > 1:
            # Group by weights and calculate mean performance
            weight_results = {}
            for r in results:
                weight = r.get('elo_weight', 0)
                if weight not in weight_results:
                    weight_results[weight] = []
                weight_results[weight].append(r['brier_score'])
            
            # Calculate mean Brier score for each weight
            weight_means = {w: np.mean(scores) for w, scores in weight_results.items()}
            
            # Create a plot
            weights = sorted(weight_means.keys())
            mean_scores = [weight_means[w] for w in weights]
            
            plt.figure(figsize=(10, 6))
            plt.plot(weights, mean_scores, 'o-', linewidth=2)
            plt.xlabel('ELO Weight')
            plt.ylabel('Mean Brier Score')
            plt.title('ELO Weight Optimization')
            plt.grid(True, alpha=0.3)
            
            # Mark best weight
            best_weight = min(weights, key=lambda w: weight_means[w])
            best_score = weight_means[best_weight]
            plt.scatter([best_weight], [best_score], color='red', s=100, 
                       label=f'Best: {best_weight} (Brier: {best_score:.4f})')
            plt.legend()
            
            plt.tight_layout()
            plt.show()

    def optimize_parameters(self, validation_seasons=None, lookback_years=5, checkpoint_file='optimization_checkpoint.json'):
        """
        Optimize model parameters through backtesting.
        
        Parameters:
        validation_seasons (list): List of seasons to use for validation. If None, use last 5 available seasons.
        lookback_years (int): Number of previous years to use for training for each test season
        checkpoint_file (str): File to save checkpoint results during optimization
        
        Returns:
        dict: Optimal parameters and performance metrics
        """
        import json
        import os
        import time
        
        print("Optimizing model parameters through backtesting...")
        print("New")
        
        # Get available seasons
        available_seasons = sorted(self.data['tourney_results']['Season'].unique())
        
        # Set validation_seasons if not provided
        if validation_seasons is None:
            # Use last 5 available tournament seasons
            validation_seasons = available_seasons[-5:]
            
        print(f"Using validation seasons: {validation_seasons}")
        
        # Parameters to test
        elo_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        feature_combinations = [
            {"include_elo": True, "include_advanced_stats": True},
            {"include_elo": True, "include_advanced_stats": False},
            {"include_elo": False, "include_advanced_stats": True}
        ]
        k_factors = [15, 20, 25, 30]
        carry_over_factors = [0.6, 0.7, 0.75, 0.8]
        
        # Initialize results tracking
        all_results = []
        best_result = None
        best_brier = float('inf')
        
        # Try to load checkpoint if it exists
        try:
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    all_results = checkpoint_data.get('all_results', [])
                    best_result = checkpoint_data.get('best_result')
                    best_brier = checkpoint_data.get('best_brier', float('inf'))
                    print(f"Loaded checkpoint with {len(all_results)} previous results")
                    
                    # If we have a best result, extract the best parameters
                    if best_result:
                        print(f"Current best: {best_result} (Brier: {best_brier:.4f})")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            all_results = []
            best_result = None
            best_brier = float('inf')
            
        # Helper function to save checkpoint
        def save_checkpoint():
            checkpoint_data = {
                'all_results': all_results,
                'best_result': best_result,
                'best_brier': best_brier,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            try:
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                print(f"Saved checkpoint to {checkpoint_file}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
        
        # Test ELO weight parameter if needed or we're starting from scratch
        if best_result is None or 'elo_weight' not in best_result:
            print("\n1. Testing ELO blending weights...")
            for weight in elo_weights:
                # Skip if we've already tested this weight (from checkpoint)
                if any(r.get('elo_weight') == weight and 
                       r.get('include_elo') == True and 
                       r.get('include_advanced_stats') == True and
                       len(r.keys()) <= 4  # Basic config without k_factor etc.
                       for r in all_results):
                    print(f"Skipping weight {weight} (already tested)")
                    continue
                    
                print(f"Testing ELO weight: {weight}")
                result = self.backtest_multiple_seasons(
                    test_seasons=validation_seasons,
                    lookback_years=lookback_years,
                    include_elo=True,
                    include_advanced_stats=True,
                    blend_elo=True if weight > 0 else False,
                    elo_weight=weight,
                    visualize=False
                )

                # Track this configuration
                config = {
                    'elo_weight': weight,
                    'include_elo': True,
                    'include_advanced_stats': True,
                    'brier_score': result['aggregate']['brier_score']
                }
                
                if np.isnan(result['aggregate']['brier_score']):
                    print(f"WARNING: Got NaN Brier score, skipping this configuration")
                    config['brier_score'] = float('inf')  # Use infinity instead of NaN
                else:
                    config['brier_score'] = result['aggregate']['brier_score']
                
                all_results.append(config)
                
                # Update best result if better
                if result['aggregate']['brier_score'] < best_brier:
                    best_brier = result['aggregate']['brier_score']
                    best_result = config.copy()
                    
                # Save checkpoint after each test
                save_checkpoint()
            
            # If we still don't have a best result, use a default
            if best_result is None:
                best_result = {
                    'elo_weight': 0.3,  # Default value
                    'include_elo': True,
                    'include_advanced_stats': True,
                    'brier_score': float('inf')
                }
        
        # Identify best ELO weight
        best_elo_weight = best_result['elo_weight']
        print(f"Best ELO weight: {best_elo_weight} (Brier: {best_brier:.4f})")
        
        # Test feature combinations with best ELO weight
        if 'feature_combinations_tested' not in best_result:
            print("\n2. Testing feature combinations...")
            for combo in feature_combinations:
                # Skip if we've already tested this exact combination
                if combo['include_elo'] == best_result['include_elo'] and \
                   combo['include_advanced_stats'] == best_result['include_advanced_stats']:
                    continue
                    
                # Also skip if we find this combination in previous results
                if any(r.get('include_elo') == combo['include_elo'] and 
                       r.get('include_advanced_stats') == combo['include_advanced_stats'] and
                       r.get('elo_weight') == best_elo_weight and
                       len(r.keys()) <= 4  # Basic config without k_factor etc.
                       for r in all_results):
                    print(f"Skipping feature combo {combo} (already tested)")
                    continue
                
                print(f"Testing feature combination: {combo}")
                result = self.backtest_multiple_seasons(
                    test_seasons=validation_seasons,
                    lookback_years=lookback_years,
                    include_elo=combo['include_elo'],
                    include_advanced_stats=combo['include_advanced_stats'],
                    blend_elo=True if best_elo_weight > 0 else False,
                    elo_weight=best_elo_weight,
                    visualize=False
                )
                
                # Track this configuration
                config = {
                    'elo_weight': best_elo_weight,
                    'include_elo': combo['include_elo'],
                    'include_advanced_stats': combo['include_advanced_stats'],
                    'brier_score': result['aggregate']['brier_score']
                }
                all_results.append(config)
                
                # Update best result if better
                if result['aggregate']['brier_score'] < best_brier:
                    best_brier = result['aggregate']['brier_score']
                    best_result = config.copy()
                
                # Save checkpoint after each test
                save_checkpoint()
            
            # Mark that we've tested feature combinations
            best_result['feature_combinations_tested'] = True
            save_checkpoint()
        
        # Update best features
        best_include_elo = best_result['include_elo']
        best_include_adv_stats = best_result['include_advanced_stats']
        print(f"Best feature combination: include_elo={best_include_elo}, include_advanced_stats={best_include_adv_stats} (Brier: {best_brier:.4f})")
        
        # Only test ELO parameters if we're using ELO
        if best_include_elo:
            # Test ELO k-factor if not already tested
            if 'k_factor' not in best_result:
                print("\n3. Testing ELO k-factor...")
                
                # Calculate ELO with different k-factors
                for k in k_factors:
                    # Skip if we've already tested this k-factor
                    if any(r.get('k_factor') == k and 
                           r.get('include_elo') == best_include_elo and
                           r.get('include_advanced_stats') == best_include_adv_stats and
                           r.get('elo_weight') == best_elo_weight and
                           r.get('carry_over_factor', 0.75) == 0.75  # Default value
                           for r in all_results):
                        print(f"Skipping k-factor {k} (already tested)")
                        continue
                        
                    print(f"Testing k-factor: {k}")
                    # Need to recalculate ELO with this k-factor
                    self.calculate_elo_ratings(start_year=min(available_seasons), 
                                             k_factor=k, 
                                             carry_over_factor=0.75)  # Use default carry-over for now
                    
                    result = self.backtest_multiple_seasons(
                        test_seasons=validation_seasons,
                        lookback_years=lookback_years,
                        include_elo=best_include_elo,
                        include_advanced_stats=best_include_adv_stats,
                        blend_elo=True if best_elo_weight > 0 else False,
                        elo_weight=best_elo_weight,
                        visualize=False
                    )
                    
                    # Track this configuration
                    config = {
                        'elo_weight': best_elo_weight,
                        'include_elo': best_include_elo,
                        'include_advanced_stats': best_include_adv_stats,
                        'k_factor': k,
                        'carry_over_factor': 0.75,  # Default
                        'brier_score': result['aggregate']['brier_score']
                    }
                    all_results.append(config)
                    
                    # Update best result if better
                    if result['aggregate']['brier_score'] < best_brier:
                        best_brier = result['aggregate']['brier_score']
                        best_result = config.copy()
                    
                    # Save checkpoint after each test
                    save_checkpoint()
            
            # Update best k-factor
            best_k = best_result.get('k_factor', 20)  # Default if not present
            print(f"Best k-factor: {best_k} (Brier: {best_brier:.4f})")
            
            # Test ELO carry-over factor if not already tested
            if 'carry_over_factor' not in best_result:
                print("\n4. Testing ELO carry-over factor...")
                
                # Calculate ELO with different carry-over factors
                for carry_over in carry_over_factors:
                    # Skip if we've already tested this carry-over factor
                    if any(r.get('carry_over_factor') == carry_over and 
                           r.get('k_factor', best_k) == best_k and
                           r.get('include_elo') == best_include_elo and
                           r.get('include_advanced_stats') == best_include_adv_stats and
                           r.get('elo_weight') == best_elo_weight
                           for r in all_results):
                        print(f"Skipping carry-over factor {carry_over} (already tested)")
                        continue
                        
                    print(f"Testing carry-over factor: {carry_over}")
                    # Need to recalculate ELO with this carry-over factor
                    self.calculate_elo_ratings(start_year=min(available_seasons), 
                                             k_factor=best_k,
                                             carry_over_factor=carry_over)
                    
                    result = self.backtest_multiple_seasons(
                        test_seasons=validation_seasons,
                        lookback_years=lookback_years,
                        include_elo=best_include_elo,
                        include_advanced_stats=best_include_adv_stats,
                        blend_elo=True if best_elo_weight > 0 else False,
                        elo_weight=best_elo_weight,
                        visualize=False
                    )
                    
                    # Track this configuration
                    config = {
                        'elo_weight': best_elo_weight,
                        'include_elo': best_include_elo,
                        'include_advanced_stats': best_include_adv_stats,
                        'k_factor': best_k,
                        'carry_over_factor': carry_over,
                        'brier_score': result['aggregate']['brier_score']
                    }
                    all_results.append(config)
                    
                    # Update best result if better
                    if result['aggregate']['brier_score'] < best_brier:
                        best_brier = result['aggregate']['brier_score']
                        best_result = config.copy()
                    
                    # Save checkpoint after each test
                    save_checkpoint()
            
            # Update best carry-over factor
            best_carry_over = best_result.get('carry_over_factor', 0.75)  # Default if not present
            print(f"Best carry-over factor: {best_carry_over} (Brier: {best_brier:.4f})")
        
        # Remove any temporary fields used for tracking process
        if 'feature_combinations_tested' in best_result:
            del best_result['feature_combinations_tested']
        
        # Test optimal configuration one more time with visualization
        print("\n5. Validating optimal configuration...")
        
        # If using ELO, recalculate with optimal parameters
        if best_include_elo:
            self.calculate_elo_ratings(start_year=min(available_seasons), 
                                      k_factor=best_result.get('k_factor', 20),
                                      carry_over_factor=best_result.get('carry_over_factor', 0.75))
        
        # Final backtest with visualization
        final_result = self.backtest_multiple_seasons(
            test_seasons=validation_seasons,
            lookback_years=lookback_years,
            include_elo=best_result['include_elo'],
            include_advanced_stats=best_result['include_advanced_stats'],
            blend_elo=True if best_result['elo_weight'] > 0 else False,
            elo_weight=best_result['elo_weight'],
            visualize=True
        )
        
        # Display final optimal parameters
        print("\nOptimal parameters:")
        for key, value in best_result.items():
            if key != 'brier_score':
                print(f"- {key}: {value}")
        print(f"Final Brier score: {best_brier:.4f}")
        
        # Save final result
        final_checkpoint = {
            'all_results': all_results,
            'best_result': best_result,
            'best_brier': best_brier,
            'final_metrics': final_result['aggregate'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to both checkpoint and final results file
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(final_checkpoint, f, indent=2)
            
            with open('optimal_parameters.json', 'w') as f:
                json.dump(final_checkpoint, f, indent=2)
                
            print(f"Saved final results to optimal_parameters.json")
        except Exception as e:
            print(f"Error saving final results: {e}")
        
        # Return the best parameters and all tested configurations
        return {
            'optimal_params': best_result,
            'all_results': all_results,
            'final_metrics': final_result['aggregate']
        }
    