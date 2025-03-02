import pandas as pd
import numpy as np
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
        self.team_elo_ratings = {}  # Store ELO ratings by (season, team_id, day)
        self.advanced_team_stats = {}  # Store advanced team stats by season
        self.seed_lookup = {}  # Store seed lookups by (season, team_id)
    
    def load_data(self):
        """Load all necessary data files"""
        # Teams data
        all_teams = pd.read_csv(f"{self.data_dir}/{self.gender}Teams.csv")

        # Women's teams don't have the first/last season cols for some reason
        if self.gender == "M":
            self.data['teams'] = all_teams[all_teams['LastD1Season'] >= self.current_season]
        else:
            mens_teams = pd.read_csv(f"{self.data_dir}/MTeams.csv")
            self.data['teams'] = all_teams[all_teams['TeamName'].isin(mens_teams[mens_teams['LastD1Season'] >= self.current_season]['TeamName'])]
        
        # Regular season results
        self.data['regular_season'] = pd.read_csv(
            f"{self.data_dir}/{self.gender}RegularSeasonCompactResults.csv"
        )
        
        # Tournament results
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
            
        # Try to load rankings data if available
        try:
            self.data['rankings'] = pd.read_csv(f"{self.data_dir}/{self.gender}MasseyOrdinals.csv")
            self.rankings_available = True
        except FileNotFoundError:
            self.rankings_available = False
            
        # Load secondary tournament results if available
        try:
            self.data['secondary_tourney'] = pd.read_csv(
                f"{self.data_dir}/{self.gender}SecondaryTourneyCompactResults.csv"
            )
            self.secondary_tourney_available = True
        except FileNotFoundError:
            self.secondary_tourney_available = False
            
        # Process seeds
        self.preprocess_seeds()
        
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
        
    def calculate_elo_ratings(self, start_year=2003, k_factor=30, home_advantage=100, 
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

    def calculate_advanced_team_stats(self, start_season=2003):
        """
        Calculate advanced team statistics for all seasons where detailed data is available.
        These include:
        - Offensive/Defensive Efficiency
        - Four Factors (eFG%, TOV%, ORB%, FT Rate)
        - Pace
        - Shooting percentages
        - Advanced possession-based metrics
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
    
    def predict_game(self, team1_id, team2_id, day_num, season):
        """
        Predict the outcome of a game between team1 and team2
        
        Parameters:
        team1_id: ID of the first team
        team2_id: ID of the second team
        day_num: Day number of the game (for tournament games)
        season: Season of the game
        
        Returns:
        float: Probability of team1 winning
        """
        # Ensure we have ELO ratings
        if not self.team_elo_ratings:
            self.calculate_elo_ratings()
            
        # Get ELO ratings
        team1_elo = self.get_team_elo(season, team1_id, day_num - 1)
        team2_elo = self.get_team_elo(season, team2_id, day_num - 1)
        
        # Calculate win probability
        return self.elo_win_probability(team1_elo, team2_elo)
    
    def generate_predictions(self, submission_file=None):
        """
        Generate predictions for the current tournament
        
        Parameters:
        submission_file (str): Path to save the submission file
        
        Returns:
        DataFrame: Prediction results
        """
        # Ensure we have necessary data
        if not self.team_elo_ratings:
            self.calculate_elo_ratings()
            
        # Get current season seeds
        current_seeds = self.data['processed_seeds'][self.data['processed_seeds']['Season'] == self.current_season]
        
        if len(current_seeds) == 0:
            raise ValueError(f"No seed data found for season {self.current_season}")
        
        # Generate all possible matchups
        team_ids = current_seeds['TeamID'].unique()
        matchups = []
        
        for i, team1_id in enumerate(team_ids):
            for team2_id in team_ids[i+1:]:
                # Create ID in required format
                matchup_id = f"{self.current_season}_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"
                
                # Make prediction
                if team1_id < team2_id:
                    pred = self.predict_game(team1_id, team2_id, 134, self.current_season)
                else:
                    pred = 1.0 - self.predict_game(team2_id, team1_id, 134, self.current_season)
                
                matchups.append({
                    'ID': matchup_id,
                    'Pred': pred
                })
        
        # Create submission DataFrame
        submission_df = pd.DataFrame(matchups)
        
        # Save to file if requested
        if submission_file:
            submission_df.to_csv(submission_file, index=False)
            print(f"Saved {len(submission_df)} predictions to {submission_file}")
        
        return submission_df
    
    def backtest_tournament(self, test_season, visualize=True):
        """
        Backtest predictions on a historical tournament
        
        Parameters:
        test_season: Season to test on
        visualize: Whether to visualize results
        
        Returns:
        dict: Evaluation metrics
        """
        print(f"Backtesting on {test_season} tournament...")
        
        # Get tournament games for the season
        tourney_games = self.data['tourney_results']
        test_games = tourney_games[tourney_games['Season'] == test_season]
        
        if len(test_games) == 0:
            print(f"No games found for {test_season} tournament")
            return None
            
        # Generate predictions and evaluate
        predictions = []
        actuals = []
        game_details = []
        
        for _, game in test_games.iterrows():
            day_num = game['DayNum']
            team1_id = game['WTeamID']  # Winner
            team2_id = game['LTeamID']  # Loser
            
            # Get prediction
            pred = self.predict_game(team1_id, team2_id, day_num, test_season)
            
            # Store prediction and result
            predictions.append(pred)
            actuals.append(1)  # Team1 won
            
            # Get seeds
            team1_seed = self.seed_lookup.get((test_season, team1_id), None)
            team2_seed = self.seed_lookup.get((test_season, team2_id), None)
            
            # Store game details
            game_details.append({
                'DayNum': day_num,
                'Round': self._get_tournament_round(day_num),
                'Team1ID': team1_id,
                'Team2ID': team2_id,
                'Team1Seed': team1_seed,
                'Team2Seed': team2_seed,
                'SeedDiff': team2_seed - team1_seed if team1_seed and team2_seed else None,
                'Team1Score': game['WScore'],
                'Team2Score': game['LScore'],
                'ScoreDiff': game['WScore'] - game['LScore'],
                'Prediction': pred,
                'Actual': 1,
                'Correct': pred >= 0.5  # Prediction was correct if >= 0.5
            })
            
            # Also add reversed matchup for evaluation
            predictions.append(1 - pred)
            actuals.append(0)  # Team2 lost
            
            # Store reversed game details
            game_details.append({
                'DayNum': day_num,
                'Round': self._get_tournament_round(day_num),
                'Team1ID': team2_id,
                'Team2ID': team1_id,
                'Team1Seed': team2_seed,
                'Team2Seed': team1_seed,
                'SeedDiff': team1_seed - team2_seed if team1_seed and team2_seed else None,
                'Team1Score': game['LScore'],
                'Team2Score': game['WScore'],
                'ScoreDiff': game['LScore'] - game['WScore'],
                'Prediction': 1 - pred,
                'Actual': 0,
                'Correct': (1 - pred) < 0.5  # Prediction was correct if < 0.5
            })
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Brier score (mean squared error)
        brier_score = np.mean((predictions - actuals) ** 2)
        
        # Accuracy
        accuracy = np.mean((predictions > 0.5) == actuals)
        
        # Log loss
        epsilon = 1e-15  # Prevent log(0)
        predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
        log_loss_value = -np.mean(actuals * np.log(predictions_clipped) + 
                                 (1 - actuals) * np.log(1 - predictions_clipped))
        
        # Store results
        results = {
            'season': test_season,
            'num_games': len(test_games),
            'brier_score': brier_score,
            'accuracy': accuracy,
            'log_loss': log_loss_value,
            'game_details': game_details
        }
        
        # Print summary
        print(f"Brier Score: {brier_score:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Log Loss: {log_loss_value:.4f}")
        
        # Analyze performance by round
        round_performance = self._analyze_round_performance(game_details)
        for round_name, metrics in round_performance.items():
            print(f"{round_name}: Games={metrics['count']}, Accuracy={metrics['accuracy']:.4f}")
        
        # Analyze performance by seed difference
        seed_performance = self._analyze_seed_performance(game_details)
        
        # Visualize if requested
        if visualize:
            self._visualize_backtest(results)
            
        return results
    
    def backtest_multiple_seasons(self, seasons=None, visualize=True):
        """
        Backtest on multiple tournament seasons
        
        Parameters:
        seasons: List of seasons to test (default: last 5 available seasons)
        visualize: Whether to visualize aggregate results
        
        Returns:
        dict: Aggregate and per-season metrics
        """
        if seasons is None:
            # Use last 5 seasons by default
            all_seasons = sorted(self.data['tourney_results']['Season'].unique())
            seasons = all_seasons[-5:]
            
        print(f"Backtesting on {len(seasons)} tournament seasons: {seasons}")
        
        # Run backtests
        all_results = []
        for season in seasons:
            result = self.backtest_tournament(season, visualize=False)
            if result:
                all_results.append(result)
                
        if not all_results:
            print("No valid backtest results")
            return None
            
        # Calculate aggregate metrics
        agg_metrics = {
            'num_seasons': len(all_results),
            'brier_score': np.mean([r['brier_score'] for r in all_results]),
            'accuracy': np.mean([r['accuracy'] for r in all_results]),
            'log_loss': np.mean([r['log_loss'] for r in all_results])
        }
        
        # Print aggregate results
        print("\nAggregate Results:")
        print(f"Seasons: {len(all_results)}")
        print(f"Average Brier Score: {agg_metrics['brier_score']:.4f}")
        print(f"Average Accuracy: {agg_metrics['accuracy']:.4f}")
        print(f"Average Log Loss: {agg_metrics['log_loss']:.4f}")
        
        # Visualize if requested
        if visualize:
            self._visualize_multiple_backtests(all_results)
            
        return {
            'aggregate': agg_metrics,
            'per_season': all_results
        }
    
    def _get_tournament_round(self, day_num):
        """Determine tournament round from day number"""
        if day_num <= 135:
            return "Play-In"
        elif day_num <= 137:
            return "Round 1"
        elif day_num <= 139:
            return "Round 2"
        elif day_num <= 144:
            return "Sweet 16"
        elif day_num <= 146:
            return "Elite 8"
        elif day_num <= 152:
            return "Final 4"
        else:
            return "Championship"
    
    def _analyze_round_performance(self, game_details):
        """Analyze prediction performance by tournament round"""
        # Group games by round
        rounds = {}
        
        # Get unique games (every other game is a duplicate with teams flipped)
        unique_games = [game for i, game in enumerate(game_details) if i % 2 == 0]
        
        # Group by round
        for game in unique_games:
            round_name = game['Round']
            if round_name not in rounds:
                rounds[round_name] = {
                    'count': 0,
                    'correct': 0
                }
                
            rounds[round_name]['count'] += 1
            if game['Correct']:
                rounds[round_name]['correct'] += 1
                
        # Calculate metrics for each round
        for round_name, stats in rounds.items():
            stats['accuracy'] = stats['correct'] / stats['count'] if stats['count'] > 0 else 0
            
        return rounds
    
    def _analyze_seed_performance(self, game_details):
        """Analyze prediction performance by seed difference"""
        # Group by seed difference
        seed_groups = {}
        
        # Get unique games (every other game is a duplicate with teams flipped)
        unique_games = [game for i, game in enumerate(game_details) if i % 2 == 0]
        
        # Filter games where seed info is available
        seed_games = [game for game in unique_games if game['SeedDiff'] is not None]
        
        # Group by seed difference range
        for game in seed_games:
            seed_diff = abs(game['SeedDiff'])
            
            # Define seed difference categories
            if seed_diff == 0:
                group = "Even (0)"
            elif seed_diff <= 2:
                group = "Close (1-2)"
            elif seed_diff <= 5:
                group = "Moderate (3-5)"
            elif seed_diff <= 10:
                group = "Large (6-10)"
            else:
                group = "Extreme (11+)"
                
            if group not in seed_groups:
                seed_groups[group] = {
                    'count': 0,
                    'correct': 0,
                    'upsets': 0
                }
                
            seed_groups[group]['count'] += 1
            if game['Correct']:
                seed_groups[group]['correct'] += 1
                
            # Count upset if lower seed won
            if game['SeedDiff'] > 0 and game['Actual'] == 1:
                seed_groups[group]['upsets'] += 1
                
        # Calculate metrics for each group
        for group, stats in seed_groups.items():
            stats['accuracy'] = stats['correct'] / stats['count'] if stats['count'] > 0 else 0
            stats['upset_rate'] = stats['upsets'] / stats['count'] if stats['count'] > 0 else 0
            
        return seed_groups
    
    def _visualize_backtest(self, results):
        """Visualize backtest results for a single season"""
        # Get game details
        game_details = results['game_details']
        
        # Get unique games (odd indices are duplicates with flipped teams)
        unique_games = [game for i, game in enumerate(game_details) if i % 2 == 0]
        
        # Convert to DataFrame
        df = pd.DataFrame(unique_games)
        
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Add team names
        team_df = self.data['teams']
        
        def get_team_name(team_id):
            team = team_df[team_df['TeamID'] == team_id]
            if len(team) > 0:
                return team.iloc[0]['TeamName']
            return f"Team {team_id}"
        
        df['Team1Name'] = df['Team1ID'].apply(get_team_name)
        df['Team2Name'] = df['Team2ID'].apply(get_team_name)
        df['MatchupLabel'] = df.apply(lambda x: f"{x['Team1Name']} vs {x['Team2Name']}", axis=1)
        
        # 1. Predictions vs. Actual outcomes
        axs[0, 0].scatter(df['Prediction'], df['Actual'], alpha=0.7)
        axs[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axs[0, 0].set_xlabel('Predicted Probability')
        axs[0, 0].set_ylabel('Actual Outcome')
        axs[0, 0].set_title('Prediction Calibration')
        axs[0, 0].grid(True, alpha=0.3)
        
        # 2. Prediction by seed difference
        df_with_seeds = df[df['SeedDiff'].notna()].copy()
        if len(df_with_seeds) > 0:
            df_with_seeds['AbsSeedDiff'] = df_with_seeds['SeedDiff'].abs()
            seed_groups = df_with_seeds.groupby('AbsSeedDiff').agg({
                'Prediction': 'mean',
                'Actual': 'mean',
                'Team1ID': 'count'
            }).rename(columns={'Team1ID': 'Count'}).reset_index()
            
            x = seed_groups['AbsSeedDiff']
            width = 0.35
            
            axs[0, 1].bar(x - width/2, seed_groups['Prediction'], width, label='Predicted', color='blue', alpha=0.7)
            axs[0, 1].bar(x + width/2, seed_groups['Actual'], width, label='Actual', color='green', alpha=0.7)
            
            for i, (_, row) in enumerate(seed_groups.iterrows()):
                axs[0, 1].text(row['AbsSeedDiff'], 0.05, f"n={row['Count']}", ha='center', fontsize=8)
            
            axs[0, 1].set_xlabel('Absolute Seed Difference')
            axs[0, 1].set_ylabel('Win Rate')
            axs[0, 1].set_title('Prediction vs. Actual by Seed Difference')
            axs[0, 1].legend()
            axs[0, 1].grid(True, alpha=0.3)
        
        # 3. Performance by round
        round_order = ['Play-In', 'Round 1', 'Round 2', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship']
        df['Round'] = pd.Categorical(df['Round'], categories=round_order, ordered=True)
        
        round_metrics = df.groupby('Round').agg({
            'Correct': 'mean',
            'Team1ID': 'count'
        }).rename(columns={'Correct': 'Accuracy', 'Team1ID': 'Count'}).reset_index()
        
        round_metrics = round_metrics.sort_values('Round')
        
        bars = axs[1, 0].bar(round_metrics['Round'], round_metrics['Accuracy'], color='skyblue')
        
        # Add counts above bars
        for i, bar in enumerate(bars):
            count = round_metrics.iloc[i]['Count']
            axs[1, 0].text(bar.get_x() + bar.get_width()/2, 
                         bar.get_height() + 0.02, 
                         f"n={count}", 
                         ha='center',
                         fontsize=9)
        
        axs[1, 0].set_ylim(0, 1.1)
        axs[1, 0].set_xlabel('Tournament Round')
        axs[1, 0].set_ylabel('Prediction Accuracy')
        axs[1, 0].set_title('Accuracy by Tournament Round')
        axs[1, 0].grid(True, alpha=0.3)
        plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha='right')
        
        # 4. Interesting games (upsets and close calls)
        # Find upsets (higher seed lost) and wrong predictions
        df_with_seeds['HigherSeedWon'] = df_with_seeds['SeedDiff'] < 0
        df_with_seeds['Upset'] = df_with_seeds['SeedDiff'] > 0 & (df_with_seeds['Actual'] == 1)
        
        # Filter for upsets or wrong predictions
        interesting_games = df_with_seeds[(df_with_seeds['Upset']) | (~df_with_seeds['Correct'])].copy()
        interesting_games['AbsError'] = np.abs(interesting_games['Prediction'] - interesting_games['Actual'])
        interesting_games = interesting_games.sort_values('AbsError', ascending=False).head(10)
        
        interesting_games['Label'] = interesting_games.apply(
            lambda x: f"{x['Team1Name']} vs {x['Team2Name']} "
                     f"({x['Team1Seed']} vs {x['Team2Seed']}) - "
                     f"Pred: {x['Prediction']:.2f}, {'✓' if x['Correct'] else '✗'}", 
            axis=1
        )
        
        y_pos = np.arange(len(interesting_games))
        colors = ['green' if correct else 'red' for correct in interesting_games['Correct']]
        
        bars = axs[1, 1].barh(y_pos, interesting_games['AbsError'], color=colors, alpha=0.7)
        axs[1, 1].set_yticks(y_pos)
        axs[1, 1].set_yticklabels(interesting_games['Label'], fontsize=9)
        axs[1, 1].set_xlabel('Prediction Error')
        axs[1, 1].set_title('Most Interesting Games')
        axs[1, 1].grid(True, alpha=0.3)
        
        # Title
        fig.suptitle(f"Backtest Results for {results['season']} Tournament\n"
                    f"Accuracy: {results['accuracy']:.4f}  Brier Score: {results['brier_score']:.4f}",
                    fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    def _visualize_multiple_backtests(self, results_list):
        """Visualize aggregate results from multiple backtests"""
        # Convert to DataFrame
        seasons = [r['season'] for r in results_list]
        accuracy = [r['accuracy'] for r in results_list]
        brier = [r['brier_score'] for r in results_list]
        log_loss = [r['log_loss'] for r in results_list]
        
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        
        # 1. Accuracy by season
        axs[0, 0].plot(seasons, accuracy, 'o-', color='blue', linewidth=2)
        axs[0, 0].set_xlabel('Season')
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 0].set_title('Prediction Accuracy by Season')
        axs[0, 0].grid(True, alpha=0.3)
        
        # Add mean line
        mean_acc = np.mean(accuracy)
        axs[0, 0].axhline(y=mean_acc, color='red', linestyle='--', alpha=0.7)
        axs[0, 0].text(seasons[0], mean_acc + 0.01, f"Mean: {mean_acc:.4f}", color='red')
        
        # 2. Brier score by season
        axs[0, 1].plot(seasons, brier, 'o-', color='green', linewidth=2)
        axs[0, 1].set_xlabel('Season')
        axs[0, 1].set_ylabel('Brier Score')
        axs[0, 1].set_title('Brier Score by Season')
        axs[0, 1].grid(True, alpha=0.3)
        
        # Add mean line
        mean_brier = np.mean(brier)
        axs[0, 1].axhline(y=mean_brier, color='red', linestyle='--', alpha=0.7)
        axs[0, 1].text(seasons[0], mean_brier + 0.01, f"Mean: {mean_brier:.4f}", color='red')
        
        # 3. Round analysis
        round_order = ['Play-In', 'Round 1', 'Round 2', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship']
        round_data = {}
        
        # Extract round data from each season
        for result in results_list:
            # Get game details
            game_details = result['game_details']
            season = result['season']
            
            # Get unique games
            unique_games = [game for i, game in enumerate(game_details) if i % 2 == 0]
            
            # Group by round
            for game in unique_games:
                round_name = game['Round']
                if round_name not in round_data:
                    round_data[round_name] = {'correct': 0, 'total': 0}
                    
                round_data[round_name]['total'] += 1
                if game['Correct']:
                    round_data[round_name]['correct'] += 1
        
        # Calculate accuracy by round
        round_accuracy = {round_name: data['correct'] / data['total'] if data['total'] > 0 else 0 
                          for round_name, data in round_data.items()}
        
        # Convert to DataFrame
        round_df = pd.DataFrame({
            'Round': list(round_accuracy.keys()),
            'Accuracy': list(round_accuracy.values()),
            'Games': [round_data[r]['total'] for r in round_accuracy.keys()]
        })
        
        # Sort by round order
        round_df['Round'] = pd.Categorical(round_df['Round'], categories=round_order, ordered=True)
        round_df = round_df.sort_values('Round')
        
        # Plot
        bars = axs[1, 0].bar(round_df['Round'], round_df['Accuracy'], color='purple', alpha=0.7)
        
        # Add counts above bars
        for i, bar in enumerate(bars):
            games = round_df.iloc[i]['Games']
            axs[1, 0].text(bar.get_x() + bar.get_width()/2, 
                         bar.get_height() + 0.02, 
                         f"n={games}", 
                         ha='center',
                         fontsize=9)
        
        axs[1, 0].set_ylim(0, 1.1)
        axs[1, 0].set_xlabel('Tournament Round')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].set_title('Accuracy by Tournament Round (All Seasons)')
        axs[1, 0].grid(True, alpha=0.3)
        plt.setp(axs[1, 0].get_xticklabels(), rotation=45, ha='right')
        
        # 4. Seed difference analysis
        seed_data = {}
        
        # Extract seed data from each season
        for result in results_list:
            # Get game details
            game_details = result['game_details']
            
            # Get unique games
            unique_games = [game for i, game in enumerate(game_details) if i % 2 == 0]
            
            # Filter games with seed info
            seed_games = [game for game in unique_games if game.get('SeedDiff') is not None]
            
            # Group by seed difference
            for game in seed_games:
                seed_diff = abs(game['SeedDiff'])
                
                if seed_diff not in seed_data:
                    seed_data[seed_diff] = {'correct': 0, 'total': 0}
                    
                seed_data[seed_diff]['total'] += 1
                if game['Correct']:
                    seed_data[seed_diff]['correct'] += 1
        
        # Calculate accuracy by seed difference
        seed_accuracy = {diff: data['correct'] / data['total'] if data['total'] > 0 else 0 
                         for diff, data in seed_data.items()}
        
        # Convert to DataFrame
        seed_df = pd.DataFrame({
            'SeedDiff': list(seed_accuracy.keys()),
            'Accuracy': list(seed_accuracy.values()),
            'Games': [seed_data[diff]['total'] for diff in seed_accuracy.keys()]
        })
        
        # Sort by seed difference
        seed_df = seed_df.sort_values('SeedDiff')
        
        # Plot
        bars = axs[1, 1].bar(seed_df['SeedDiff'], seed_df['Accuracy'], color='orange', alpha=0.7)
        
        # Add counts above bars
        for i, bar in enumerate(bars):
            games = seed_df.iloc[i]['Games']
            axs[1, 1].text(bar.get_x() + bar.get_width()/2, 
                         bar.get_height() + 0.02, 
                         f"n={games}", 
                         ha='center',
                         fontsize=9)
        
        axs[1, 1].set_ylim(0, 1.1)
        axs[1, 1].set_xlabel('Absolute Seed Difference')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].set_title('Accuracy by Seed Difference (All Seasons)')
        axs[1, 1].grid(True, alpha=0.3)
        
        # Title
        fig.suptitle(f"Aggregate Backtest Results for {len(results_list)} Seasons\n"
                    f"Average Accuracy: {np.mean(accuracy):.4f}  Average Brier Score: {np.mean(brier):.4f}",
                    fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
