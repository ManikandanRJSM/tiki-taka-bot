
pre_process_schema = ['home_score', 'away_score', 'tournament', 'city', 'country', 'neutral', 'match_result']


# Feature engineering vectors:

x_training_schema = [
    'h2h_win_ratio_home', 
    'h2h_avg_goals', 
    'home_elo', 
    'away_elo', 
    'elo_diff', 
    'expected_prob', 
    'home_team_win_rate_5', 
    'away_team_win_rate_5', 
    'home_team_avg_goals_rate_5', 
    'away_team_avg_goals_rate_5', 
    'home_avg_goals_conceded_last5', 
    'away_avg_goals_conceded_last5',
    'last_5_home_win_rate_home_only',
    'last_5_away_win_rate_away_only'
]

y_training_schema = ['match_result']

x_test_schema = [
    'h2h_win_ratio_home', 
    'h2h_avg_goals', 
    'home_elo', 
    'away_elo', 
    'elo_diff', 
    'expected_prob', 
    'home_team_win_rate_5', 
    'away_team_win_rate_5', 
    'home_team_avg_goals_rate_5', 
    'away_team_avg_goals_rate_5', 
    'home_avg_goals_conceded_last5', 
    'away_avg_goals_conceded_last5',
    'last_5_home_win_rate_home_only',
    'last_5_away_win_rate_away_only'
]

y_test_schema = ['match_result']