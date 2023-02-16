import pandas as pd

def engineer_features(data):
    # Convert data types
    data = data.astype({'HOME_TEAM_ID': int, 'VISITOR_TEAM_ID': int, 'SEASON': int, 'PTS_home': int, 'PTS_away': int})

    # Add a column for the home team's win percentage
    data['HOME_TEAM_WINS'] = data.groupby('HOME_TEAM_ID').cumsum()['PTS_home'] > data.groupby('HOME_TEAM_ID').cumsum()['PTS_away']
    data['HOME_WIN_PCT'] = data.apply(lambda x: (data.loc[x.name-1]['HOME_TEAM_WINS'] if x.name > 0 else 0) / x.name if x.name != 0 else 0, axis=1)

    # Add a column for the away team's win percentage
    data['AWAY_TEAM_WINS'] = ~data['HOME_TEAM_WINS']
    data['AWAY_WIN_PCT'] = data.apply(lambda x: (data.loc[x.name-1]['AWAY_TEAM_WINS'] if x.name > 0 else 0) / x.name if x.name != 0 else 0, axis=1)

    # Add a column for the home team's winning streak
    data['HOME_WIN_STREAK'] = data.apply(lambda x: get_winning_streak(x.name, x['HOME_TEAM_ID'], data), axis=1)

    # Add a column for the away team's winning streak
    data['AWAY_WIN_STREAK'] = data.apply(lambda x: get_winning_streak(x.name, x['VISITOR_TEAM_ID'], data), axis=1)

    # Drop any rows with missing data
    data.dropna(inplace=True)

    # Drop any unneeded columns
    data = data[['HOME_TEAM_WINS','HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON', 'PTS_home', 'PTS_away', 'HOME_WIN_PCT', 'AWAY_WIN_PCT', 'HOME_WIN_STREAK', 'AWAY_WIN_STREAK']]

    return data



def get_winning_streak(index, team_id, data):
    # Get the team's game results up to the current game
    team_results = data.loc[data.index <= index].loc[(data['HOME_TEAM_ID'] == team_id) | (data['VISITOR_TEAM_ID'] == team_id)]

    # Calculate the team's winning streak
    streak = 0
    for i in range(len(team_results)):
        if team_results.iloc[i]['HOME_TEAM_ID'] == team_id:
            if team_results.iloc[i]['PTS_home'] > team_results.iloc[i]['PTS_away']:
                streak += 1
            else:
                streak = 0
        else:
            if team_results.iloc[i]['PTS_away'] > team_results.iloc[i]['PTS_home']:
                streak += 1
            else:
                streak = 0
        if streak == 5:
            break

    return streak

if __name__ == '__main__':
    data = pd.read_csv('data/processed_data.csv',delimiter=';')
    
    print(data.columns)
    data = engineer_features(data)
    data.to_csv('data/processed_data.csv', index=False)
