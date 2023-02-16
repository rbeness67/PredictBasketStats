import pandas as pd

def prepare_data():
    # Load the games data from the csv file
    games = pd.read_csv('data/games.csv')

    # Drop any rows with missing data
    games.dropna(inplace=True)

    # Drop any rows where the game was cancelled
    games = games[games.GAME_STATUS_TEXT == 'Final']

    # Drop any unneeded columns
    games = games[[ 'GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON', 'PTS_home', 'PTS_away', 'HOME_TEAM_WINS']]


    # Save the processed data to a csv file
    games.to_csv('data/processed_data.csv', index=False)

if __name__ == '__main__':
    prepare_data()
