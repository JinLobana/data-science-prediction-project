import pandas as pd
import numpy as np

def add_rookies():

    # reading file from my friend then sorting and keeping only first season of each player
    df = pd.read_csv(f'stats/from_friends/player_avg_data.csv')
    df = df.sort_values(by=['PLAYER_ID', 'SEASON'])
    first_season = df.drop_duplicates(subset='PLAYER_ID', keep='first')

    df['ROOKIE'] = 0.0

    df.loc[first_season.index, 'ROOKIE'] = 1.0

    # choosing right columns, deleting ones that I do not need
    columns = ['PLAYER_ID', 'MIN', 'FG_PCT', 'FG3_PCT', 'FTM', 'FT_PCT', 'OREB',
                'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'PTS','PLUS_MINUS',"SEASON" ,"ROOKIE", "ALL_ROOKIE_TEAM_NUMBER"]
    df = df[columns]

    # Find season 1996 and delete it 
    rows_to_drop = df[df['SEASON'] == '1996-97'].index
    df = df.drop(rows_to_drop)

    # sorting by seasons
    df = df.sort_values(by=['SEASON'])
    df.to_csv("stats/rookies/rookies.csv")

    # for loop for getting right string of season
    for i in range (1997, 2024):
        if i >= 1999 and i < 2009:
            season = str(i)+"-0"+str((i+1)%10)
        elif i>=2009 and i < 2019:
            season = str(i)+"-1"+str((i+1)%10)
        elif i>=2019:
            season = str(i)+"-2"+str((i+1)%10)
        elif i < 1999:
            season = str(i)+"-"+str(i%100+1)
        print(season)


        # Filtering rows with season same as in for loop
        season_df = df[df['SEASON'] == season]

        # copying dataframe
        rookies_df = season_df.copy()

        # choosing only rookies
        rookies_df = rookies_df[rookies_df['ROOKIE'] == 1]
        all_rookies_df = rookies_df.copy()
        
        rookies_df = rookies_df.reset_index(drop=True)
        # deleting bad columns
        rookies_df.drop(columns=['ROOKIE', "SEASON", "ALL_ROOKIE_TEAM_NUMBER"], inplace=True)

        
        all_rookies_df.loc[all_rookies_df['ALL_ROOKIE_TEAM_NUMBER'] == 2.0, 'ALL_ROOKIE_TEAM_NUMBER'] = 1.0
        all_rookies_df = all_rookies_df.fillna(0.0)
        
        all_rookies_df = all_rookies_df.reset_index(drop=True)
        all_rookies_df.drop(columns=['ROOKIE', "SEASON"], inplace=True)
        columns = ["ALL_ROOKIE_TEAM_NUMBER"]
        all_rookies_df = all_rookies_df[columns]
        print(all_rookies_df)


        all_rookies_df.to_csv(f"stats/rookies/all_rookies/{i}")
        rookies_df.to_csv(f"stats/rookies/{i}")
        print(len(all_rookies_df), len(rookies_df))

    return df

def main():
    add_rookies()



if __name__ == "__main__":
    main()