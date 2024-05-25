# nie ma pozycji
# musial grac conajmniej 65 gier - powyzej 20 min - dwie gry moze miec powyzej 15
# łącznie jest 82 meczy. 
# od 1988 uczyc, tego roku byly 3 druzyny wybierane
# od 2022 WNBA zmienilo na positionless, wiec mamy dwa dodatkowe lata na uczenie, większa waga ich

# nba_api -> analysis_archive -> stats
# nba_api -> nba_api -> stats -> endpoints

# FG_PCT fields goals percentage made/attmptet
# TOV turnovers per 100 plays
# PF personal fouls

from nba_api.stats.endpoints import playergamelogs, commonallplayers
from nba_api.stats.static import players, teams

import pandas as pd
import numpy as np

# function not used at the end, 
# gets ids of all players active in given season
def get_active_player_ids_giv_season(Season):
    players_data = commonallplayers.CommonAllPlayers(is_only_current_season=1, league_id='00', season=Season)
    players_df = players_data.get_data_frames()[0]
    # choosing just id for later on 
    players_active_df = players_df['PERSON_ID']
    return players_active_df

# gets stats of all players in all games they played,
# then sorts that info, for colculating mean values
def get_players_stats_from_season(Season):
    # getting all logs of all games of all players in a season
    try:
        player_logs = playergamelogs.PlayerGameLogs(season_nullable=Season, season_type_nullable="Regular Season")
    except Exception as e: 
        print(f"Wyjątek: {e}")
    
    game_logs_df = player_logs.get_data_frames()[0]
    # choosing right statistics
    # print(game_logs_df.columns)
    columns = ['PLAYER_ID', 'MIN', 'FG_PCT', 'FG3_PCT', 'FTM', 'FT_PCT', 'OREB',
                'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'PTS', 'PLUS_MINUS']
    game_logs_df = game_logs_df[columns]
    # sorting ascending
    game_logs_df = game_logs_df.sort_values(by=["PLAYER_ID"], ignore_index=True)
    # saving to a file
    game_logs_df.to_csv(f'stats/allPlayers_allGames/{Season}')

# saving stats to files for allPlayers_allGames
def saving_stats_csv():
    # iterating trough seasons
    for i in range (1988, 2024):
        if i >= 1999 and i < 2009:
            season = str(i)+"-0"+str((i+1)%10)
        elif i>=2009 and i < 2019:
            season = str(i)+"-1"+str((i+1)%10)
        elif i>=2019:
            season = str(i)+"-2"+str((i+1)%10)
        elif i < 1999:
            season = str(i)+"-"+str(i%100+1)
        print(season)
        get_players_stats_from_season(season)

# from a csv file get list of unique ids
def get_unique_id_and_df(Season):
    season_df = pd.read_csv(f'stats/allPlayers_allGames/{Season}')
    ids = []
    ids.append(0)
    for id in season_df["PLAYER_ID"]:
        if id > ids[-1]: ids.append(id)
    ids.pop(0)
    # print(ids)
    return ids, season_df

# calculating mean value of stats and
# adds column with no of games played
def calculate_mean_value(dataframe, PlayerID):

    filtered_rows = dataframe[dataframe['PLAYER_ID'] == PlayerID]
    filtered_rows = filtered_rows.astype(float)


    # value of no of games played
    value_to_add = len(filtered_rows)
    print(value_to_add)
    season_avg_stats = filtered_rows.mean()

    season_avg_stats["no_games_played"] = value_to_add
    print(type(season_avg_stats), season_avg_stats)

    df_avg = pd.DataFrame(season_avg_stats).transpose()
    df_avg.loc[0, 'PLAYER_ID'] = PlayerID
    columns = ['PLAYER_ID', 'MIN', 'FG_PCT', 'FG3_PCT', 'FTM', 'FT_PCT', 'OREB', 
               'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'PTS', 'PLUS_MINUS', "no_games_played"]
    df_avg = df_avg[columns]
    return df_avg

def saving_avg_values():
    for i in range (1988, 2024):
        if i >= 1999 and i < 2009:
            season = str(i)+"-0"+str((i+1)%10)
        elif i>=2009 and i < 2019:
            season = str(i)+"-1"+str((i+1)%10)
        elif i>=2019:
            season = str(i)+"-2"+str((i+1)%10)
        elif i < 1999:
            season = str(i)+"-"+str(i%100+1)
        print(season)
        ids, season_df = get_unique_id_and_df(season)
        avg_array = []
        for id in ids:
            avg_array.append(calculate_mean_value(season_df, id))
        datafram_to_save = pd.concat(avg_array, ignore_index=True)
        # print(datafram_to_save)
        datafram_to_save.to_csv(f'stats/avgPlayersStats/{i}')

def main():
    saving_stats_csv()

    saving_avg_values()

main()

#######################
### TODO
# extract players from 2023-24 season with 20 or more min
######################


# just for 2023 season
# na później
# def players_id_twenty_or_more_minutes():
#     # Pobierz dzienniki gier dla wszystkich graczy w bieżącym sezonie
#     player_logs = playergamelogs.PlayerGameLogs().get_data_frames()[0]

#     # Zgrupuj dzienniki gier według identyfikatora gracza i oblicz sumę minut dla każdego gracza
#     player_minutes = player_logs.groupby('PLAYER_ID')['MIN'].sum()
#     print(player_logs.columns)

#     # Wybierz graczy, którzy grali co najmniej 65 meczy i każdy mecz trwał co najmniej 20 minut
#     qualified_players = player_minutes[(player_logs.groupby('PLAYER_ID')['GAME_ID'].count() >= 65) & (player_minutes >= (65 * 20))]

#     # Wyświetl wyniki
#     print(qualified_players)

# def players_id_twenty_or_more_minutes(PlayersIDs):
#     counter = 0
#     Ids = []
#     for PlayerID in PlayersIDs:
#         gamelog = playergamelog.PlayerGameLog(player_id=PlayerID, season=2023)
#         gamelog_df = gamelog.get_data_frames()[0]
#         min_df = gamelog_df["MIN"]
#         for min in min_df:
#             if min >= 20:
#                 counter += 1
#         if counter >= 65:
#             Ids.append(PlayerID)
#         counter = 0






