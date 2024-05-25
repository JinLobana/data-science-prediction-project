#

#'PLAYER_ID', 'MIN', 'FG_PCT', 'FG3_PCT', 'FTM', 'FT_PCT', 'OREB','DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'PTS', 'PLUS_MINUS'
# FTM - free throws made
# FG_PCT fields goals percentage made/attmptet
# TOV - turnovers per 100 plays
# PF personal fouls
# no_games_played - number of games played

# after analysis, there is not a single player chosen to all nba with less than 20min average time on court

# Grid search didnt work as good as I thought, probably because of cross validation, samples could be too low to assess properly (only 15 true values for 500 sampels)
# So I made my own grid search with for loops and custom score function

from scipy.stats import zscore
from sklearn import datasets
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, HalvingGridSearchCV


from sklearn import tree
# from sklearn import svm
# from sklearn import linear_model
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, mean_absolute_error, make_scorer, f1_score, precision_score
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import impute
from sklearn import ensemble

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import missingno as msno
import seaborn as sns
import os

from nba_api.stats.static import players, teams


def reading_all_nba():
    # reading from csv our y for ML,
    # creating df of ids, identical to all_nba names
    # saving to csv "all_nba_id.csv"
    df = pd.read_csv("stats/all_nba.csv")
    rows, columns = df.shape
    i = 0
    ids = []
    for row in range(rows):
        for column in range(1,columns):
            player = players.find_players_by_full_name(df.iloc[row, column])
            if player:
                id = player[0]
                true_id = id["id"]
                i += 1
                ids.append(true_id)
                df.iloc[row, column] = true_id
            #df[column, row] = 
    print(i)
    df.to_csv("stats/all_nba_id.csv")


def reading_from_csv():
    # reading stats from csv
    # creating list of dataframes containing avg statistics 
    dfs = []
    for i in range(1988,2024):

        df = pd.read_csv("stats/avgPlayersStats/"+str(i))
        # print(df)
        df.pop("Unnamed: 0")
        # after analysis, there is not a single player chosen to all nba with less than 20min average time on court
        # so I delete every player with less than 20min
        mask = (df["MIN"] > 25) & (df['no_games_played'] > 40) #& (df['PTS'] > 7)
        df_filtered = df[mask]
        df_filtered = df_filtered.reset_index(drop=True)
        dfs.append(df_filtered)

        # visualizing missing values
        # msno.matrix(df)
    return dfs


def searching_through_df(df, ids):
    # function searches for ids of players selected to all nba teams in season statistics
    # return list of length equal to no of players in given season
    # each value in that list represents whether and in which team player was selected

    # creating a lists of true false values
    # true means selected for a specific all nba team
    a = df["PLAYER_ID"].isin(ids)
    a = a.tolist()
    # making one list of int values from three of boolean logic
    all_nba_or_not = [True if a[i] == True else False for i in range(len(a))]
    return all_nba_or_not


def creating_list_of_all_nba_teams(X_list):
    # function creates our Y in ML, that is, whether and in which team a player was selected
    # returns a list of seasons made of lists of Y
    Y_list = []
    iterator = 0
    # looping through every season
    # choosing right value of string "season", for proper acquiring information
    for i in range (1988, 2024):
        if i >= 1999 and i < 2009:
            season = str(i)+"-0"+str((i+1)%10)
        elif i>=2009 and i < 2019:
            season = str(i)+"-1"+str((i+1)%10)
        elif i>=2019:
            season = str(i)+"-2"+str((i+1)%10)
        elif i < 1999:
            season = str(i)+"-"+str(i%100+1)
        # reading from file
        Y = pd.read_csv("stats/all_nba_id.csv")
        # deleting some awful column
        Y.pop("Unnamed: 0")
        # locating row for a value of season, then correcting df
        Y = Y[Y['SEASON'] == season]
        Y.pop("SEASON")
        # iterator just for correct way to locate rows of Y
        Y = Y.loc[iterator]
        # choosing right players from csv file
        Y_teams = [Y[f"NAME{i}"] for i in range(1,16)]
        # adding list of selected players to a list 
        # values 0,1,2,3 represents a player who: 
        # look inside a dict_of_values
        Y_list.append(searching_through_df(X_list[iterator], Y_teams))
        iterator += 1
    
    return Y_list

# think before calculating that, it takes a while (case when line with "for" loop is uncommented)
def visualisation(X, Y):
    # deleting season 2023-24 from visualisation
    X.pop()
    for x, y in zip(X, Y):
        data = x
        data["all_nba"] = y
        # correlation
        correlation = data.corr(method="pearson")
        # sns.heatmap(correlation, annot=True, cmap="coolwarm")
        # pair plots
        # sns.pairplot(data, vars=['MIN'], hue='all_nba')
        # sns.pairplot(data, vars=['MIN','PTS',"no_games_played", 'FTM','TOV'], hue='all_nba')
    # sns.pairplot(data, vars=['MIN','PTS','REB','AST','PLUS_MINUS'], hue='all_nba')
        sns.pairplot(data, vars=['PTS','STL','BLK','PF','REB','AST','PLUS_MINUS'], hue='all_nba')
    # sns.pairplot(data, vars=['MIN','PTS',"no_games_played", 'FG_PCT', 'FG3_PCT','FT_PCT'], hue='all_nba')
    plt.show()

def get_names_from_id(ids):
    all_nba_names = []
    for id in ids:
        dictionary = players.find_player_by_id(int(id))
        all_nba_names.append(dictionary.get('full_name'))
    return all_nba_names

def find_15_best(array):
    # function gets array of probability of belongingness to every class
    # it's a returned array from a method predict_proba()

    # creating lists of 0 of array's length
    all_nba_or_not = [0] * len(array)
    first_team_or_not = [0] * len(array)
    second_team_or_not = [0] * len(array)
    third_team_or_not = [0] * len(array)
    # choosing indices for highest probability of belongingness
    top_15_indices = np.argpartition(array[:, 1], -15)[-15:]
    # assignment to a specific team
    first_team = top_15_indices[10:]
    second_team = top_15_indices[5:10]
    third_team = top_15_indices[:5]
    
    # giving 1 for right indices 
    for i in top_15_indices: all_nba_or_not[i] = 1
    for i in first_team: first_team_or_not[i] = 1
    for i in second_team:  second_team_or_not[i] = 1
    for i in third_team: third_team_or_not[i] = 1

    return all_nba_or_not, first_team_or_not, second_team_or_not, third_team_or_not

def merge_all_data(X_list, Y_list):
    # Creates combined statistics of all seasons
    # The same for output data
    X_list_train = X_list[:35];  Y_list_train = Y_list[:35]
    X_combined_df = pd.DataFrame()
    for season in range(len(X_list_train)):
        X_combined_df = pd.concat([X_combined_df, X_list_train[season]], axis=0, ignore_index=True)
    X_combined_df.to_csv("stats/X_combined.csv")

    Y_combined_list = sum(Y_list_train, [])

    return X_combined_df, Y_combined_list

def get_best_hyper_parameters(X_list, Y_list, choose_method):
    # predykcja
    # default:              max_iter = 100, learning_rate = 0.1, max_depth = None, max_leaf_nodes = 31, l2_regularization = 0.0
    best_parameters_hist = []
    # Przykładowe parametry dla histGradBoost
    max_ite = [int(x) for x in np.linspace(2,37,36)]
    learning_rate = [np.round(x, 3) for x in np.linspace(0.1,0.7,61)]
    max_depth = [int(x) for x in np.linspace(1,13,12)]
    max_depth.append(None)

    param_distributions = {'max_iter': max_ite,
                          'learning_rate': learning_rate,
                          'max_depth': max_depth}

    if choose_method==0:
        for season in range(0,34):
            print("SEASON: ", season)
            
            old_precision = 0.0
            for m_i in max_ite:
                for l_r in learning_rate:
                    for m_d in max_depth:

                        clf_hist = ensemble.HistGradientBoostingClassifier(max_iter = m_i, learning_rate = l_r, max_depth = m_d)
                        clf_hist.fit(X_list[season], Y_list[season])
                        hist_proba = clf_hist.predict_proba(X_list[35])
                        Y_predict = find_15_best(hist_proba)

                        # trying to maximise precision, I think it is best score function here
                        new_precision = precision_score(Y_list[35], Y_predict)
                        if new_precision > old_precision:
                            best_mi = m_i; best_lr = l_r; best_md = m_d
                            old_precision = new_precision

            best_parameters_hist.append(f"SEASON:,{season},precision:,{old_precision},best mi:,{best_mi},best lr:,{best_lr},best md:,{best_md}")
            print(best_parameters_hist[season])

    if choose_method==1:
        # from <0 to 34> 
        X_list.pop(35)
        for season in range(0,35):
            print("SEASON: ", season)
            X_season = X_list[season]; Y_season = Y_list[season]
            X_list_new = [x for i, x in enumerate(X_list) if i != season]
            Y_list_new = [x for i, x in enumerate(Y_list) if i != season]

            X_combined_df, Y_combined_list = merge_all_data(X_list_new, Y_list_new)
            
            old_precision = 0.0
            for m_i in max_ite:
                for l_r in learning_rate:
                    for m_d in max_depth:

                        clf_hist = ensemble.HistGradientBoostingClassifier(max_iter = m_i, learning_rate = l_r, max_depth = m_d)
                        clf_hist.fit(X_combined_df, Y_combined_list)
                        hist_proba = clf_hist.predict_proba(X_season)
                        Y_predict = find_15_best(hist_proba)

                        # trying to maximise precision, I think it is best score function here
                        new_precision = precision_score(Y_season, Y_predict)
                        if new_precision > old_precision:
                            best_mi = m_i; best_lr = l_r; best_md = m_d
                            old_precision = new_precision

            best_parameters_hist.append(f"SEASON:,{season},precision:,{old_precision},best mi:,{best_mi},best lr:,{best_lr},best md:,{best_md}")
            print(best_parameters_hist[season])
    
    if choose_method==2:
        X_combined_df, Y_combined_list = merge_all_data(X_list, Y_list)
        
        old_precision = 0.0
        for m_i in max_ite:
            for l_r in learning_rate:
                for m_d in max_depth:

                    clf_hist = ensemble.HistGradientBoostingClassifier(max_iter = m_i, learning_rate = l_r, max_depth = m_d)
                    clf_hist.fit(X_combined_df, Y_combined_list)
                    hist_proba = clf_hist.predict_proba(X_list[35])
                    Y_predict, _,_,_ = find_15_best(hist_proba)

                    # trying to maximise precision, I think it is best score function here
                    new_precision = precision_score(Y_list[35], Y_predict)
                    if new_precision > old_precision:
                        best_mi = m_i; best_lr = l_r; best_md = m_d
                        old_precision = new_precision

        print(f"precision:,{old_precision},best mi:,{best_mi},best lr:,{best_lr},best md:,{best_md}")

    return best_parameters_hist

def ensembling_classifiers(X_list, Y_list, choose_method):

    X_combined_df, Y_combined_list = merge_all_data(X_list, Y_list)
    ############# first voting
    if choose_method == 0:
        df = pd.read_csv("stats/GridSearchHistogram.csv")
        columns = ["m_i","l_r","m_d"]
        parameters = df[columns]

        list_of_classifiers = [(f"clf{season}" ,ensemble.HistGradientBoostingClassifier(max_iter=parameters.loc[season, 'm_i'],
                                                                    learning_rate=parameters.loc[season, 'l_r'],
                                                                    max_depth=parameters.loc[season, 'm_d'])) for season in range(34)]
        
        clf_vote = ensemble.VotingClassifier(estimators=list_of_classifiers, voting="soft", n_jobs=-1, verbose=False)
       
        clf_vote.fit(X_combined_df, Y_combined_list)
        vote_proba = clf_vote.predict_proba(X_list[35])
        Y_predict, first_team, second_team, third_team = find_15_best(vote_proba)

        first_team_ids = [X_list[35].iloc[i]['PLAYER_ID'] for i in range(len(first_team)) if first_team[i] == 1]
        second_team_ids = [X_list[35].iloc[i]['PLAYER_ID'] for i in range(len(second_team)) if second_team[i] == 1]
        third_team_ids = [X_list[35].iloc[i]['PLAYER_ID'] for i in range(len(third_team)) if third_team[i] == 1]
        print("pierwszy voting")
        print(get_names_from_id(first_team_ids))
        print(get_names_from_id(second_team_ids))
        print(get_names_from_id(third_team_ids))
        ############# end first voting
    ############# second histogram voting
    if choose_method == 1:
        df2 = pd.read_csv("stats/betterSearchHistogram.csv",keep_default_na=False,na_values=['NaN'])
        columns2 = ["m_i","l_r","m_d"]
        parameters2 = df2[columns2]

        list_of_classifiers2 = [(f"clf{season}" ,ensemble.HistGradientBoostingClassifier(max_iter=parameters2.loc[season, 'm_i'],
                                                                    learning_rate=parameters2.loc[season, 'l_r'],
                                                                    max_depth=parameters2.loc[season, 'm_d'])) for season in range(34)]
        
        clf_vote2 = ensemble.VotingClassifier(estimators=list_of_classifiers2, voting="soft", n_jobs=-1, verbose=False)

        clf_vote2.fit(X_combined_df, Y_combined_list)
        vote_proba2 = clf_vote2.predict_proba(X_list[35])
        Y_predict, first_team, second_team, third_team = find_15_best(vote_proba2)

        first_team_ids = [X_list[35].iloc[i]['PLAYER_ID'] for i in range(len(first_team)) if first_team[i] == 1]
        second_team_ids = [X_list[35].iloc[i]['PLAYER_ID'] for i in range(len(second_team)) if second_team[i] == 1]
        third_team_ids = [X_list[35].iloc[i]['PLAYER_ID'] for i in range(len(third_team)) if third_team[i] == 1]
        print("niby lepszy voting")
        print(get_names_from_id(first_team_ids))
        print(get_names_from_id(second_team_ids))
        print(get_names_from_id(third_team_ids))
        ############# end second histogram voting
    ############# normal hist Gradient boosting classifier
    if choose_method == 2:
        # best parameters for altogether combined statistic from 1988 to 2021, predicting 2022 with 12/15 precision
        # max_iter=6 ,learning_rate=0.12 , max_depth=8
        # best for 2023-24 season:
        # max_iter=33,learning_rate=0.55,max_depth=11
        clf_hist = ensemble.HistGradientBoostingClassifier(max_iter=33,learning_rate=0.55,max_depth=11)
        clf_hist.fit(X_combined_df, Y_combined_list)
        hist_proba = clf_hist.predict_proba(X_list[35])
        Y_predict, first_team, second_team, third_team = find_15_best(hist_proba)


        first_team_ids = [X_list[35].iloc[i]['PLAYER_ID'] for i in range(len(first_team)) if first_team[i] == 1]
        second_team_ids = [X_list[35].iloc[i]['PLAYER_ID'] for i in range(len(second_team)) if second_team[i] == 1]
        third_team_ids = [X_list[35].iloc[i]['PLAYER_ID'] for i in range(len(third_team)) if third_team[i] == 1]
        print("zwyczajny hist grad boosting")
        print(get_names_from_id(first_team_ids))
        print(get_names_from_id(second_team_ids))
        print(get_names_from_id(third_team_ids))

    cm = confusion_matrix(Y_list[35], Y_predict)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.show()

    return Y_predict


def get_names_from_Y_list(X, all_nba_or_not):
    # Konwersja listy do serii pandas, aby móc używać jej jako maski
    mask = pd.Series(all_nba_or_not)

    # Zastosowanie maski do dataframe
    player_ids = X['PLAYER_ID'][mask == 1].tolist()

    return get_names_from_id(player_ids)

def main():
    # 0 is 1988-89 season
    # 34 is 2022-2023 season 
    # 35 is our prediction, 2023-24 season
    X_list = reading_from_csv()
    Y_list = creating_list_of_all_nba_teams(X_list)

    # visualisation(X_list, Y_list)

    # best = get_best_hyper_parameters(X_list, Y_list, 2)
    # print(len(best))
    # with open(r'stats/betterSearchHistogram.csv', 'w') as fp:
    #     for item in best:
    #         # write each item on a new line
    #         fp.write("%s\n" % item)
    #     print('Done')

    Y_predict = ensembling_classifiers(X_list, Y_list, 2)
    get_names_from_Y_list(X_list[35], Y_predict)

    # os.system('say "your program has finished! Come to me my master!"')


main()


# !!! didnt use that at the end
# def most_frequent(List):
#     counter = 0
#     num = List[0]
#     for i in List:
#         curr_frequency = List.count(i)
#         if(curr_frequency> counter):
#             counter = curr_frequency
#             num = i
#     return num

# # !!! didnt use that at the end
# def find_5_best_pred_for_each_team(dictionary, array):
#     # function searches for 5, 10, 15 best picks all nba teams, given probabilities from predict_proba
#     # then creates (and returns) a list in a way, that Y is created by searching_through_df
#     # so we can compare predictions and true values of Y
#     all_nba_or_not = [0] * len(array)
#     # Using argpartition to find index of 5,10,15 biggest values of probablities
#     top_5_indices = np.argpartition(array[:, 1], -5)[-5:]
#     top_10_indices = np.argpartition(array[:, 2], -10)[-10:]
#     top_15_indices = np.argpartition(array[:, 3], -15)[-15:]
#     # sorting indexes in order of the biggest values (biggest probabilities last)
#     sorted_top_5_indices = top_5_indices[np.argsort(array[top_5_indices, 1])]
#     sorted_top_10_indices = top_10_indices[np.argsort(array[top_10_indices, 2])]
#     sorted_top_15_indices = top_15_indices[np.argsort(array[top_15_indices, 3])]
#     # changing orded to get indexes diminishing values (biggest probabilities first)
#     top_5_indices_team1 = sorted_top_5_indices[::-1].tolist()
#     top_10_indices_team2 = sorted_top_10_indices[::-1]
#     top_15_indices_team3 = sorted_top_15_indices[::-1]

#     # checking if the biggest probabilities of two all nba teams did not occur for the same index
#     top_indices_team2 = [x for x in top_10_indices_team2 if x not in top_5_indices_team1]
#     top_indices_team3 = [x for x in top_15_indices_team3 if x not in top_5_indices_team1 and x not in top_10_indices_team2]
#     top_5_indices_team2 = top_indices_team2[:5]
#     top_5_indices_team3 = top_indices_team3[:5]

#     # getting 5 highest values for 5 indexes
#     top_5_values_team1 = array[top_5_indices_team1, 1]
#     top_5_values_team2 = array[top_5_indices_team2, 2]
#     top_5_values_team3 = array[top_5_indices_team3, 3]
    
#     # for right indexes save which all nba team 
#     for i in top_5_indices_team1: all_nba_or_not[i] = dictionary["first team"]
#     for j in top_5_indices_team2: all_nba_or_not[j] = dictionary["second team"]
#     for k in top_5_indices_team3: all_nba_or_not[k] = dictionary["third team"]

#     return all_nba_or_not



# to function get_best_hyper_parameters: 

#  if choose_method==3:
#             random_search = RandomizedSearchCV(
#                             ensemble.HistGradientBoostingClassifier(),
#                             param_distributions=param_distributions,
#                             n_iter=500,
#                             scoring='precision',
#                             random_state=42,
#                             n_jobs=-1)
#             random_search.fit(X_combined_df, Y_combined_list)
#             best_parameters = random_search.best_params_
            

#             hist_proba = random_search.predict_proba(X_season)
#             Y_predict = find_15_best(hist_proba)
#             precision = precision_score(Y_season, Y_predict)

#             print(f"precision: {precision}, best_parameters:{best_parameters}")
#             best_parameters_hist.append(f"precision:,{random_search.best_score_},best_parameters:,{best_parameters}")