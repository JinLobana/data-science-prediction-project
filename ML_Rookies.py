from scipy.stats import zscore
from sklearn import datasets
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, HalvingGridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, mean_absolute_error, make_scorer, f1_score, precision_score
from sklearn import tree

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
import json

from nba_api.stats.static import players, teams


def reading_from_csv():
    # reading stats from csv
    # creating list of dataframes containing avg statistics 
    dfs = []
    for i in range(1997,2024):

            df = pd.read_csv("stats/rookies/"+str(i))
            # print(df)
            df.pop("Unnamed: 0")
            # so I delete every player with less than 20min
            # mask = (df["MIN"] > 25) & (df['no_games_played'] > 40) #& (df['PTS'] > 7)
            # df_filtered = df[mask]
            df = df.reset_index(drop=True)
            dfs.append(df)

            # visualizing missing values
            # msno.matrix(df)
    return dfs

def reading_all_rookies():
    # reading csv with length of no. of rookies, where 1.0 means first or second all rookies team
    dfs = []
    for i in range(1997,2024):

        df = pd.read_csv("stats/rookies/all_rookies/"+str(i))
        df.pop("Unnamed: 0")
        df = df.reset_index(drop=True)
        dfs.append(df)

    return dfs

def find_10_best(array):
    # function gets array of probability of belongingness to every class
    # it's a returned array from a method predict_proba()

    # creating lists of 0 of array's length
    all_nba_or_not = [0] * len(array)
    first_team_or_not = [0] * len(array)
    second_team_or_not = [0] * len(array)
    # choosing indices for highest probability of belongingness
    top_10_indices = np.argpartition(array[:, 1], -10)[-10:]
    # assignment to a specific team
    first_team = top_10_indices[5:]
    second_team = top_10_indices[:5]
    
    # giving 1 for right indices 
    for i in top_10_indices: all_nba_or_not[i] = 1
    for i in first_team: first_team_or_not[i] = 1
    for i in second_team:  second_team_or_not[i] = 1

    return all_nba_or_not, first_team_or_not, second_team_or_not

def merge_all_data(X_list, Y_list):
    # Creates combined statistics of all seasons
    # The same for output data
    X_list_train = X_list[:26];  Y_list_train = Y_list[:26]
    print("len of train combained dataframe:", len(X_list_train), len(Y_list_train))
    X_combined_df = pd.DataFrame()
    for season in range(len(X_list_train)):
        X_combined_df = pd.concat([X_combined_df, X_list_train[season]], axis=0, ignore_index=True)
    X_combined_df.to_csv("stats/X_combined_rookies.csv")

    Y_combined_df = pd.DataFrame()
    for season in range(len(Y_list_train)):
        Y_combined_df = pd.concat([Y_combined_df, Y_list_train[season]], axis=0, ignore_index=True)
    Y_combined_df.to_csv("stats/Y_combined_rookies_df.csv")

    return X_combined_df, Y_combined_df

def get_best_hyper_parameters(X_list, Y_list):
    # looking for the parameters that maximalise precision
    # default:  max_iter = 100, learning_rate = 0.1, max_depth = None, max_leaf_nodes = 31, l2_regularization = 0.0
    best_parameters_hist = []

    # list of hyper parameters for Grid Search
    max_ite = [int(x) for x in np.linspace(2,37,36)]
    learning_rate = [np.round(x, 3) for x in np.linspace(0.1,0.7,61)]
    max_depth = [int(x) for x in np.linspace(1,13,12)]
    max_depth.append(None)
    
    X_combined_df, Y_combined_df = merge_all_data(X_list, Y_list)
    # Make a vector from output dataframe
    Y_combined_list = Y_combined_df['ALL_ROOKIE_TEAM_NUMBER'].values.ravel()
    
    old_precision = 0.0
    # custom Grid Search
    for m_i in max_ite:
        for l_r in learning_rate:
            for m_d in max_depth:

                clf_hist = ensemble.HistGradientBoostingClassifier(max_iter = m_i, learning_rate = l_r, max_depth = m_d)
                clf_hist.fit(X_combined_df, Y_combined_list)
                hist_proba = clf_hist.predict_proba(X_list[26])
                Y_predict, _,_ = find_10_best(hist_proba)

                # trying to maximise precision, I think it is the best score function here
                new_precision = precision_score(Y_list[26], Y_predict)
                if new_precision > old_precision:
                    best_mi = m_i; best_lr = l_r; best_md = m_d
                    old_precision = new_precision

    print(f"precision:,{old_precision},best mi:,{best_mi},best lr:,{best_lr},best md:,{best_md}")

    return best_parameters_hist

def get_names_from_id(ids):
    all_nba_names = []
    for id in ids:
        dictionary = players.find_player_by_id(int(id))
        all_nba_names.append(dictionary.get('full_name'))
    return all_nba_names

def save_to_json(team1, team2):
    # saving teams to json file as dictionary
    all_rookies_teams = {
        "first rookie all-nba team": team1,
        "second rookie all-nba team": team2
    }
    with open("Lubina_Jan_rookies.json", 'w') as file:
        json.dump(all_rookies_teams, file, indent=2)


def ML(X_list,Y_list):
    # best parameters for altogether combined statistic from 1997 to 2021, predicting 2022 with 9/10y precision

    # best for 2023-24 season:
    # max_iter=6 ,learning_rate=0.12 , max_depth=8

    # Machine learning
    X_combined_df, Y_combined_df = merge_all_data(X_list, Y_list)
    Y_combined_list = Y_combined_df['ALL_ROOKIE_TEAM_NUMBER'].values.ravel()
    clf_hist = ensemble.HistGradientBoostingClassifier(max_iter=6 ,learning_rate=0.12 , max_depth=8)
    clf_hist.fit(X_combined_df, Y_combined_list)

    # proediction with probabilities
    hist_proba = clf_hist.predict_proba(X_list[26])
    Y_predict, first_team, second_team = find_10_best(hist_proba)

    # looking for players ids for all rookies teams
    first_team_ids = [X_list[26].iloc[i]['PLAYER_ID'] for i in range(len(first_team)) if first_team[i] == 1]
    second_team_ids = [X_list[26].iloc[i]['PLAYER_ID'] for i in range(len(second_team)) if second_team[i] == 1]
    
    print(get_names_from_id(first_team_ids))
    print(get_names_from_id(second_team_ids))
    print("precision", precision_score(Y_list[26], Y_predict))

    #visualisation of precision, confusion matrix
    # cm = confusion_matrix(Y_list[26], Y_predict)
    # cm_display = ConfusionMatrixDisplay(cm).plot()
    # plt.show()


    save_to_json(get_names_from_id(first_team_ids),get_names_from_id(second_team_ids))

    return Y_predict

def get_ids_from_names(names):
    # list of names as an input, output is information about that player including his Player_ID
    ids = []
    for name in names:
        dictionary = players.find_players_by_full_name(name)
        dictionary = dictionary[0]
        ids.append(dictionary["id"])
        print(dictionary["id"])
    return ids

def main():
    # index 0 is for year 1997-98, 26 for 2023-24
    X_list = reading_from_csv()
    Y_list = reading_all_rookies()

    # get_ids_from_names(["Chet Holmgren", "Victor Wembanyama","Brandon Miller","Jaime Jaquez Jr.", "Brandin Podziemski", "Dereck Lively", "Amen Thompson", "Keyonte George", "Cason Wallace", "GG Jackson"])

    # get_best_hyper_parameters(X_list, Y_list)

    predict = ML(X_list, Y_list)

if __name__ == "__main__":
        main()