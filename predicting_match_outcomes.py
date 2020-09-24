"""
This file attempts to predict the outcomes of soccer matches using data
obtained from top-flight European soccer leagues. Predictions are made
using parameters such as team win rate, home/away performance, average
player rating (on FIFA), highest/lowest player rating, etc.
"""

import numpy as np
import pandas as pd
import featuretools as ft
import sqlite3
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import RFE

database = "datasets/database.sqlite"
conn = sqlite3.connect(database)

teams = pd.read_sql("SELECT * FROM TEAM", conn)
players = pd.read_sql("SELECT * FROM PLAYER_ATTRIBUTES", conn)
matches = pd.read_sql("SELECT * FROM MATCH", conn)

matches['date'] = pd.to_datetime(matches['date'], format='%Y-%m-%d 00:00:00')

home_players = ["home_player_" + str(i) for i in range(1, 12)]
away_players = ["away_player_" + str(i) for i in range(1, 12)]

match_cols = ["id", "date", "home_team_api_id", "away_team_api_id", "home_team_goal", "away_team_goal"] + home_players + away_players
matches = matches[match_cols]

matches["goal_diff"] = matches["home_team_goal"] - matches["away_team_goal"]
matches["result"] = "D" # "W" if home team wins "D" if it's a draw "L" if away team wins

matches["result"] = np.where(matches["goal_diff"] > 0, "W", matches["result"])
matches["result"] = np.where(matches["goal_diff"] < 0, "L", matches["result"])

for player in home_players:
    matches = pd.merge(matches, players[["id", "overall_rating"]], left_on=[player], right_on=["id"], suffixes=["", "_" + player])
for player in away_players:
    matches = pd.merge(matches, players[["id", "overall_rating"]], left_on=[player], right_on=["id"], suffixes=["", "_" + player])

matches = matches.rename(columns={"overall_rating": "overall_rating_home_player_1"})

matches = matches[ matches[["overall_rating_" + r for r in home_players]].isnull().sum(axis = 1) <= 0]
matches = matches[ matches[["overall_rating_" + r for r in away_players]].isnull().sum(axis = 1) <= 0]

matches["overall_rating_home"] = matches[["overall_rating_" + r for r in home_players]].sum(axis=1)
matches["overall_rating_away"] = matches[["overall_rating_" + r for r in away_players]].sum(axis=1)
matches["overall_rating_difference"] = matches["overall_rating_home"] - matches["overall_rating_away"]

matches["min_overall_rating_home"] = matches[["overall_rating_" + r for r in home_players]].min(axis=1)
matches["min_overall_rating_away"] = matches[["overall_rating_" + r for r in away_players]].min(axis=1)

matches["max_overall_rating_home"] = matches[["overall_rating_" + r for r in home_players]].max(axis=1)
matches["max_overall_rating_away"] = matches[["overall_rating_" + r for r in away_players]].max(axis=1)

matches["mean_overall_rating_home"] = matches[["overall_rating_" + r for r in home_players]].mean(axis=1)
matches["mean_overall_rating_away"] = matches[["overall_rating_" + r for r in away_players]].mean(axis=1)

matches["std_overall_rating_home"] = matches[["overall_rating_" + r for r in home_players]].std(axis=1)
matches["std_overall_rating_away"] = matches[["overall_rating_" + r for r in away_players]].std(axis=1)

for col in matches.columns:
    if "_player_" in col:
        matches = matches.drop(col, axis=1)

ct_home_matches = pd.DataFrame()
ct_away_matches = pd.DataFrame()

ct_matches = pd.DataFrame()

# Trick to exclude current match from statistics and do not bias predictions
ct_home_matches['time'] = matches['date'] - pd.Timedelta(hours=1)
ct_home_matches['instance_id'] = matches['home_team_api_id']
ct_home_matches['label'] = (ct_home_matches['instance_id'] == ct_home_matches['instance_id'])

# Trick to exclude current match from statistics and do not bias predictions
ct_away_matches['time'] = matches['date'] - pd.Timedelta(hours=1)
ct_away_matches['instance_id'] = matches['away_team_api_id']
ct_away_matches['label'] = (ct_away_matches['instance_id'] == ct_away_matches['instance_id'])

ct_matches = ct_home_matches.append(ct_away_matches)

# using featuretools to create new features
es = ft.EntitySet("entityset")

es.entity_from_dataframe(entity_id="home_matches",
                        index="id",
                        time_index="date",
                        dataframe=matches,
                        variable_types={"home_team_api_id": ft.variable_types.Categorical,
                                              "away_team_api_id": ft.variable_types.Categorical,
                                              "result": ft.variable_types.Categorical,
                                              "home_team_goal":     ft.variable_types.Numeric,
                                              "away_team_goal":     ft.variable_types.Numeric})

es.entity_from_dataframe(entity_id="away_matches",
                        index="id",
                        time_index="date",
                        dataframe=matches,
                        variable_types={"home_team_api_id": ft.variable_types.Categorical,
                                              "away_team_api_id": ft.variable_types.Categorical,
                                              "result": ft.variable_types.Categorical,
                                              "home_team_goal":     ft.variable_types.Numeric,
                                              "away_team_goal":     ft.variable_types.Numeric})

es.entity_from_dataframe(entity_id="teams",
                         index="team_api_id",
                         dataframe=teams)

es.add_last_time_indexes()

new_relationship = ft.Relationship(es["teams"]["team_api_id"],
                                   es["home_matches"]["home_team_api_id"])
es = es.add_relationship(new_relationship)

new_relationship = ft.Relationship(es["teams"]["team_api_id"],
                                   es["away_matches"]["away_team_api_id"])
es = es.add_relationship(new_relationship)

feature_matrix, features_defs = ft.dfs(entities=es,
                                       entityset=es,
                                       cutoff_time=ct_matches,
                                       cutoff_time_in_index=True,
                                       training_window='60 days',
                                       max_depth=3,
                                       target_entity="teams",
                                       verbose=True
                                      )

# Recover the true datetime 
feature_matrix = feature_matrix.reset_index()
feature_matrix['time'] = feature_matrix['time'] + pd.Timedelta(hours=1)

final = pd.merge(matches, feature_matrix, left_on=['date', 'home_team_api_id'], right_on=['time','team_api_id'], suffixes=('', '_HOME'))
final = pd.merge(final, feature_matrix, left_on=['date', 'away_team_api_id'], right_on=['time','team_api_id'], suffixes=('', '_AWAY'))

columns_to_drop = ["id", "team_fifa_api_id", "date", "team_long_name","team_long_name_AWAY", "team_short_name","team_short_name_AWAY", "result", "home_team_goal", "away_team_goal", "home_team_api_id", "away_team_api_id", "label_AWAY", "label", "goal_diff", 'team_api_id', 'time', 'team_api_id_AWAY', 'time_AWAY']

for col in final.columns:
    if 'MODE' in col:
        columns_to_drop.append(col)

# create test data
y = final["result"]
data = final.drop(columns_to_drop, axis=1)
data = data.fillna(0)

# Split X and y into a train and test set
X_train, X_test, y_train, y_test = train_test_split(data, y, shuffle=True, random_state=42)

# Select features using RFE
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
estimator = clf
selector = RFE(estimator, 10, step=1)
selector = selector.fit(X_train, y_train)

# Test accuracy using a random forest
clf.fit(selector.transform(X_train), y_train)

score_rf = clf.score(selector.transform(X_test), y_test)
y_pred_rf = clf.predict(selector.transform(X_test))

print("Random Forest score: {}".format(score_rf))

clf = SVC()
clf.fit(selector.transform(X_train), y_train)

score_svm = clf.score(selector.transform(X_test), y_test)
y_pred_svm = clf.predict(selector.transform(X_test))

print("SVM score: {}".format(score_svm))