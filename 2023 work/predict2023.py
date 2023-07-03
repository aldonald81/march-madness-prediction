from audioop import avg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
import random

def predict_2022(year, k, features):
   
    data = pd.read_csv("model_data.csv")
    data['Win %'] = data['Wins']/data['Games']
    
    #X and Y Data
    train_data = data[(data['YEAR'] != year) & (data['YEAR'] != 2021) ]
    
    bracket_data = data[data['YEAR'] == year]
    
    #Features
    train_features = train_data[features]
    bracket_features = bracket_data[features]
    
    #Classes
    train_classes = train_data.loc[:, "POSTSEASON"]
    bracket_classes = bracket_data.loc[:, "POSTSEASON"].values  
    
    
    #NORMALIZE - only features since the prediction is categorical
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)

    scaler.fit(bracket_features)
    bracket_features = scaler.transform(bracket_features)

    #Using sci-kit-learn K neighbors classification to predict
    neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
    neigh.fit(train_features, train_classes)
    
    #Predict based on this years bracket teams' data
    predictions = neigh.predict(bracket_features)

    # print("******************")
    teams_train = train_data[['TEAM', 'YEAR', 'POSTSEASON']]
    #teams_train = teams_train.flatten()
    print(teams_train)
    teams_test = bracket_data[['TEAM', 'YEAR']]
    #teams_test = teams_test.flatten()
    print(teams_test)
    k_nearest_neighbors = neigh.kneighbors(bracket_features, return_distance=False)
    print(k_nearest_neighbors)
    # Print out the k-nearest neighbors used in prediction
    mapping = {"R68": 0, "R64": 0, "R32": 1, "S16": 2, "E8": 3, "F4": 4, "2ND": 5, "Champions": 6}
    cols = {'Team': [],'Prediction': []}
    #create dataframe
    df = pd.DataFrame(cols)

    print("The k-nearest neighbors used in prediction are:")
    for i in range(len(k_nearest_neighbors)): #len of test_x
        print(teams_test.iloc[i,0] + ": ", end="")
        score = 0
        for team_index in k_nearest_neighbors[i]:
            score += mapping[teams_train.iloc[team_index,2]]
            print(teams_train.iloc[team_index, 0] + f" ({teams_train.iloc[team_index,1]} {teams_train.iloc[team_index,2]}), ", end="")
        prediction = score/8
        print(prediction)
        prediction = round(prediction)
        mapdos = {0:'R64', 1:'R32', 2:'S16', 3:'E8', 4:'F4', 5:'2ND', 6:'Champions'}
        pred_round = mapdos[prediction]
        newRow = {"Team": teams_test.iloc[i,0], "Prediction": pred_round}
        df = df.append(newRow, ignore_index=True)
        print("---------------------------------")
    print("******************")


    
    # cols = {'Team': [],'Prediction': [],'Outcome': []}

    # #create dataframe
    # df = pd.DataFrame(cols)

    # for i in range(len(predictions)):
    #     if predictions[i] == bracket_classes[i]:
    #         result = "TRUE"
    #     else:
    #         result = "FALSE"
    #     #print("Team: " + bracket_data["TEAM"].values[i] + "  Prediction: " + predictions[i] + "  Actual: " + bracket_classes[i] + "  => " + result)
    #     newRow = {"Team": bracket_data["TEAM"].values[i], "Prediction": predictions[i], "Outcome": result, "Outcome": bracket_classes[i]}
    #     df = df.append(newRow, ignore_index=True)
    
      
    return df
    
def get_neighbors(train_data, bracket_data, train_features, neigh):
    #### Get what teams are used in the prediction
    #Get the indices of the k-nearest neighbors used in prediction
    print("******************")
    teams_train = train_data[['TEAM', 'YEAR', 'POSTSEASON']]
    #teams_train = teams_train.flatten()
    print(teams_train)
    teams_test = bracket_data[['TEAM', 'YEAR']]
    #teams_test = teams_test.flatten()
    print(teams_test)

    features = ['SEED', 'Wins', 'BARTHAG', 'ADJOE', 'ADJDE', 'EFGD%', 'ORB', '2P%', '2P%D', 'WAB']

    bracket_data_x = bracket_data[features]
    k_nearest_neighbors = neigh.kneighbors(train_features, return_distance=False)
    print(k_nearest_neighbors)
    # Print out the k-nearest neighbors used in prediction
    print("The k-nearest neighbors used in prediction are:")
    for i in range(len(k_nearest_neighbors)): #len of test_x
        print(teams_test.iloc[i,0] + ": ", end="")
        for team_index in k_nearest_neighbors[i]:
            print(teams_train.iloc[team_index, 0] + f" ({teams_train.iloc[team_index,1]} {teams_train.iloc[team_index,2]}), ", end="")
        print("---------------------------------")
    print("******************")

    # Get the indices of the k-nearest neighbors used in prediction
        # print("******************")
        # teams_train = train[['TEAM', 'YEAR', 'POSTSEASON']]
        # #teams_train = teams_train.flatten()
        # print(teams_train)
        # teams_test = test[['TEAM', 'YEAR']]
        # #teams_test = teams_test.flatten()
        # print(teams_test)
        # k_nearest_neighbors = neigh.kneighbors(test_x, return_distance=False)
        # print(k_nearest_neighbors)
        # # Print out the k-nearest neighbors used in prediction
        # print("The k-nearest neighbors used in prediction are:")
        # for i in range(len(k_nearest_neighbors)): #len of test_x
        #     print(teams_test.iloc[i,0] + ": ", end="")
        #     for team_index in k_nearest_neighbors[i]:
        #         print(teams_train.iloc[team_index, 0] + f" ({teams_train.iloc[team_index,1]} {teams_train.iloc[team_index,2]}), ", end="")
        #     print("---------------------------------")
        # print("******************")

#df = predict_2022(2023, 9, ['Rank', 'SEED', 'Wins', 'ADJOE', 'ADJDE', 'BARTHAG', 'TOR', 'ORB', 'WAB', 'Win %']) # first bracket
#df = predict_2022(2023, 8, ['SEED', 'Games', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFGD%', 'ORB', '2P%D', 'WAB', 'Win %']) # second bracket
#df = predict_2022(2023, 8, ['SEED', 'Games', 'ADJOE', 'ADJDE', 'EFGD%', 'ORB', '2P%', '2P%D', 'WAB','Win %']) #most variable
df = predict_2022(2023, 8, ['SEED', 'Wins', 'BARTHAG', 'ADJOE', 'ADJDE', 'EFGD%', 'ORB', '2P%', '2P%D', 'WAB']) 
df.to_csv('2023 Predictions NEW.csv')