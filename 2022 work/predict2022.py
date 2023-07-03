from audioop import avg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
import random


def calculate_possible_points(predicted_round, actual_round):
        rounds = {"R68": 0, "R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "2ND": 6, "Champions": 7 }

        predicted_num = rounds[predicted_round]
        actual_num = rounds[actual_round]

        #standard 1-2-4-8-16-32 system, ie you get 1 point for project a team to R32 since they won the first game
        points = [0,0,1,2,4,8,16,32]
        
        if predicted_num <= actual_num:
            total_points = sum(points[:predicted_num+1])
        elif predicted_num > actual_num:
            total_points = sum(points[:actual_num+1])
            
    
        
        return total_points

def predict_2022(year, k, features):
   
    data = pd.read_csv("model_data.csv")
    data['Win %'] = data['Wins']/data['Games']
    
    #X and Y Data
    train_data = data[data['YEAR'] != year]
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
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_features, train_classes)
    
    #Predict based on this years bracket teams' data
    predictions = neigh.predict(bracket_features)

    #Calculate possible points (BASED on predictions)
    #total_points = 0
    #for j in range(len(predictions)):
        #total_points += calculate_possible_points(predictions[j], bracket_classes[j])
    
    cols = {'Team': [],'Prediction': [],'Outcome': []}

    #create dataframe
    df = pd.DataFrame(cols)

    for i in range(len(predictions)):
        if predictions[i] == bracket_classes[i]:
            result = "TRUE"
        else:
            result = "FALSE"
        #print("Team: " + bracket_data["TEAM"].values[i] + "  Prediction: " + predictions[i] + "  Actual: " + bracket_classes[i] + "  => " + result)
        newRow = {"Team": bracket_data["TEAM"].values[i], "Prediction": predictions[i], "Outcome": result, "Outcome": bracket_classes[i]}
        df = df.append(newRow, ignore_index=True)
    
      
    return df


df = predict_2022(2022, 3, ['SEED', 'Win %', 'ADJOE', 'BARTHAG', 'EFGD%', 'TOR', '3P%D', 'ADJ T.', 'Losses', 'ORB', 'DRB', '2P%D', 'WAB','Rank'])
df.to_csv('2022 Predictions.csv')