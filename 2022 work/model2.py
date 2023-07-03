from audioop import avg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
import random


def calculate_ESPN_score(test_y, predictions, teams, year):
    points = 0
    # Dictionary with final bracket
    # Correctly predicted teams that got to the xth round
    final_bracket = {"R32":[], "S16":[], "E8":[], "F4":[], "Finals":[], "Champions":[]}
    
    tournament_results = pd.read_csv("tournament_scores.csv")
    #tournament_results = pd.read_csv("test_games_data.csv")
    tournament_results = tournament_results[tournament_results['YEAR'] == year]
    # Loop through rounds
    points_arr = [10,30,70,150,310,630]
    rounds = { "1": "R32", "2":"S16", "3": "E8", "4":"F4", "5":"Finals", "6": "Champions"}
    
    roundToInt = {"R68":0, "R64":1,"R32":2,"S16":3, "E8": 4, "F4": 5, "2ND":6, "Champions":7}
    
    teams_left = {} # USED to make sure that all predicitions we are checking are valid teams - value of 1 means they are still valid
    for team in teams:
        teams_left[team] = 1
    
    
    #for i in range(len(predictions)):
        #print("Team: " + teams[i] + "  Prediction: " + predictions[i])
        
    
    # Get rid of teams who lost in R68
    for i in range(len(test_y)):
        if test_y[i] == "R68":
            teams_left[teams[i]] = 0
    
    # Calculating points by round
    wrong_team_preds = {}  # PREDICTIONS that are what the team in my bracket would have had - however this team has already lost
    for round in range(1,7):
        correct = []
        games = tournament_results[tournament_results['ROUND'] == round] # Data for a specific round
        
       
        for i,row in games.iterrows():
            winner = row["WTEAM"]
           
            loser = row["LTEAM"]
            
            #if winner not in checked_teams:
            # Check if that would be predicted
            # Find the index of the two teams in the teams array and find corresponding prediction
            
            indexW = np.where(teams == winner)[0][0]
            
            
            
            indexL = np.where(teams == loser)[0][0]

            #Predict game based on my prediction
            if teams[indexW] not in wrong_team_preds:    #IF WINNING TEAM is in wrong_team_preds then we know we CAN NOT predict the game correct # do i need to do anything to teams_left???
                pred_winning_team = predictions[indexW]
            else: 
                pred_winning_team = wrong_team_preds[teams[indexW]]
            if teams[indexL] not in wrong_team_preds:
                pred_losing_team = predictions[indexL]
            else: 
                pred_losing_team = wrong_team_preds[teams[indexL]] # CHECK TO SEE IF I WOULD PREDICT THE LOSING TEAM TO WIN THE GAME
            
            #print("---")
            #print(teams[indexW] + ": " + pred_winning_team)
            #print(teams[indexL] + ": " + pred_losing_team)
            #print("---")
            # Correct Prediction
            
            ####  and (teams[indexW] not in wrong_team_preds) additional condition that should be same as teams_left[teams[indexW]] == 1
            if roundToInt[pred_winning_team] > roundToInt[pred_losing_team] and teams_left[teams[indexW]] == 1 :
                
                if teams[indexW] not in wrong_team_preds:
                    points += points_arr[round-1]
                teams_left[teams[indexL]] = 0 # THEY ARE DONE

                correct.append(teams[indexW])

            # Two teams are tied in my prediciton
            elif roundToInt[pred_winning_team] == roundToInt[pred_losing_team] and teams_left[teams[indexW]] == 1:
                rand = random.choice([0,1]) # Randomly select the prediction

                # We'll say 0 will be the case when winner is chosen by prediction
                if rand == 0:
                    if teams[indexW] not in wrong_team_preds:
                        points += points_arr[round-1]
                    correct.append(teams[indexW])

                    teams_left[teams[indexL]] = 0
                else:
                    teams_left[teams[indexL]] = 0
                    teams_left[teams[indexW]] = 0

                    wrong_team_preds[teams[indexW]] = pred_losing_team #losing ????????????????



            # Wrong prediction
            else: 
                teams_left[teams[indexW]] = 0
                teams_left[teams[indexL]] = 0
                
                # if already in there don't update???? or update only if higher
                if teams[indexW] not in wrong_team_preds or roundToInt[pred_losing_team] >  roundToInt[wrong_team_preds[teams[indexW]]]:
                    wrong_team_preds[teams[indexW]] = pred_losing_team
        
        #print(wrong_team_preds)
        
        #print(correct)
        for key,value in teams_left.items():
            if value == 1:
                #print(key)
          
                final_bracket[rounds[str(round)]].append(key)
        #print(teams_left)
    print('----------------')
    print("Year: " + str(year))
    print("Points: " + str(points))
    print(final_bracket)
    print('----------------')
    
    return points    
    
    
    
            
"""
    Making predictions for each year based off optimal k-value
"""
def predict_brackets():
   
    data = pd.read_csv("model_data.csv")
    #data = data.dropna() #because 2022 was missing data
    data['Win %'] = data['Wins']/data['Games']
    
    years = list(data['YEAR'].unique())
    years.remove(2021) # 2021 tournement games data is funky rn
    
    # ALL FEATURES
    #features = ['Rank', 'SEED', 'Games', 'Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', 'ADJ T.', 'WAB', 'Win %']  ALL FEATURES
    
    # features with correlation above .1
    #features = ['Rank', 'SEED', 'Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'TOR', 'ORB', 'WAB', 'Win %'] 
    
    # features with correlation above .1 NORMALIZED
    features = ['Rank', 'SEED', 'Wins', 'ADJOE', 'ADJDE', 'BARTHAG', 'TOR', 'ORB', 'WAB', 'Win %'] 
    
    # features with correlation above .05 NORMALIZED
    #features = ['Rank', 'SEED', 'Games', 'Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFGD%', 'TOR', 'ORB', 'FTRD', '2P%', '3P%D', 'WAB', 'Win %']
    
    # features with correlation above .15 NORMALIZED
    #features = ['SEED', 'ADJOE', 'WAB'] # Pretty solid -- similar to .1
    total_points = 0
    
    years = [2022, 2019,2018, 2017, 2016, 2015, 2014,2013,2012,2010]
    for year in years:        
        #Each year will be used to cross validate
        train = data[data['YEAR'] != year]
        test = data[data['YEAR'] == year]
        
        train_x = train[features]
      
        test_x = test[features]
        train_y = train.loc[:, "POSTSEASON"]
        test_y = test.loc[:, "POSTSEASON"].values  


        #NORMALIZE - only features since the prediction is categorical
        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)

        scaler.fit(test_x)
        test_x = scaler.transform(test_x)

        
        k = 9
        
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train_x, train_y)

        predictions = neigh.predict(test_x)

        teams = test[['TEAM']].values
        teams = teams.flatten()

        points = calculate_ESPN_score(test_y, predictions, teams, year)
        
        total_points += points
        
    
        #for i in range(len(predictions)):
            #print("Team: " + teams[i] + "  Prediction: " + predictions[i])
    
    
    cols = {'Team': [],'Prediction': [],'Outcome': []}

    #create dataframe
    df = pd.DataFrame(cols)

    for i in range(len(predictions)):
        newRow = {"Team": teams[i], "Prediction": predictions[i], "Outcome": test_y[i]}
        df = df.append(newRow, ignore_index=True)

    df.to_csv('2022 Predictions (Model2).csv')
    
    
    avg_points = total_points/len(years)
    print("AVG POINTS: " + str(avg_points))
            


def get_correlation(data, threshold):

    data['POSTSEASON'] = pd.factorize(data['POSTSEASON'])[0]
    
    # NORMALIZE
    scaler = StandardScaler()
    scaler.fit(data)
    train_x = scaler.transform(data)
    
    #print(data)
    corr = data.corr()
    
    # Find the subset of futures that are most correlated to the outcome
    corr = corr[['POSTSEASON']]
    
    corr = corr.loc[abs(corr['POSTSEASON']) >= threshold]
    
    best_features = list(corr.index.values.tolist())
    
    best_features.remove('POSTSEASON')
    
    return best_features



"""
    Using k-fold cross validation to train based off official espn scoring (find best k value)
"""
def cross_validation_train():
   
    data = pd.read_csv("model_data.csv")
    data = data.dropna() #ADD 2022 RESULTS!!!!!!!!!!!!!!!
    data['Win %'] = data['Wins']/data['Games']
    
    years = list(data['YEAR'].unique())
    years.remove(2021) # 2021 tournement games data is funky rn
    
    features = ['Rank', 'SEED', 'Games', 'Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', 'ADJ T.', 'WAB', 'Win %']
    #list(data.columns.values)
    
    k_vals = []
    score_vals = []
    for year in years:
        
        
        #Each year will be used to cross validate
        train = data[data['YEAR'] != year]
        test = data[data['YEAR'] == year]
        
        train_x = train[features]
      
        test_x = test[features]
        train_y = train.loc[:, "POSTSEASON"]
        test_y = test.loc[:, "POSTSEASON"].values  


        #NORMALIZE - only features since the prediction is categorical
        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)

        scaler.fit(test_x)
        test_x = scaler.transform(test_x)

        # Experiment with different k values
        # Run 10 times for each year
        avg_k = 0
        avg_score = 0
        for i in range(10):
            best_k = 0
            best_score = 0
            for k in range(1,15):
                neigh = KNeighborsClassifier(n_neighbors=k)
                neigh.fit(train_x, train_y)

                predictions = neigh.predict(test_x)

                teams = test[['TEAM']].values
                teams = teams.flatten()

                points = calculate_ESPN_score(test_y, predictions, teams, year)
                
                if points > best_score:
                    best_score = points
                    best_k = k
            
            avg_k += best_k
            avg_score += best_score
        avg_k /= 10
        avg_score /= 10
        k_vals.append(avg_k)
        score_vals.append(avg_score)
        
        print(year)
    k_best = sum(k_vals)/len(k_vals)
    score_best = sum(score_vals)/len(score_vals)
    print(k_best) # 8.733333333333333 => 9
    print(score_best) # 1739.1666666666667
    




def main():
    data = pd.read_csv("model_data.csv")
    data = data.dropna() 
    data['Win %'] = data['Wins']/data['Games']

    years = list(data['YEAR'].unique())
    years.remove(2021) # 2021 tournement games data is funky rn
    
    #ALL cols in data
    cols = ['Rank', 'SEED', 'Games', 'Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', 'ADJ T.', 'WAB', 'Win %', 'POSTSEASON']
    
    #print(get_correlation(data[cols], 0.15))
    
    predict_brackets()

    #cross_validation_train()
    #test(['SEED', 'Win %', 'ADJOE', 'BARTHAG', 'EFGD%', 'TOR', '3P%D', 'ADJ T.', 'Losses', 'ORB', 'DRB', '2P%D', 'WAB','Rank'])
    
if __name__ == "__main__":
    main()


