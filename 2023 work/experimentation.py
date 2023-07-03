"""
    March Madness Prediction model
    Tournament scores data since 1985: https://www.kaggle.com/datasets/woodygilbertson/ncaam-march-madness-scores-19852021

"""




from audioop import avg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
import random
from sklearn.feature_selection import SelectKBest, f_classif



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

    added_points = 10
    round = 1
    points=0
    for key,value in final_bracket.items():
        points += added_points*len(value)
        added_points*=2

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
    data = data[data['YEAR'] != 2023]
    data = data[data['YEAR'] != 2021]
    years = list(data['YEAR'].unique())
    #years.remove(2021) # 2021 tournement games data is funky rn
    
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
    
    years = [2019,2018, 2017, 2016, 2015, 2014,2013,2012, 2010, 2022] #2011 not included because it was an unusual year
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

        
        k = 8
        
        neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
        neigh.fit(train_x, train_y)

        predictions = neigh.predict(test_x)        

        ####
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

    df.to_csv(f"{year} Predictions (Model2).csv")
    
    
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


def feature_selector(data):
    
    # Split data into input features and target variable
    X = data.drop('POSTSEASON', axis=1)
    y = data['POSTSEASON']

    # Use correlation-based feature selection to select the top k features
    selector = SelectKBest(f_classif, k=10)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()]

    print(selected_features)


"""
    Using k-fold cross validation to train based off official espn scoring (find best k value)
"""
def cross_validation_train():
   
    data = pd.read_csv("model_data.csv")
    data = data.dropna() 
    data['Win %'] = data['Wins']/data['Games']
    data = data[data['YEAR'] != 2023]
    data = data[data['YEAR'] != 2021]

    ########### REMOVE 2022 so it isn't tainted
    #data = data[data['YEAR'] != 2022]
    ###########

    years = list(data['YEAR'].unique())
    #years.remove(2021) # 2021 tournement games data is funky rn
    
    #features = ['Rank', 'SEED', 'Games', 'Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', 'ADJ T.', 'WAB', 'Win %']
    features = ['SEED', 'Wins', 'BARTHAG', 'ADJOE', 'ADJDE', 'EFGD%', 'ORB', '2P%','2P%D', 'WAB']
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
                neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
                neigh.fit(train_x, train_y)

                predictions = neigh.predict(test_x)
                ########################
                k_nearest_neighbors = neigh.kneighbors(test_x, return_distance=False)
                print(k_nearest_neighbors)
                # Print out the k-nearest neighbors used in prediction
                mapping = {"R68": 0, "R64": 0, "R32": 1, "S16": 2, "E8": 3, "F4": 4, "2ND": 5, "Champions": 6}
                cols = {'Team': [],'Prediction': []}
                #create dataframe
                df = pd.DataFrame(cols)
                

                print("The k-nearest neighbors used in prediction are:")
                preds = []
                for i in range(len(k_nearest_neighbors)): #len of test_x
                    
                    score = 0
                    for team_index in k_nearest_neighbors[i]:
                        score += mapping[train_y.iloc[team_index]]
                        #print(teams_train.iloc[team_index, 0] + f" ({teams_train.iloc[team_index,1]} {teams_train.iloc[team_index,2]}), ", end="")
                    prediction = score/k
                    prediction = round(prediction)
                    mapdos = {0:'R64', 1:'R32', 2:'S16', 3:'E8', 4:'F4', 5:'2ND', 6:'Champions'}
                    pred_round = mapdos[prediction]
                    predictions[i] = pred_round
                    print("---------------------------------")
                ##############3
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
    


"""
    Using k-fold cross validation to train based off official espn scoring (find best k value)
"""
def cross_validation_train_new():
   
    data = pd.read_csv("model_data.csv")
    data = data.dropna() 
    data['Win %'] = data['Wins']/data['Games']
    data = data[data['YEAR'] != 2023]
    data = data[data['YEAR'] != 2021]


    years = list(data['YEAR'].unique())
    
    #features = ['SEED', 'Wins', 'BARTHAG', 'ADJOE', 'ADJDE', 'EFG%', 'EFGD%', 'TOR','ORB', '2P%', '2P%D', 'WAB']
    features = ['SEED', 'Wins', 'BARTHAG', 'ADJOE', 'ADJDE', 'EFGD%', 'ORB', '2P%','2P%D', 'WAB']
    
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
                neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
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
    
    #predict_brackets()

    #cross_validation_train()
    #test(['SEED', 'Win %', 'ADJOE', 'BARTHAG', 'EFGD%', 'TOR', '3P%D', 'ADJ T.', 'Losses', 'ORB', 'DRB', '2P%D', 'WAB','Rank'])
    
    cross_validation_train()
    # 5 features: k-8.807692307692307 points-1006.7692307692307
    # 10 features: k-8.169230769230769 points-1038.6923076923076 # Rocking with this one
    # 15 features: k-8.607692307692307 points-993.9230769230769
    # 8 features: k-9.553846153846154 points 1040.8461538461538
    # 12 features: k-8.823076923076922 points-931.3076923076923
    # Better 10 : ['SEED', 'Games', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFGD%', 'ORB', '2P%D', 'WAB', 'Win %'] -- gets rid of rank and wins and losses 

    """#FEATURE SELECTION
    data = pd.read_csv("model_data.csv")
    #data = data.dropna() #because 2022 was missing data
    
    data['Win %'] = data['Wins']/data['Games']

    #cols = [ 'SEED', 'Games',  'ADJOE', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', 'ADJ T.', 'WAB', 'Win %', 'POSTSEASON']
    cols = [ 'SEED', 'Wins', 'BARTHAG',  'ADJOE', 'ADJDE', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', 'ADJ T.', 'WAB',  'POSTSEASON']
    data = data[data['YEAR'] != 2023]
    data = data[data['YEAR'] != 2021]


    data = data[cols]

    feature_selector(data)
    # 5: ['Rank', 'SEED', 'ADJDE', 'BARTHAG', 'WAB']
    # 10: ['Rank', 'SEED', 'Games', 'Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG','WAB', 'Win %'] 
    # 15: ['Rank', 'SEED', 'Games', 'Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'ORB', '2P%', '2P%D', 'WAB', 'Win %']
    # 12: ['Rank', 'SEED', 'Games', 'Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFGD%', '2P%D', 'WAB', 'Win %']
    # 8: ['Rank', 'SEED', 'Wins', 'ADJOE', 'ADJDE', 'BARTHAG', 'WAB', 'Win %']
    # 10: ['Rank', 'SEED',  'Wins', 'EFGD%', '2P%D', 'ADJOE', 'ADJDE', 'BARTHAG','WAB', 'Win %'] 
    """
if __name__ == "__main__":
    main()

# Explaining and predicting
## Explain the data
## Explain the cross-validation training 
## Explain the ESPN scoring fitness function
## Final parameters and features
## Test results
## Predicting this year's bracket

#UPDATE it so testing runs for each year to get an untainted score based off cross-validation training on all other years

