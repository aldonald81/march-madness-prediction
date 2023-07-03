from audioop import avg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
import random

def calculate_rounds_off(predicted_round, actual_round):
        rounds = {"R68": 0, "R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "2ND": 6, "Champions": 7 }

        rounds_off = abs(rounds[predicted_round] - rounds[actual_round])
        
        return rounds_off

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
            
        # Add penalty for over/underestimation
        rounds_off = abs(predicted_num - actual_num)
        total_points -= rounds_off
        
        return total_points
        #print(calculate_possible_points("Champions", "Champions"))
def no_validation():
    data = pd.read_csv("model_data.csv")
    features = data.loc[:, "Rank": "WAB"]
    print(features)
    classes = data.loc[:, "POSTSEASON"]
    print(classes)
    train_x, test_x, train_y, test_y = train_test_split(features, classes,  test_size=.15, train_size=.85, shuffle=True)

    #Convert y columns to numpy arrays since they are pandas series - dataframe columns should be numpy arrays
    train_y = train_y.values
    test_y = test_y.values

    #NORMALIZE - only features since the prediction is categorical
    scaler = StandardScaler()
    scaler.fit(train_x)
    trainX_scaled = scaler.transform(train_x)

    scaler.fit(test_x)
    testX_scaled = scaler.transform(test_x)


    sum_of_ks =0

    highest_percentage = 0
    lowest_avg_rounds_off = 1

    best_predictions = []
    corr_test_data = []

    #Experimenting with different Ks
    for i in range(50):

        train_x, test_x, train_y, test_y = train_test_split(features, classes,  test_size=.15, train_size=.85, shuffle=True)

        #Convert y columns to numpy arrays since they are pandas series - dataframe columns should be numpy arrays
        train_y = train_y.values
        test_y = test_y.values

        #NORMALIZE - only features since the prediction is categorical
        scaler = StandardScaler()
        scaler.fit(train_x)
        trainX_scaled = scaler.transform(train_x)

        scaler.fit(test_x)
        testX_scaled = scaler.transform(test_x)



        best_k = 0
        best_k_sum = -100



        for i in range(1, 5):
            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(trainX_scaled, train_y)

            predictions = neigh.predict(testX_scaled)
            correct = 0
            rounds_off = 0 # Possibly better measure of success than correct
            for j in range(len(predictions)):
                if(predictions[j] == test_y[j]):
                    correct += 1
                rounds_off += calculate_rounds_off(predictions[j], test_y[j])

            percentage = correct/len(predictions)
            
            avg_rounds_off = rounds_off/len(predictions)

            if percentage > highest_percentage:
                highest_percentage = percentage

            if avg_rounds_off < lowest_avg_rounds_off:
                lowest_avg_rounds_off = avg_rounds_off
                best_predictions= predictions
                corr_test_data = test_y
            if(percentage - avg_rounds_off > best_k_sum): # want this to be as small as possible
                best_k = i
                best_k_sum = percentage - avg_rounds_off

            print("Neighbors: " + str(i) + ",  Percentage: " + str(percentage) + "  ---  Avg rounds off: " + str(avg_rounds_off))

        sum_of_ks += best_k

    best_k = sum_of_ks/50
    print("AVG BEST K (100 Runs): " + str(best_k))
    print("Best percentage: " + str(highest_percentage))
    print("Lowes Avg Rounds off: " + str(lowest_avg_rounds_off))

    for i in range(len(predictions)):
        if predictions[i] == test_y[i]:
            result = "TRUE"
        else:
            result = "FALSE"
        print("Prediction: " + predictions[i] + "  -  Actual: " + test_y[i] + "  => " + result)

    #print("BEST K: " + str(best_k) + "  -  Sum of: " + str(best_k_sum))
    #print(neigh.predict_proba([[0.9]]))


    ### CONSIDER WHETHER OR NOT THIS MODEL WEIGHTS VARIABLES BASED ON THEIR INFLUENCE
    ##VALIDATION SET???

def kneighbors_validate():
    data = pd.read_csv("model_data.csv")
    features = data.loc[:, "Rank": "WAB"]
    print(features)
    classes = data.loc[:, "POSTSEASON"]
    print(classes)
    train_x, test_x, train_y, test_y = train_test_split(features, classes,  test_size=.15, train_size=.85, shuffle=True)

    #Convert y columns to numpy arrays since they are pandas series - dataframe columns should be numpy arrays
    train_y = train_y.values
    test_y = test_y.values

    # validation set
    trainV_x, validate_x, trainV_y, validate_y = train_test_split(train_x, train_y,  test_size=.15, train_size=.85, shuffle=True)

    #NORMALIZE - only features since the prediction is categorical
    scaler = StandardScaler()
    scaler.fit(trainV_x)
    trainVX_scaled = scaler.transform(trainV_x)

    scaler.fit(validate_x)
    validateX_scaled = scaler.transform(validate_x)


    sum_of_ks =0

    highest_percentage = 0
    lowest_avg_rounds_off = 1

    best_predictions = []
    corr_test_data = []

    #Experimenting with different Ks
    for i in range(100):

        # validation set
        trainV_x, validate_x, trainV_y, validate_y = train_test_split(train_x, train_y,  test_size=.15, train_size=.85, shuffle=True)

        #NORMALIZE - only features since the prediction is categorical
        scaler = StandardScaler()
        scaler.fit(trainV_x)
        trainVX_scaled = scaler.transform(trainV_x)

        scaler.fit(validate_x)
        validateX_scaled = scaler.transform(validate_x)



        best_k = 0
        best_k_sum = -100



        for i in range(1, 50):
            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(trainVX_scaled, trainV_y)

            predictions = neigh.predict(validateX_scaled)
            correct = 0
            rounds_off = 0 # Possibly better measure of success than correct
            for j in range(len(predictions)):
                if(predictions[j] == validate_y[j]):
                    correct += 1
                rounds_off += calculate_rounds_off(predictions[j], validate_y[j])

            percentage = correct/len(predictions)
            
            avg_rounds_off = rounds_off/len(predictions)

            if percentage > highest_percentage:
                highest_percentage = percentage

            if avg_rounds_off < lowest_avg_rounds_off:
                lowest_avg_rounds_off = avg_rounds_off
                best_predictions= predictions
                corr_test_data = test_y
            if(percentage - avg_rounds_off > best_k_sum): # want this to be as big as possible
                best_k = i
                best_k_sum = percentage - avg_rounds_off

            print("Neighbors: " + str(i) + ",  Percentage: " + str(percentage) + "  ---  Avg rounds off: " + str(avg_rounds_off))

        sum_of_ks += best_k

    best_k = sum_of_ks/100
    print("AVG BEST K (100 Runs): " + str(best_k))
    print("Best percentage: " + str(highest_percentage))
    print("Lowes Avg Rounds off: " + str(lowest_avg_rounds_off))

    for i in range(len(predictions)):
        if predictions[i] == test_y[i]:
            result = "TRUE"
        else:
            result = "FALSE"
        print("Prediction: " + predictions[i] + "  -  Actual: " + test_y[i] + "  => " + result)
    
    return int(best_k)

    #print("BEST K: " + str(best_k) + "  -  Sum of: " + str(best_k_sum))
    #print(neigh.predict_proba([[0.9]]))


    ### CONSIDER WHETHER OR NOT THIS MODEL WEIGHTS VARIABLES BASED ON THEIR INFLUENCE
    ##VALIDATION SET???
    # Add more weight to later rounds??


def kneighbors_validate_ESPN_scoring():
    #Fitness to determine best k is now based on total points
    data = pd.read_csv("model_data.csv")
    data['Win %'] = data['Wins']/data['Games']
    #print(list(df.columns.values))
    #features = data.loc[:, "Rank": "WAB"]
    #['TEAM', 'Record', 'YEAR', 'CONF', 'Rank', 'SEED', 'Games', 'Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', 'ADJ T.', 'WAB', 'POSTSEASON']
    features = data[['SEED', 'Win %', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', 'ADJ T.', 'WAB']]
    print(features)
    classes = data.loc[:, "POSTSEASON"]
    print(classes)
    
    #TRAIN AND TEST
    train_x, test_x, train_y, test_y = train_test_split(features, classes,  test_size=.15, train_size=.85, shuffle=True)

    #Convert y columns to numpy arrays since they are pandas series - dataframe columns should be numpy arrays
    train_y = train_y.values
    test_y = test_y.values
    
    
    

    # validation set
    trainV_x, validate_x, trainV_y, validate_y = train_test_split(train_x, train_y,  test_size=.15, train_size=.85, shuffle=True)

    #NORMALIZE - only features since the prediction is categorical
    scaler = StandardScaler()
    scaler.fit(trainV_x)
    trainVX_scaled = scaler.transform(trainV_x)

    scaler.fit(validate_x)
    validateX_scaled = scaler.transform(validate_x)


    sum_of_ks =0

    highest_percentage = 0
    lowest_avg_rounds_off = 1

    best_predictions = []
    corr_test_data = []

    #Experimenting with different Ks
    for i in range(15):

        # validation set
        trainV_x, validate_x, trainV_y, validate_y = train_test_split(train_x, train_y,  test_size=.2, train_size=.8, shuffle=True)

        #NORMALIZE - only features since the prediction is categorical
        scaler = StandardScaler()
        scaler.fit(trainV_x)
        trainVX_scaled = scaler.transform(trainV_x)

        scaler.fit(validate_x)
        validateX_scaled = scaler.transform(validate_x)



        best_k = 0
        most_points = -100


        best_predictions = []
        for i in range(1, 10):
            neigh = KNeighborsClassifier(n_neighbors=i)
            neigh.fit(trainVX_scaled, trainV_y)

            predictions = neigh.predict(validateX_scaled)
            
            total_points = 0
            for j in range(len(predictions)):
                total_points += calculate_possible_points(predictions[j], validate_y[j])
            
            
            if(total_points > most_points): # want this to be as big as possible
                best_k = i
                most_points  = total_points
                #best_predictions = predictions

            print("Neighbors: " + str(i) + ",  Total Points: " + str(total_points))
            #print(total_points)
        sum_of_ks += best_k


    #best_k = sum_of_ks/100
    #print("AVG BEST K (100 Runs): " + str(best_k))
    #print("Best percentage: " + str(highest_percentage))
    #print("Lowes Avg Rounds off: " + str(lowest_avg_rounds_off))
    
    
    
    # USING THE VALIDATED K VALUE ON TEST SET
    # Normalize data
    #NORMALIZE - only features since the prediction is categorical
    scaler = StandardScaler()
    scaler.fit(train_x)
    trainX_scaled = scaler.transform(train_x)

    scaler.fit(test_x)
    testX_scaled = scaler.transform(test_x)
    
    neigh = KNeighborsClassifier(n_neighbors=int(best_k))
    neigh.fit(trainX_scaled, train_y)

    predictions = neigh.predict(testX_scaled)
    correct = 0
    rounds_off = 0 # Possibly better measure of success than correct
    for j in range(len(predictions)):
        if(predictions[j] == test_y[j]):
            correct += 1
        rounds_off += calculate_rounds_off(predictions[j], test_y[j])

    percentage = correct/len(predictions)

    avg_rounds_off = rounds_off/len(predictions)
    print('-----------------------------')
    for i in range(len(predictions)):
        if predictions[i] == test_y[i]:
            result = "TRUE"
        else:
            result = "FALSE"
        print("Prediction: " + predictions[i] + "  -  Actual: " + test_y[i] + "  => " + result)
        
        
    print(percentage)
    print(avg_rounds_off)
    
            
            

    return int(best_k) # SEEMS LIKE k=3 is best right now

#print(kneighbors_validate_ESPN_scoring())


# Develop algo to predict a whole bracket's worth of games
# Use all historical data to find best k and use all this data as train
# Get this year's data and this will be what we are attempting to predict


def predict_xYears_bracket_bestK(year):
   
    data = pd.read_csv("model_data.csv")
    train_data = data[data['YEAR'] != year]
    bracket_data = data[data['YEAR'] == year]
    
    #print(train_data)
    #print(bracket_data)
    
    train_features = train_data.loc[:, "Rank": "WAB"]
    bracket_features = bracket_data.loc[:, "Rank": "WAB"]
    train_classes = train_data.loc[:, "POSTSEASON"]
    bracket_classes = bracket_data.loc[:, "POSTSEASON"].values  
    
    print(train_features)
    print(bracket_features)
    print(train_classes)
    print(bracket_classes)
    
    #NORMALIZE - only features since the prediction is categorical
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)

    scaler.fit(bracket_features)
    bracket_features = scaler.transform(bracket_features)


    best_predictions = []
    corr_test_data = []


    best_k = 0
    most_points = -100


    best_predictions = []
    for i in range(1, 20):
        neigh = KNeighborsClassifier(n_neighbors=i)
        neigh.fit(train_features, train_classes)

        predictions = neigh.predict(bracket_features)
    
        total_points = 0
        for j in range(len(predictions)):
            total_points += calculate_possible_points(predictions[j], bracket_classes[j])


        if(total_points > most_points): # want this to be as big as possible
            best_k = i
            most_points  = total_points
            #best_predictions = predictions

        #print("Neighbors: " + str(i) + ",  Total Points: " + str(total_points))
        #print(total_points)
    


    return best_k

    
    
#print(kneighbors_validate_ESPN_scoring())
#k=0
#for year in range(2008, 2020):
    #k += predict_xYears_bracket_bestK(year)
#k+= predict_xYears_bracket(2021)
#print(k/13)
# Develop algo to predict a whole bracket's worth of games
# Use all historical data to find best k and use all this data as train
# Get this year's data and this will be what we are attempting to predict

def predict_xYears_bracket(year, k, features):
   
    data = pd.read_csv("model_data.csv")
    data['Win %'] = data['Wins']/data['Games']

    train_data = data[data['YEAR'] != year]
    bracket_data = data[data['YEAR'] == year]
    
    #print(train_data)
    #print(bracket_data)
    
    #train_features = train_data[['Rank', 'SEED', 'Win %', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', 'ADJ T.', 'WAB']]   #train_data.loc[:, "Rank": "WAB"]
    #train_features = train_data.loc[:, "Rank":"WAB"]
    train_features = train_data[features]
    #bracket_features = bracket_data[['Rank','SEED', 'Win %', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', 'ADJ T.', 'WAB']]    #bracket_data.loc[:, "Rank": "WAB"]
    #print(bracket_features)
    #bracket_features = bracket_data.loc[:, "Rank": "WAB"]
    bracket_features = bracket_data[features]
    train_classes = train_data.loc[:, "POSTSEASON"]
    bracket_classes = bracket_data.loc[:, "POSTSEASON"].values  
    
    
    #NORMALIZE - only features since the prediction is categorical
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)

    scaler.fit(bracket_features)
    bracket_features = scaler.transform(bracket_features)



    best_predictions = []
   
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_features, train_classes)

    predictions = neigh.predict(bracket_features)

    total_points = 0
    for j in range(len(predictions)):
        total_points += calculate_possible_points(predictions[j], bracket_classes[j])

    for i in range(len(predictions)):
        if predictions[i] == bracket_classes[i]:
            result = "TRUE"
        else:
            result = "FALSE"
        #print("Team: " + bracket_data["TEAM"].values[i] + "  Prediction: " + predictions[i] + "  Actual: " + bracket_classes[i] + "  => " + result)
    
    print("Points: " + str(total_points))
    
    
    return total_points

"""
ksum=0.0
count=0.0
for j in range(2008, 2020):
    best_k = 0
    most_points = 0
    for k in range(1,10):
        print("Year " + str(j) + "  k: " + str(k))
        points = predict_xYears_bracket(j, k)
        if points > most_points:
            best_k = k
            most_points = points
    ksum+= best_k
    count+=1
    
print(ksum/count) # 3.1666666666666665 THINK K AROUND 3 IS IDEAL
"""

def find_subset_features():
    
    set = ['Rank', 'SEED', 'Win %','Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '2P%D', '3P%', '3P%D', 'ADJ T.', 'WAB']
    
    options = []
    for i in range(1,len(set)):
        data = itertools.combinations(set, i)
        subsets = list(data)
        options += subsets
    print(len(options))
        
    options = np.random.choice(options, 50000)
    return options



def find_best_features_set():
    pointsSum=0.0
    count=0.0
    possible_features = find_subset_features()
    """
    print(predict_xYears_bracket(2008, 3,['SEED', 'ADJDE', 'FTR', '2P%D']))
    print(predict_xYears_bracket(2009, 3,['SEED', 'ADJDE', 'FTR', '2P%D']))
    print(predict_xYears_bracket(2010, 3,['SEED', 'ADJDE', 'FTR', '2P%D']))
    print(predict_xYears_bracket(2011, 3,['SEED', 'ADJDE', 'FTR', '2P%D']))
    print(predict_xYears_bracket(2012, 3,['SEED', 'ADJDE', 'FTR', '2P%D']))
    print(predict_xYears_bracket(2013, 3,['SEED', 'ADJDE', 'FTR', '2P%D']))
    print(predict_xYears_bracket(2014, 3,['SEED', 'ADJDE', 'FTR', '2P%D']))
    print(predict_xYears_bracket(2015, 3,['SEED', 'ADJDE', 'FTR', '2P%D']))
    print(predict_xYears_bracket(2016, 3,['SEED', 'ADJDE', 'FTR', '2P%D']))
    print(predict_xYears_bracket(2017, 3,['SEED', 'ADJDE', 'FTR', '2P%D']))
    print(predict_xYears_bracket(2018, 3,['SEED', 'ADJDE', 'FTR', '2P%D']))
    print(predict_xYears_bracket(2019, 3,['SEED', 'ADJDE', 'FTR', '2P%D']))
    """
    points_arr = ["", "","", "", "", "","", "","", "","", ""]
    features_arr = ["", "","", "", "", "","", "","", "","", ""]
    #print(possible_features)
    for j in range(2008, 2020):
        print(j)
        most_points = 0
        for features_tuple in possible_features:
            features_list = list(features_tuple)
            points = predict_xYears_bracket(j, 3, features_list)
            if points > most_points:
                points_arr[j-2008] = points
                most_points = points
                features_arr[j-2008] = features_list

    #RESULT 
    """
    [127, 131, 108, 91, 111, 98, 85, 136, 114, 126, 113, 132]
    [['SEED', 'Win %', 'Wins', 'ADJOE', 'BARTHAG', 'EFGD%', 'TOR', 'TORD', 'FTR', '3P%D', 'ADJ T.'], ['Rank', 'Win %', 'Losses', 'BARTHAG', 'ORB', 'DRB', '2P%D'], ['Rank', 'SEED', 'Wins', 'Losses', 'EFG%', 'EFGD%', 'ORB', 'DRB', 'FTRD', '2P%D', 'WAB'], ['SEED', 'Win %', 'ORB', '2P%', '2P%D', '3P%', 'ADJ T.'], ['SEED', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'TOR', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '3P%D', 'ADJ T.', 'WAB'], ['Rank', 'Win %', 'Losses', 'ADJOE', 'ADJDE', 'EFGD%', 'TOR', 'FTR', '2P%D', 'ADJ T.', 'WAB'], ['Rank', 'Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'TOR', 'TORD', 'DRB', '2P%', '2P%D', '3P%D', 'WAB'], ['Losses', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'ORB', 'DRB', '2P%', '3P%', 'ADJ T.'], ['SEED', 'Win %', 'Losses', 'EFG%', 'ORB', 'FTR', 'WAB'], ['SEED', 'Win %', 'ADJOE', 'EFGD%', 'TOR', 'TORD', 'ORB', '2P%', '2P%D', '3P%D', 'ADJ T.', 'WAB'], ['Wins', 'Losses', 'ADJDE', 'BARTHAG', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%D', '3P%', '3P%D', 'WAB'], ['ADJOE', 'BARTHAG', 'EFGD%', 'TORD', '2P%D', '3P%D']]
    """



#pointsSum += predict_xYears_bracket(2021, 3, list(features)
#count+=1
#print("Avg POINTS: " + str(pointsSum/count))

#predict_xYears_bracket(2021, 3)
    
#kneighbors_validate_ESPN_scoring()
features0 = ['Losses', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'ORB', 'DRB', '2P%', '3P%', 'ADJ T.']
features1 =['SEED', 'Win %', 'Wins', 'ADJOE', 'BARTHAG', 'EFGD%', 'TOR', 'TORD', 'FTR', '3P%D', 'ADJ T.', 'Rank', 'Losses', 'ORB', 'DRB', '2P%D', 'EFG%', 'FTRD', 'WAB', '2P%', '3P%', 'ADJDE']
features2 = ['SEED', 'Win %', 'ADJOE', 'BARTHAG', 'EFGD%', 'TOR', '3P%D', 'ADJ T.', 'Losses', 'ORB', 'DRB', '2P%D', 'WAB','Rank'] 
features3 = ['Losses', 'ORB', '2P%D']
score0 = 0
score1 = 0
score2 = 0
score3 = 0
for year in range(2008, 2020):
    score0 += predict_xYears_bracket(year, 3,features0)
    score1 += predict_xYears_bracket(year, 3,features1)
    score2 += predict_xYears_bracket(year, 3,features2)
    score3 += predict_xYears_bracket(year, 3,features3)

score0 += predict_xYears_bracket(2021, 3,features1)
score1 += predict_xYears_bracket(2021, 3,features1)
score2 += predict_xYears_bracket(2021, 3,features2)
score3 += predict_xYears_bracket(2021, 3,features3)

print(score0/13)
print(score1/13)
print(score2/13)
print(score3/13)