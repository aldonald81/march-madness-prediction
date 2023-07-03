set = [['SEED', 'Win %', 'Wins', 'ADJOE', 'BARTHAG', 'EFGD%', 'TOR', 'TORD', 'FTR', '3P%D', 'ADJ T.'], ['Rank', 'Win %', 'Losses', 'BARTHAG', 'ORB', 'DRB', '2P%D'], ['Rank', 'SEED', 'Wins', 'Losses', 'EFG%', 'EFGD%', 'ORB', 'DRB', 'FTRD', '2P%D', 'WAB'], ['SEED', 'Win %', 'ORB', '2P%', '2P%D', '3P%', 'ADJ T.'], ['SEED', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'TOR', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%', '3P%D', 'ADJ T.', 'WAB'], ['Rank', 'Win %', 'Losses', 'ADJOE', 'ADJDE', 'EFGD%', 'TOR', 'FTR', '2P%D', 'ADJ T.', 'WAB'], ['Rank', 'Wins', 'Losses', 'ADJOE', 'ADJDE', 'BARTHAG', 'TOR', 'TORD', 'DRB', '2P%', '2P%D', '3P%D', 'WAB'], ['Losses', 'ADJDE', 'BARTHAG', 'EFG%', 'EFGD%', 'TOR', 'ORB', 'DRB', '2P%', '3P%', 'ADJ T.'], ['SEED', 'Win %', 'Losses', 'EFG%', 'ORB', 'FTR', 'WAB'], ['SEED', 'Win %', 'ADJOE', 'EFGD%', 'TOR', 'TORD', 'ORB', '2P%', '2P%D', '3P%D', 'ADJ T.', 'WAB'], ['Wins', 'Losses', 'ADJDE', 'BARTHAG', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P%D', '3P%', '3P%D', 'WAB'], ['ADJOE', 'BARTHAG', 'EFGD%', 'TORD', '2P%D', '3P%D']]

dict = {}
for element in set:
    for feature in element:
        if feature in dict:
            dict[feature]+=1
        else:
            dict[feature] = 1
print(dict)
# Find features that were in 8 out of 12 of years best model
list=[]
list2=[]
list3=[]
for key in dict:
    if dict[key] >= 2:
        list.append(key)
    if dict[key] >= 6:
        list2.append(key)
    if dict[key] >= 8:
        list3.append(key)
print(list)
print(list2)
print(list3)