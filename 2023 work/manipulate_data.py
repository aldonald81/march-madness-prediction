from posixpath import split
import numpy as np
import pandas as pd
import math


#NEED postseason and seed
data = pd.read_csv("2023 raw data.csv")
print(data)
data = data[data['TEAM'] != "TEAM"]


print(data)

cols = {'Team': [],
	'Seed': [],
	'Outcome': []}

#create dataframe
df = pd.DataFrame(cols)

#new_row = {'name':'Geo', 'physics':87, 'chemistry':92, 'algebra':97}
#append row to the dataframe
#df_marks = df_marks.append(new_row, ignore_index=True)

vals = data['TEAM'].values
years = data['YEAR'].values
for i in range(0, len(vals), 2):
    teamName = vals[i]
    info = vals[i+1]
    #print("A: " + vals[i])
    #print("B: " + str(vals[i+1]))
    #CHECK IF TYPE FLOAT
    
    if (isinstance(info, float)) :
        continue

    elif("seed" in info):
        #string = info.replace('�', '') 
        seed, result = info.split(",")
    else:
        seed = "NA"
        result = "NA"
    # print(teamName)
    # print(seed)
    # print(result)
    # print("------------")

    newRow = {"Team": teamName, "Seed": seed, "Outcome": "NA", "Year": int(years[i])}
    df = df.append(newRow, ignore_index=True)

print(df)

# Merge this data frame with the rest of the stats
merged_df = pd.merge(df, data, left_on='Team', right_on='TEAM', how='left')
print(merged_df)
merged_df.to_csv("full_2023_cleaned.csv")
df.to_csv("data_addition_2023.csv")