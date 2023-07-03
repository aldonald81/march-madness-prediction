from posixpath import split
import numpy as np
import pandas as pd
import math


#NEED postseason and seed
#SOURCE: https://www.barttorvik.com/trank.php#
data = pd.read_csv("2008_2012.csv")
print(data)
data = data[data['1'] != "TEAM"]


print(data)

cols = {'Team': [],
	'Seed': [],
	'Outcome': []}

#create dataframe
df = pd.DataFrame(cols)

#new_row = {'name':'Geo', 'physics':87, 'chemistry':92, 'algebra':97}
#append row to the dataframe
#df_marks = df_marks.append(new_row, ignore_index=True)

vals = data['1'].values
years = data['22'].values
for i in range(0, len(vals), 2):
    teamName = vals[i]
    info = vals[i+1]
    print("A: " + vals[i])
    print("B: " + str(vals[i+1]))
    #CHECK IF TYPE FLOAT
    
    if (isinstance(info, float)) :
        continue

    elif("seed" in info):
        #string = info.replace('�', '') 
        seed, result = info.split(",")
    else:
        seed = "NA"
        result = "NA"
    print(teamName)
    print(seed)
    print(result)
    print("------------")

    newRow = {"Team": teamName, "Seed": seed, "Outcome": result, "Year": int(years[i])}
    df = df.append(newRow, ignore_index=True)

print(df)

df.to_csv("cbb_addition.csv")