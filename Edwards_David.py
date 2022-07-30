"""
David Edwards
CSC525 - Principles of Machine Learning
Dr. Issac Gang
7/30/2022
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance
from statistics import mode

# Read in Training data
games = pd.read_csv("data.csv")
X = games.iloc[:, :-1].values
y = games.iloc[:, 4].values

while True:
    try:
        age = int(input("Enter the age: "))
    except ValueError:
        print("Invalid Age")
        continue
    else:
        break
while True:
    try:
        height = int(input("Enter the height: "))
    except ValueError:
        print("Invalid Height")
        continue
    else:
        break
while True:
    try:
        weight = int(input("Enter the weight: "))
    except ValueError:
        print("Invalid weight")
        continue
    else:
        break
while True:
    try:
        gender = int(input("Enter the gender (female=0, male=1: "))
    except ValueError:
        print("Invalid Gender")
        continue
    else:
        if gender not in [0, 1]:
            print("Invalid Gender")
            continue
        else:
            break

k = 5

results = []
test = [age, height, weight, gender]
for idx, train in enumerate(X):
    dist = distance.euclidean(test, train)
    results.append([dist, y[idx]])
results.sort(key=lambda x: x[0])
top_k = np.array(results[:k])
print(mode(top_k[:, 1]))
