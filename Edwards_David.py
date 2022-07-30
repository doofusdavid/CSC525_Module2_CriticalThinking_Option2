"""
David Edwards
CSC525 - Principles of Machine Learning
Module 2 - Critical Thinking - Option 2
Dr. Issac Gang
7/30/2022
"""
import pandas as pd
import numpy as np
from scipy.spatial import distance
from statistics import mode

# Read in Training data
games = pd.read_csv("data.csv")

# Break training data into the features and the category
X = games.iloc[:, :-1].values
y = games.iloc[:, 4].values

# read in our data, ensuring the user enters integers (and 0,1 for gender)
print("This will perform a k_nn classifier using the games data.  We are using a k of 5.")

# Assign our value of k
k = 5

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


def k_nn_game_classifier(k, age, height, weight, gender, X, y):
    results = []
    # make our test data look the same as our training data
    test = [age, height, weight, gender]

    # iterate through every training data point
    for idx, train in enumerate(X):
        # Calculate the Euclidean distance between our input data and our training data
        dist = distance.euclidean(test, train)
        # Append that distance to a new list
        results.append([dist, y[idx]])

    # Sort this list by the distance
    results.sort(key=lambda x: x[0])

    # Take the top k items
    top_k = np.array(results[:k])

    # print the most common (mode) game genre as a result
    return mode(top_k[:, 1])


print("This player would be classified as a", k_nn_game_classifier(k, age, height, weight, gender, X, y), "gamer.")
