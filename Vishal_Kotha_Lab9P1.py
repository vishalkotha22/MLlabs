import pandas as pd

from sklearn.model_selection import train_test_split
from statistics import mode
import math

import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

y = []
for val in df['class']:
    if 'setosa' in val:
        y.append(0)
    elif 'versicolor' in val:
        y.append(1)
    else:
        y.append(2)
df['class'] = y

X, y = df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']], df['class'].tolist()
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=.33, random_state=10)
xTrain['class'] = yTrain

K = 5

yPred = []

for row in xTest.iterrows():
    distances = []
    for row2 in xTrain.iterrows():
        dist = 0
        for idx in range(len(xTrain.columns)-1):
            dist += (row2[1][idx] - row[1][idx]) * (row2[1][idx] - row[1][idx])
        dist = math.sqrt(dist)
        distances.append((dist, row2[1][len(xTrain.columns)-1]))
    distances.sort()
    yPred.append(mode([distances[i][1] for i in range(K)]))

print(yPred)
print(f'Accuracy: {sum([1 for i in range(len(yPred)) if yPred[i] == yTest[i]])*100/len(yPred)}% and k: {K}')