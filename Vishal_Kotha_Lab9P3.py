import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import math
from statistics import mode

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

x = []
yAcc = []
kBest = []

xTrain['class'] = yTrain
for K in range(1, len(xTrain)+1):
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
    x.append(K)
    yAcc.append(accuracy_score(yPred, yTest))
    if yAcc[-1] == .98:
        kBest.append(K)

print(kBest)

plt.plot(x, yAcc)
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.show()