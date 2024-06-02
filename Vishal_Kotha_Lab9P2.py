import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

K = 20

knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(xTrain, yTrain)
yPred = knn.predict(xTest)
print(f'Accuracy: {accuracy_score(yPred, yTest)*100}%')