import pandas as pd

from sklearn.model_selection import train_test_split
from statistics import mode
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=[15, 35, 100], n_features=4, random_state=0)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.33, random_state=10)

yTrain = yTrain.tolist()
yTest = yTest.tolist()

print(yTrain.count(0), yTrain.count(1), yTrain.count(2))
print(yTest.count(0), yTest.count(1), yTest.count(2))

K = 8

knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(xTrain, yTrain)
knnPred = knn.predict(xTest)
print(f'Accuracy: {accuracy_score(knnPred, yTest)*100}%')