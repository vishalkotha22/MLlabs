from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=[15, 35, 100], n_features=4, random_state=0)
X = X.tolist()
y = y.tolist()
X.append([100, 100, 100, 100])
y.append(2)
X.append([500, 500, 500, 500])
y.append(2)
X.append([2500, 2500, 2500, 250])
y.append(1)
X.append([5000, 500, 5000, 50000])
y.append(0)
X.append([10000, 1000, 100, 10])
y.append(1)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.33, random_state=10)

first = [data[0] for data in X]
sec = [data[1] for data in X]
thir = [data[2] for data in X]
four = [data[3] for data in X]

print(min(first), max(first))
print(min(sec), max(sec))
print(min(thir), max(thir))
print(min(four), max(four))

K = 8

knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(xTrain, yTrain)
knnPred = knn.predict(xTest)
print(f'Accuracy: {accuracy_score(knnPred, yTest)*100}%')