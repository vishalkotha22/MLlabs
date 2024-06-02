import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

df = pd.read_csv('iris.csv')
X = df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']]
y = []
for val in df['class']:
    if val == 'Iris-setosa':
        y.append(0)
    elif val == 'Iris-virginica':
        y.append(1)
    else:
        y.append(2)
pca = PCA(n_components=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=27)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)
xVals = np.linspace(min(X_train[:,0]), max(X_train[:,0]))
for i in range(1, len(clf.coef_)):
    slope = clf.coef_[i]
    intercept = clf.intercept_[i]
    yVals = -(slope[0] / slope[1]) * xVals - (intercept / slope[1])
    plt.plot(xVals, yVals, c='g')
plt.scatter(x=X_train[:,0], y=X_train[:,1], c=y_train)
plt.show()
xVals = np.linspace(min(X_test[:,0]), max(X_test[:,0]))
for i in range(1, len(clf.coef_)):
    slope = clf.coef_[i]
    intercept = clf.intercept_[i]
    yVals = -(slope[0] / slope[1]) * xVals - (intercept / slope[1])
    plt.plot(xVals, yVals, c='g')
plt.scatter(x=X_test[:,0], y=X_test[:,1], c=y_test)
plt.show()
predictions = clf.predict(X_test)
accuracy = sum([1 for i in range(len(predictions)) if predictions[i] == y_test[i]]) / len(predictions) * 100
print('Accuracy: ' + str(accuracy) + '%')