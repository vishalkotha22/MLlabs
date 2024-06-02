import pandas as pd

from sklearn.model_selection import train_test_split
from statistics import mode
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def generateConfusionMatrix(yPred, yTest):
    confusionMatrix = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    for i in range(len(yPred)):
        if yPred[i] == 0 and yTest[i] == 0:
            confusionMatrix['1'] += 1
        if yPred[i] == 1 and yTest[i] == 0:
            confusionMatrix['4'] += 1
        if yPred[i] == 2 and yTest[i] == 0:
            confusionMatrix['7'] += 1
        if yPred[i] == 0 and yTest[i] == 1:
            confusionMatrix['2'] += 1
        if yPred[i] == 1 and yTest[i] == 1:
            confusionMatrix['5'] += 1
        if yPred[i] == 2 and yTest[i] == 1:
            confusionMatrix['8'] += 1
        if yPred[i] == 0 and yTest[i] == 2:
            confusionMatrix['3'] += 1
        if yPred[i] == 1 and yTest[i] == 2:
            confusionMatrix['6'] += 1
        if yPred[i] == 2 and yTest[i] == 2:
            confusionMatrix['9'] += 1
    return confusionMatrix

def macroAverageRecall(cfMatrix):
    r1 = cfMatrix['1'] / (cfMatrix['1'] + cfMatrix['4'] + cfMatrix['7'])
    r2 = cfMatrix['5'] / (cfMatrix['2'] + cfMatrix['5'] + cfMatrix['8'])
    r3 = cfMatrix['9'] / (cfMatrix['3'] + cfMatrix['6'] + cfMatrix['9'])
    return (r1 + r2 + r3) / 3

def macroAveragePrecision(cfMatrix):
    p1 = cfMatrix['1'] / (cfMatrix['1'] + cfMatrix['2'] + cfMatrix['3'])
    p2 = cfMatrix['5'] / (cfMatrix['4'] + cfMatrix['5'] + cfMatrix['6'])
    p3 = cfMatrix['9'] / (cfMatrix['7'] + cfMatrix['8'] + cfMatrix['9'])
    return (p1 + p2 + p3) / 3

def microAverage(cfMatrix, pooling, toggler):
    cfMatrix1 = {'1' : 0, '2' : 0, '3' : 0, '4' : 0}
    cfMatrix1['1'] = cfMatrix['1']
    cfMatrix1['2'] = cfMatrix['2'] + cfMatrix['3']
    cfMatrix1['3'] = cfMatrix['4'] + cfMatrix['7']
    cfMatrix1['4'] = cfMatrix['5'] + cfMatrix['6'] + cfMatrix['8'] + cfMatrix['9']
    cfMatrix2 = {'1' : 0, '2' : 0, '3' : 0, '4' : 0}
    cfMatrix2['1'] = cfMatrix['5']
    cfMatrix2['2'] = cfMatrix['4'] + cfMatrix['6']
    cfMatrix2['3'] = cfMatrix['2'] + cfMatrix['8']
    cfMatrix2['4'] = cfMatrix['1'] + cfMatrix['3'] + cfMatrix['7'] + cfMatrix['9']
    cfMatrix3 = {'1' : 0, '2' : 0, '3' : 0, '4' : 0}
    cfMatrix3['1'] = cfMatrix['9']
    cfMatrix3['2'] = cfMatrix['7'] + cfMatrix['8']
    cfMatrix3['3'] = cfMatrix['3'] + cfMatrix['6']
    cfMatrix3['4'] = cfMatrix['1'] + cfMatrix['2'] + cfMatrix['4'] + cfMatrix['5']
    if pooling == True:
        cfMatrix4 = {'1' : 0, '2' : 0, '3' : 0, '4' : 0}
        cfMatrix4['1'] = cfMatrix1['1'] + cfMatrix2['1'] + cfMatrix3['1']
        cfMatrix4['2'] = cfMatrix1['2'] + cfMatrix2['2'] + cfMatrix3['2']
        cfMatrix4['3'] = cfMatrix1['3'] + cfMatrix2['3'] + cfMatrix3['3']
        cfMatrix4['4'] = cfMatrix1['4'] + cfMatrix2['4'] + cfMatrix3['4']
        if toggler == True:
            return cfMatrix4['1'] / (cfMatrix4['1'] + cfMatrix4['2'])
        else:
            return cfMatrix4['1'] / (cfMatrix4['1'] + cfMatrix4['3'])
    else:
        if toggler == True:
            p1 = cfMatrix1['1'] / (cfMatrix1['1'] + cfMatrix1['2'])
            p2 = cfMatrix2['1'] / (cfMatrix2['1'] + cfMatrix2['2'])
            p3 = cfMatrix3['1'] / (cfMatrix3['1'] + cfMatrix3['2'])
            return (p1 + p2 + p3) / 3
        else:
            r1 = cfMatrix1['1'] / (cfMatrix1['1'] + cfMatrix1['3'])
            r2 = cfMatrix2['1'] / (cfMatrix2['1'] + cfMatrix2['3'])
            r3 = cfMatrix3['1'] / (cfMatrix3['1'] + cfMatrix3['3'])
            return (r1 + r2 + r3) / 3

K = 8

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

knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(xTrain, yTrain)
knnPred = knn.predict(xTest)

xTrain['class'] = yTrain

yPred = []

#print(yTest.count(0), yTest.count(1), yTest.count(2))

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

knnCF = generateConfusionMatrix(knnPred, yTest)
table = [
        [knnCF['1'], knnCF['2'], knnCF['3']],
        [knnCF['4'], knnCF['5'], knnCF['6']],
        [knnCF['7'], knnCF['8'], knnCF['9']]
    ]
cfMatrix = pd.DataFrame(table, columns=['1', '2', '3'], index=['1', '2', '3'])
print(cfMatrix)
print(f'SKLearn Macro Average Precision: {macroAveragePrecision(knnCF)}')
print(f'SKLearn Macro Average Recall: {macroAverageRecall(knnCF)}')
print(f'KNN from Scratch Micro Average Pooling Precision: {microAverage(knnCF, True, True)}')
print(f'KNN from Scratch Micro Average Pooling Recall: {microAverage(knnCF, True, False)}')
print(f'KNN from Scratch Micro Average Pooling Precision: {microAverage(knnCF, False, True)}')
print(f'KNN from Scratch Micro Average Pooling Recall: {microAverage(knnCF, False, False)}')
