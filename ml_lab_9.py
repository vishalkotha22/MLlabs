import numpy as np
import pandas as pd
import math
import random

from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

for col in df.columns:
    if col != 'class':
        using = []
        vals = df[col].tolist()
        for i in range(len(vals)):
            using.append((vals[i], i))
        using = sorted(using)
        bin1, bin2, bin3 = using[:50], using[50:100], using[100:]
        minB1, maxB1 = min(bin1)[0], max(bin1)[0]
        minB2, maxB2 = min(bin2)[0], max(bin2)[0]
        minB3, maxB3 = min(bin3)[0], max(bin3)[0]
        for i, tup in enumerate(bin1):
            if tup[0] - minB1 <= maxB1 - tup[0]:
                df.iloc[tup[1], df.columns.get_loc(col)] = minB1
            else:
                df.iloc[tup[1], df.columns.get_loc(col)] = maxB1
        for i, tup in enumerate(bin2):
            if tup[0] - minB2 <= maxB2 - tup[0]:
                df.iloc[tup[1], df.columns.get_loc(col)] = minB2
            else:
                df.iloc[tup[1], df.columns.get_loc(col)] = maxB2
        for i, tup in enumerate(bin3):
            if tup[0] - minB3 <= maxB3 - tup[0]:
                df.iloc[tup[1], df.columns.get_loc(col)] = minB3
            else:
                df.iloc[tup[1], df.columns.get_loc(col)] = maxB3

class Node:
    def __init__(self, data, attr, output, children):
        self.data = data
        self.attr = attr
        self.output = output
        self.children = children

    def addChild(self, obj):
        self.children.append(obj)

def calcEntropy(data):
    p1 = len(data[data['class'] == 0]) / len(data)
    p2 = len(data[data['class'] == 1]) / len(data)
    p3 = len(data[data['class'] == 2]) / len(data)
    #print(p1, p2, p3)
    sum = 0
    if p1 > 0:
        sum += p1 * math.log2(p1)
    if p2 > 0:
        sum += p2 * math.log2(p2)
    if p3 > 0:
        sum += p3 * math.log2(p3)
    return -1 * sum

def recur(data, attrsList, parent):
    if len(data) == 0:
        node = Node(data, '', 'WTF', [])
        parent.addChild(node)
        parent.attr = "LEAF"
        return
    elif len(attrsList) == 0:
        node = Node(data, '', data.mode()['class'].tolist()[0], [])
        parent.addChild(node)
        parent.attr = "LEAF"
        return
    elif len(data['class'].unique().tolist()) == 1:
        node = Node(data, '', data['class'].unique().tolist()[0], [])
        parent.addChild(node)
        parent.attr = "LEAF"
        return
    else:
        maxInfoGain, maxAttr = -1, ''
        if len(attrsList) >= 2:
            #print(data.head(5))
            initEntropy = calcEntropy(data)
            for attr in attrsList:
                infoGain = initEntropy
                denom = 0
                for val in data[attr].unique():
                    subData = data[data[attr] == val]
                    denom -= (len(subData) / len(data)) * math.log2(len(subData) / len(data))
                    entropy = calcEntropy(subData)
                    if entropy > 0:
                        infoGain -= (len(subData) / len(data)) * entropy
                infoGain = infoGain / denom
                if infoGain > maxInfoGain:
                    maxInfoGain = infoGain
                    maxAttr = attr
        else:
            maxAttr = attrsList[0]
            del attrsList[0]
        parent.attr = maxAttr
        for val in data[maxAttr].unique():
            dataSubset = data[data[maxAttr] == val]
            node = Node(dataSubset, '', val, [])
            parent.addChild(node)
            recur(dataSubset, [attrib for attrib in attrsList if attrib != maxAttr], node)
        return

def dfs(node, parent, depth):
    if len(node.children) == 0:
        print('\t' * depth + f'Leaf node with a value of {node.output}')
    else:
        if depth == 0:
            print('\t' * depth + f'Using the complete dataset, we split on {parent.attr}')
        else:
            if node.attr == 'LEAF':
                print('\t' * depth + f'This branch is for a value of {node.output} for {parent.attr}. The next node is a leaf node.')
            else:
                print('\t' * depth + f'This branch is for a value of {node.output} for {parent.attr}. The next attribute we split on is {node.attr}')
        for child in node.children:
            dfs(child, node, depth+1)

numOfAttributes = int(math.sqrt(len(df.columns) - 1))
#print(numOfAttributes)

N = 25 #how many trees will be in the random forest
k = 50 #how many instances will be in the training datasets

trees = []

X, y = df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']], df['class'].tolist()
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=.2, random_state=3)
xTrainUse = xTrain.copy()
xTrain['class'] = yTrain
xTrain.to_csv('rf_train.csv')
xTestCopy = xTest.copy()
xTestCopy['class'] = yTest
xTestCopy.to_csv('rf_test.csv')

for i in range(N):
    data = xTrain.iloc[np.random.randint(0, len(xTrain), size=k)]
    root = Node(data, '', None, [])
    features = random.sample(['sepalwidth', 'sepallength', 'petalwidth', 'petallength'], 2)
    recur(data, features, root)
    trees.append(root)
#print(trees)

yPred = []

for idx, row in xTest.iterrows():
    yAggregate = []
    for r in trees:
        start = r
        while len(start.children) > 0:
            #print(start.output, start.attr, start.children)
            #print(start.attr, start.output, start.children)
            if start.attr != "LEAF":
                runner = False
                for child in start.children:
                    if child.output == row[start.attr]:
                        start = child
                        runner = True
                        break
                if runner == False:
                    #print('here')
                    start = start.children[int(random.random() * len(start.children))]
            else:
                start = start.children[0]
                yAggregate.append(start.output)
    #print(yAggregate)
    yPred.append(mode(yAggregate))
    if idx <= 30:
        print(f'{row} was classified as {yPred[-1]}\n')

print()
for j in range(5):
    dfs(trees[j], trees[j], 0)
    print()

print('My predictions:', yPred)

clf = RandomForestClassifier(n_estimators=25, max_features=2)
clf.fit(xTrainUse, yTrain)
rfPred = clf.predict(xTest)
print('Random Forest predictions:', rfPred)

def createConfusionMatrix(yPred, yTest):
    confusionMatrix = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
    for i in range(len(yPred)):
        if yPred[i] == 0 and yTest[i] == 0:
            confusionMatrix['1'] += 1
        if yPred[i] == 1 and yTest[i] == 0:
            confusionMatrix['2'] += 1
        if yPred[i] == 2 and yTest[i] == 0:
            confusionMatrix['3'] += 1
        if yPred[i] == 0 and yTest[i] == 1:
            confusionMatrix['4'] += 1
        if yPred[i] == 1 and yTest[i] == 1:
            confusionMatrix['5'] += 1
        if yPred[i] == 2 and yTest[i] == 1:
            confusionMatrix['6'] += 1
        if yPred[i] == 0 and yTest[i] == 2:
            confusionMatrix['7'] += 1
        if yPred[i] == 1 and yTest[i] == 2:
            confusionMatrix['8'] += 1
        if yPred[i] == 2 and yTest[i] == 2:
            confusionMatrix['9'] += 1
    table = [
        [confusionMatrix['1'], confusionMatrix['2'], confusionMatrix['3']],
        [confusionMatrix['4'], confusionMatrix['5'], confusionMatrix['6']],
        [confusionMatrix['7'], confusionMatrix['8'], confusionMatrix['9']]
    ]
    cfMatrix = pd.DataFrame(table, columns=['1', '2', '3'], index=['1', '2', '3'])
    return cfMatrix

print('My accuracy:', str(sum([1 for i in range(len(yPred)) if yPred[i] == yTest[i]]) / len(yPred) * 100) + '%')
print('Random Forest accuracy:', str(sum([1 for i in range(len(yPred)) if rfPred[i] == yTest[i]]) / len(yPred) * 100) + '%')
print()
print('My confusion matrix:')
cMatrix1 = createConfusionMatrix(yPred, yTest)
print(cMatrix1)
print()
print('Random Forest confusion matrix:')
cMatrix2 = createConfusionMatrix(rfPred, yTest)
print(cMatrix2)
