import pandas as pd
import math

from sklearn.model_selection import train_test_split

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

df.to_csv('binnedIris.csv')

#print(df['class'].unique().tolist())
#print(len(df['class'].unique().tolist()))
#print(df.head(100).mode()['class'].tolist())
#print(df[df['petallength'] == 5.4])

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
                for val in data[attr].unique():
                    subData = data[data[attr] == val]
                    entropy = calcEntropy(subData)
                    if entropy > 0:
                        infoGain -= (len(subData) / len(data)) * entropy
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

X, y = df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']], df['class'].tolist()
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=.2, random_state=42)
xTrainUse = xTrain.copy()
xTrain['class'] = yTrain
xTCopy = xTest.copy()
xTCopy['class'] = yTest
xTrain.to_csv('irisTrain.csv')
xTCopy.to_csv('irisTest.csv')

root = Node(xTrain, '', None, [])
recur(xTrain, ['sepalwidth', 'sepallength', 'petalwidth', 'petallength'], root)
#print(root.attr)
#print(df['sepallength'].unique())

yPred = []

for idx, row in xTest.iterrows():
    start = root
    while len(start.children) > 0:
        #print(start.attr, start.output, start.children)
        if start.attr != "LEAF":
            for child in start.children:
                if child.output == row[start.attr]:
                    start = child
                    break
        else:
            start = start.children[0]
            yPred.append(start.output)
            #print(start.output)
    if idx <= 30:
        print(f'{row} was classified as {yPred[-1]}\n')

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

#print('Using the complete dataset, we split on', root.attr)
dfs(root, root, 0)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
xTrain.drop('class', axis=1)
clf.fit(xTrainUse, yTrain)
skPred = clf.predict(xTest)
print('My predictions:', yPred)
print('SKLearn predictions:', skPred)
print()
print('My accuracy:', str(sum([1 for i in range(len(yPred)) if yPred[i] == yTest[i]]) / len(yTest) * 100) + '%')
print('SKlearn Accuracy: ', str(sum([1 for i in range(len(yPred)) if yTest[i] == skPred[i]]) / len(yTest) * 100) + '%')
confusionMatrix = {'1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0, '6' : 0, '7' : 0, '8' : 0, '9' : 0}
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
cfMatrix = pd.DataFrame(table, columns = ['1', '2', '3'], index=['1', '2', '3'])
print('My model confusion matrix: ')
print(cfMatrix)
cMatrix = {'1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0, '6' : 0, '7' : 0, '8' : 0, '9' : 0}
for i in range(len(yPred)):
    if yPred[i] == 0 and yTest[i] == 0:
        cMatrix['1'] += 1
    if yPred[i] == 1 and yTest[i] == 0:
        cMatrix['2'] += 1
    if yPred[i] == 2 and yTest[i] == 0:
        cMatrix['3'] += 1
    if yPred[i] == 0 and yTest[i] == 1:
        cMatrix['4'] += 1
    if yPred[i] == 1 and yTest[i] == 1:
        cMatrix['5'] += 1
    if yPred[i] == 2 and yTest[i] == 1:
        cMatrix['6'] += 1
    if yPred[i] == 0 and yTest[i] == 2:
        cMatrix['7'] += 1
    if yPred[i] == 1 and yTest[i] == 2:
        cMatrix['8'] += 1
    if yPred[i] == 2 and yTest[i] == 2:
        cMatrix['9'] += 1
table2 = [
    [cMatrix['1'], cMatrix['2'], cMatrix['3']],
    [cMatrix['4'], cMatrix['5'], cMatrix['6']],
    [cMatrix['7'], cMatrix['8'], cMatrix['9']]
]
cfMatrix2 = pd.DataFrame(table2, columns = ['1', '2', '3'], index=['1', '2', '3'])
print('SKLearn confusion matrix: ')
print(cfMatrix2)