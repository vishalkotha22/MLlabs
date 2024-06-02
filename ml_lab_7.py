import pandas as pd
from mlxtend.classifier import OneRClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

X, y = df[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']], df['class'].tolist()
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.33, random_state=42)

performances = []
confusionMatrixOneRScratch = {'1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0, '6' : 0, '7' : 0, '8' : 0, '9' : 0}

oneROutputs = []
for col in df.columns:
    if col != 'class':
        minVal, maxVal = min(xTrain[col]), max(xTrain[col])
        bestD1, bestD2, bestSimilarity, bestCombo = -1, -1, -1, []
        d1 = minVal
        while d1 < maxVal:
            d2 = d1+0.1
            while d2 < maxVal:
                for combo in range(6):
                    combos = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
                    yPred = []
                    for val in xTrain[col]:
                        if val <= d1:
                            yPred.append(combos[combo][0])
                        elif val <= d2:
                            yPred.append(combos[combo][1])
                        else:
                            yPred.append(combos[combo][2])
                    similarity = sum([1 for i in range(len(yPred)) if yPred[i] == yTrain[i]])
                    if similarity > bestSimilarity:
                        bestSimilarity = similarity
                        bestD1 = d1
                        bestD2 = d2
                        bestCombo = combos[combo]
                d2 += 0.1
            d1 += 0.1
        oneROutputs.append([col, bestD1, bestD2, bestSimilarity, bestCombo])
        yTestPred = []
        for i, testVal in enumerate(xTest[col]):
            if testVal <= bestD1:
                if i < 1:
                    print(xTest.iloc[i])
                    print(f'Classifying based on {col} resulted in {bestCombo[0]}')
                    print()
                yTestPred.append(bestCombo[0])
                if col == 'petalwidth':
                    if bestCombo[0] == yTest[i]:
                        if bestCombo[0] == 1:
                            confusionMatrixOneRScratch['1'] += 1
                        elif bestCombo[0] == 2:
                            confusionMatrixOneRScratch['5'] += 1
                        else:
                            confusionMatrixOneRScratch['9'] += 1
                    else:
                        if bestCombo[0] == 1:
                            if yTest[i] == 2:
                                confusionMatrixOneRScratch['2'] += 1
                            else:
                                confusionMatrixOneRScratch['3'] += 1
                        elif bestCombo[0] == 2:
                            if yTest[i] == 1:
                                confusionMatrixOneRScratch['4'] += 1
                            else:
                                confusionMatrixOneRScratch['6'] += 1
                        else:
                            if yTest[i] == 1:
                                confusionMatrixOneRScratch['7'] += 1
                            else:
                                confusionMatrixOneRScratch['8'] += 1
            elif testVal <= bestD2:
                if i < 1:
                    print(xTest.iloc[i])
                    print(f'Classifying based on {col} resulted in {bestCombo[1]}')
                    print()
                yTestPred.append(bestCombo[1])
                if col == 'petalwidth':
                    if bestCombo[1] == yTest[i]:
                        if bestCombo[1] == 1:
                            confusionMatrixOneRScratch['1'] += 1
                        elif bestCombo[1] == 2:
                            confusionMatrixOneRScratch['5'] += 1
                        else:
                            confusionMatrixOneRScratch['9'] += 1
                    else:
                        if bestCombo[1] == 1:
                            if yTest[i] == 2:
                                confusionMatrixOneRScratch['2'] += 1
                            else:
                                confusionMatrixOneRScratch['3'] += 1
                        elif bestCombo[1] == 2:
                            if yTest[i] == 1:
                                confusionMatrixOneRScratch['4'] += 1
                            else:
                                confusionMatrixOneRScratch['6'] += 1
                        else:
                            if yTest[i] == 1:
                                confusionMatrixOneRScratch['7'] += 1
                            else:
                                confusionMatrixOneRScratch['8'] += 1
            else:
                if i < 1:
                    print(xTest.iloc[i])
                    print(f'Classifying based on {col} resulted in {bestCombo[2]}')
                    print()
                yTestPred.append(bestCombo[2])
                if col == 'petalwidth':
                    if bestCombo[2] == yTest[i]:
                        if bestCombo[2] == 1:
                            confusionMatrixOneRScratch['1'] += 1
                        elif bestCombo[2] == 2:
                            confusionMatrixOneRScratch['5'] += 1
                        else:
                            confusionMatrixOneRScratch['9'] += 1
                    else:
                        if bestCombo[2] == 1:
                            if yTest[i] == 2:
                                confusionMatrixOneRScratch['2'] += 1
                            else:
                                confusionMatrixOneRScratch['3'] += 1
                        elif bestCombo[2] == 2:
                            if yTest[i] == 1:
                                confusionMatrixOneRScratch['4'] += 1
                            else:
                                confusionMatrixOneRScratch['6'] += 1
                        else:
                            if yTest[i] == 1:
                                confusionMatrixOneRScratch['7'] += 1
                            else:
                                confusionMatrixOneRScratch['8'] += 1
        performances.append((col, sum([1 for j in range(len(yTest)) if yTestPred[j] == yTest[j]])))


for output in oneROutputs:
    print(f"If {output[0]} has a value less than or equal to {output[1]}, classify it as {output[4][0]}. \n If {output[0]} has a value greater than {output[1]} but less than or equal to {output[2]}, classify it as {output[4][1]}. \n If {output[0]} has a value greater than {output[2]}, classify it as {output[4][2]}")
    print()

bestPerformance = 0
for col in confusionMatrixOneRScratch:
    if col == '1' or col == '5' or col == '9':
        bestPerformance += confusionMatrixOneRScratch[col]

print('One R from Scratch Performances: ')
for tup in performances:
    print(tup[0], tup[1])

print()
print('Accuracy for OneR from Scratch: ' + str(100 * bestPerformance / len(yTest)) + '%')

table = [
    [confusionMatrixOneRScratch['1'], confusionMatrixOneRScratch['2'], confusionMatrixOneRScratch['3']],
    [confusionMatrixOneRScratch['4'], confusionMatrixOneRScratch['5'], confusionMatrixOneRScratch['6']],
    [confusionMatrixOneRScratch['7'], confusionMatrixOneRScratch['8'], confusionMatrixOneRScratch['9']]
]
cfMatrixOneRScratch = pd.DataFrame(table, columns = ['1', '2', '3'], index=['1', '2', '3'])
print(cfMatrixOneRScratch)
print()

print('MLXtend One R Performance')
maxSimilarity = -1
cfMatrixMLX = {'1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0, '6' : 0, '7' : 0, '8' : 0, '9' : 0}
for col in ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']:
    oneR = OneRClassifier()
    oneR.fit(np.reshape(xTrain[col].tolist(), (100, 1)), np.asarray(yTrain))
    #print(oneR.prediction_dict_)
    predictions = oneR.predict(np.reshape(xTest[col].tolist(), (50, 1)))
    similarity = sum([1 for i in range(len(yTest)) if predictions[i] == yTest[i]])
    if col == 'petalwidth':
        for i in range(len(predictions)):
            if predictions[i] == yTest[i]:
                if predictions[i] == 1:
                    cfMatrixMLX['1'] += 1
                elif predictions[i] == 2:
                    cfMatrixMLX['5'] += 1
                else:
                    cfMatrixMLX['9'] += 1
            else:
                if predictions[i] == 1:
                    if yTest[i] == 2:
                        cfMatrixMLX['2'] += 1
                    else:
                        cfMatrixMLX['3'] += 1
                elif predictions[i] == 2:
                    if yTest[i] == 1:
                        cfMatrixMLX['4'] += 1
                    else:
                        cfMatrixMLX['6'] += 1
                else:
                    if predictions[i] == 1:
                        cfMatrixMLX['7'] += 1
                    else:
                        cfMatrixMLX['8'] += 1
    print(col, similarity)
    maxSimilarity = max(maxSimilarity, similarity)

print()

table = [
    [cfMatrixMLX['1'], cfMatrixMLX['2'], cfMatrixMLX['3']],
    [cfMatrixMLX['4'], cfMatrixMLX['5'], cfMatrixMLX['6']],
    [cfMatrixMLX['7'], cfMatrixMLX['8'], cfMatrixMLX['9']]
]
cfMatrixMLXTAB = pd.DataFrame(table, columns = ['1', '2', '3'], index=['1', '2', '3'])

print('Accuracy for MLXtend OneR: ' + str(100 * maxSimilarity / len(yTest)) + '%')
print()
print(cfMatrixMLXTAB)
print()

#print(oneROutputs)

probability = {}
probability[('0', 'class', 'class')] = (yTrain.count(0)+1) / len(yTrain)
probability[('1', 'class', 'class')] = (yTrain.count(1)+1) / len(yTrain)
probability[('2', 'class', 'class')] = (yTrain.count(2)+1) / len(yTrain)
for col in df.columns:
    if col != 'class':
        for uVal in xTrain[col].unique():
            probability[(uVal, col, col)] = (xTrain[col].value_counts()[uVal]+1) / len(xTrain[col])
            zeroLocs = [idx for idx, val in enumerate(yTrain) if val == 0]
            oneLocs = [idx for idx, val in enumerate(yTrain) if val == 1]
            twoLocs = [idx for idx, val in enumerate(yTrain) if val == 2]
            probability[(uVal, col, '0')] = (xTrain.iloc[zeroLocs, df.columns.get_loc(col)].tolist().count(uVal)+1) / len(zeroLocs)
            probability[(uVal, col, '1')] = (xTrain.iloc[oneLocs, df.columns.get_loc(col)].tolist().count(uVal)+1) / len(oneLocs)
            probability[(uVal, col, '2')] = (xTrain.iloc[twoLocs, df.columns.get_loc(col)].tolist().count(uVal)+1) / len(twoLocs)

def calcProbability(values, columns, classLabel):
    p1 = probability[('0', 'class', 'class')]
    p2 = probability[(values[0], columns[0], classLabel)] * probability[(values[1], columns[1], classLabel)] * probability[(values[2], columns[2], classLabel)] * probability[(values[3], columns[3], classLabel)]
    p3 = probability[(values[0], columns[0], columns[0])] * probability[(values[1], columns[1], columns[1])] * probability[(values[2], columns[2], columns[2])] * probability[(values[3], columns[3], columns[3])]
    return p1 * p2 / p3

nbMatrix = {'1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0, '6' : 0, '7' : 0, '8' : 0, '9' : 0}
yNBPred = []
for i, instance in enumerate(xTest.iterrows()):
    vals = [instance[1]['sepallength'], instance[1]['sepalwidth'], instance[1]['petallength'], instance[1]['petalwidth']]
    if i < 5:
        print(f'The instance we are predicting is {vals}')
    pZero = calcProbability(vals, ['sepallength', 'sepalwidth', 'petallength', 'petalwidth'], '0')
    pOne = calcProbability(vals, ['sepallength', 'sepalwidth', 'petallength', 'petalwidth'], '1')
    pTwo = calcProbability(vals, ['sepallength', 'sepalwidth', 'petallength', 'petalwidth'], '2')
    pMax = max(pZero, max(pOne, pTwo))
    if pZero == pMax:
        yNBPred.append(0)
        if 0 == yTest[i]:
            nbMatrix['1'] += 1
        else:
            if yTest[i] == 1:
                nbMatrix['2'] += 1
            else:
                nbMatrix['3'] += 1
    elif pOne == pMax:
        yNBPred.append(1)
        if 1 == yTest[i]:
            nbMatrix['5'] += 1
        else:
            if yTest[i] == 0:
                nbMatrix['4'] += 1
            else:
                nbMatrix['6'] += 1
    else:
        yNBPred.append(2)
        if 2 == yTest[i]:
            nbMatrix['9'] += 1
        else:
            if yTest[i] == 0:
                nbMatrix['7'] += 1
            else:
                nbMatrix['8'] += 1
    if i < 5:
        print(f'We predicted the instance as {yNBPred[-1]}')
        print()
table = [
    [nbMatrix['1'], nbMatrix['2'], nbMatrix['3']],
    [nbMatrix['4'], nbMatrix['5'], nbMatrix['6']],
    [nbMatrix['7'], nbMatrix['8'], nbMatrix['9']]
]
nbMatrixTAB = pd.DataFrame(table, columns = ['1', '2', '3'], index=['1', '2', '3'])

nbPerformance = sum([1 for i in range(len(yTest)) if yTest[i] == yNBPred[i]])
print(f'Naive Bayes from Scratch Performance: {nbPerformance}')
print()
print(f'Naive Bayes from Scratch Accuracy: {100 * nbPerformance / len(yTest)}%')
print()
print(nbMatrixTAB)
print()
#print(yNBPred)
#print(sum([1 for i in range(len(yTest)) if yTest[i] == yNBPred[i]]))

cnb = GaussianNB()
y_pred = cnb.fit(xTrain, yTrain).predict(xTest)
gbMatrix = {'1' : 0, '2' : 0, '3' : 0, '4' : 0, '5' : 0, '6' : 0, '7' : 0, '8' : 0, '9' : 0}
for i in range(len(predictions)):
    if y_pred[i] == yTest[i]:
        if y_pred[i] == 1:
            gbMatrix['5'] += 1
        elif y_pred[i] == 2:
            gbMatrix['9'] += 1
        else:
            gbMatrix['1'] += 1
    else:
        if y_pred[i] == 1:
            if yTest[i] == 2:
                gbMatrix['6'] += 1
            else:
                gbMatrix['3'] += 1
        elif y_pred[i] == 2:
            if yTest[i] == 1:
                gbMatrix['8'] += 1
            else:
                gbMatrix['2'] += 1
        else:
            if y_pred[i] == 1:
                gbMatrix['7'] += 1
            else:
                gbMatrix['4'] += 1
table = [
    [gbMatrix['1'], gbMatrix['2'], gbMatrix['3']],
    [gbMatrix['4'], gbMatrix['5'], gbMatrix['6']],
    [gbMatrix['7'], gbMatrix['8'], gbMatrix['9']]
]
gbMatrixTAB = pd.DataFrame(table, columns = ['1', '2', '3'], index=['1', '2', '3'])
#print(y_pred)
gbPerformance = sum([1 for i in range(len(yTest)) if yTest[i] == y_pred[i]])
print(f'Sci-Kit Learn Naive Bayes Performance: {gbPerformance}')
print()
print(f'Sci-Kit learn Naive Bayes Accuracy: {100 * gbPerformance / len(yTest)}%')
print()
print(gbMatrixTAB)
print()
