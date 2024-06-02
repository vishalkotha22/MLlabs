import random, time
import matplotlib.pyplot as plt
import numpy as np

inputLength = 3
learningRate = 0.1
startTime = time.time()

def activationFunction(val):
   if val >= 0:
       return 1
   else:
       return 0


trainPts = []
truth = [0] * 32
for i in range(32):
    binString = "{0:b}".format(i)
    while len(binString) < 5:
        binString = '0' + binString
    temp = []
    for ch in binString:
        temp.append(int(ch))
    trainPts.append(temp)
    if i == 6 or i == 11 or i == 12 or i == 17 or i == 23 or i == 25:
        truth[i] = 1

def run(trainPts, truth):
    weights = [[-1, -1, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1, -1, 0, -1, 0, -1, 0, -1, 0], [-1, 1, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, -1, 1], [1, 1, 1]]
    bias = [[1, 0, 1, 0, 1, 0], [0, 0, 0], [-3]]
    weights = [[-1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1], [1, 1, 1, 1, 1, 1]]
    bias = [[-2, -3, -2, -2, -4, -3], [-1]]
    idx = 0
    for a, b, c, d, e in trainPts:
       h1 = activationFunction(a*weights[0][0]+b*weights[0][1]+c*weights[0][2]+d*weights[0][3]+e*weights[0][4]+bias[0][0])
       h2 = activationFunction(a*weights[0][5]+b*weights[0][6]+c*weights[0][7]+d*weights[0][8]+e*weights[0][9]+bias[0][1])
       h3 = activationFunction(a*weights[0][10]+b*weights[0][11]+c*weights[0][12]+d*weights[0][13]+e*weights[0][14]+bias[0][2])
       h4 = activationFunction(a*weights[0][15]+b*weights[0][16]+c*weights[0][17]+d*weights[0][18]+e*weights[0][19]+bias[0][3])
       h5 = activationFunction(a*weights[0][20]+b*weights[0][21]+c*weights[0][22]+d*weights[0][23]+e*weights[0][24]+bias[0][4])
       h6 = activationFunction(a*weights[0][25]+b*weights[0][26]+c*weights[0][27]+d*weights[0][28]+e*weights[0][29]+bias[0][5])
       pOutput = activationFunction(h1*weights[1][0]+h2*weights[1][1]+h3*weights[1][2]+h4*weights[1][3]+h5*weights[1][4]+h6*weights[1][5]+bias[1][0])
       print(f'{a}, {b}, {c}, {d}, {e} was classified as {pOutput}. The truth is {truth[idx]}.')
       #if truth[idx] != pOutput:
           #print(f'{x}, {y} was misclassified as {pOutput} when actually {truth[idx]}')
       idx += 1

    print(weights, bias)
    print()

run(trainPts, truth)