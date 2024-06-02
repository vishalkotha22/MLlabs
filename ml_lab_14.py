import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

LEARNING_RATE = 0.001

model = Sequential()
model.add(Dense(128, input_dim=784, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

def oneHotEncode(num):
    temp = [0] * 10
    temp[num] = 1
    return temp

xTrain = []
yTrain = []
for row in train.iterrows():
    yTrain.append(row[1][0])
    xTrain.append(np.array(row[1][1:]))

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

xTest = []
yTest = []
for row in test.iterrows():
    yTest.append(row[1][0])
    xTest.append(np.array(row[1][1:]))

xTest = np.array(xTest)
yTest = np.array(yTest)

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

epochs = [*range(1, 11)]
losses = []
accuracies = []
weightOverTime = []
gradients = []

for epoch in range(10):
    hist = model.fit(xTrain, yTrain, epochs=1)

    losses.append(hist.history['loss'])
    accuracies.append(hist.history['sparse_categorical_accuracy'])

    weightOverTime.append(model.layers[2].get_weights()[0][0][0])

for i in range(len(weightOverTime)-1):
    gradients.append((weightOverTime[i] - weightOverTime[i+1]) / LEARNING_RATE)

plt.plot(epochs, losses)
plt.show()

plt.plot(epochs, accuracies)
plt.show()

plt.plot(weightOverTime, losses)
plt.show()

scores = model.evaluate(xTest, yTest)
print('Accuracy:', scores[1]*100)

print(weightOverTime)
print(gradients)