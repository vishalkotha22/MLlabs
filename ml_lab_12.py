import random, time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron

df = pd.read_csv('iris.csv')
encode = []
for flower in df['class']:
    if flower == 'Iris-setosa':
        encode.append(0)
    elif flower == 'Iris-versicolor':
        encode.append(1)
    else:
        encode.append(2)

df['class'] = encode

weights = [random.random()] * 4
bias = random.random()

train = df.sample(frac=0.8, random_state=37)
test = df.drop(train.index)

ppn = Perceptron(max_iter = 40, eta0 = 0.1, tol = 1e-3)
ppn.fit(train[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']], train['class'])
preds = ppn.predict(test[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']])

plt.scatter(test['sepallength'], test['sepalwidth'], c=test['class'])
plt.show()

plt.scatter(test['sepallength'], test['sepalwidth'], c=preds)
plt.show()

accuracy = sum([1 for i in range(len(preds)) if preds[i] == test['class'].tolist()[i]])/len(preds)
print(f'Accuracy: {accuracy*100}%')