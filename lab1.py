import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

data = pd.read_csv('C:/Vishal/ML/iris.csv')

print(data.head(10))

fig, (ax1, ax2) = plt.subplots(1, 2)
red_patch = mpatches.Patch(color='red', label='Iris setosa')
green_patch = mpatches.Patch(color='green', label='Iris versicolor')
blue_patch = mpatches.Patch(color='blue', label='Iris virginica')
fig.legend(handles=[red_patch, green_patch, blue_patch])
colors = ['red' if ct == 'Iris-setosa' else 'green' if ct == 'Iris-versicolor' else 'blue' for ct in data['class']]
fig.suptitle("Graphs Displaying Irises")
ax1.scatter(data['sepallength'], data['sepalwidth'], color=colors)
ax1.set_title(' ', pad=25)
ax1.set_xlabel('Sepal Length')
ax1.set_ylabel('Sepal Width')
ax2.scatter(data['petallength'], data['petalwidth'], color=colors)
ax2.set_title(' ', pad=25)
ax2.set_xlabel('Petal Length')
ax2.set_ylabel('Petal Width')
fig.tight_layout()
plt.show()