from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('unbalanced_data.csv')
X = data[[col for col in data.columns if col != 'Outcome']]
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
print(Counter(y))
print(Counter(y_train))
print(Counter(y_test))