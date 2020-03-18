from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import sklearn
import pandas as pd
import numpy as np

def optimal_values(x, y):
    max = 0
    k_max = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for k in range(1, 51):
        knn = KNeighborsClassifier(n_neighbors=k)
        accuracy = cross_val_score(knn, x, y, cv=kf, scoring='accuracy')
        avr = accuracy.mean()
        if avr > max:
            max = avr
            k_max = k
    return k_max, round(max, 2)

data = pd.read_csv('wine.data', header = None)
y = np.array(data[0])
x = np.array(data.loc[:, 1:])


print(optimal_values(x, y))

x = sklearn.preprocessing.scale(x)

print(optimal_values(x, y))
