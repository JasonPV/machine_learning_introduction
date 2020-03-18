from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import load_boston
import sklearn
import numpy as np
x = load_boston()['data']
y = load_boston()['target']
x = sklearn.preprocessing.scale(x)

kf = KFold(n_splits= 5, shuffle= True, random_state= 42)
grid = np.linspace(1, 10, 200)

neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p = 1)
ac = cross_val_score(neigh,x, y, scoring= 'neg_mean_squared_error')
max = ac.mean()
p_max = 0

for p in grid:
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p = p)
    sq_er = cross_val_score(neigh,x, y, scoring= 'neg_mean_squared_error')
    avr = sq_er.mean()
    if avr > max:
        p_max = p
        max = avr

print(round(p_max, 1))