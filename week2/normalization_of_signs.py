import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.read_csv('perceptron-train.csv', header= None)
df1 = pd.read_csv('perceptron-test.csv', header= None)
clf = Perceptron()

x_train = df[[1, 2]]
y_train = df[0]

x_test = df1[[1, 2]]
y_test = df1[0]

clf.fit(x_train, y_train)
accuracy1 = accuracy_score(y_test, clf.predict(x_test))

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


clf.fit(x_train, y_train)
accuracy2 = accuracy_score(y_test, clf.predict(x_test))
print(round(accuracy2 - accuracy1, 3))

