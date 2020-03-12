import pandas as pd
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state= 241)
df = pd.read_csv('titanic.csv', index_col = 'PassengerId')
x = df[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
x.Sex.replace(['male', 'female'], [1, 0], inplace = True)
x.dropna(inplace = True)
y = x[['Survived']]
x = x[['Pclass', 'Fare', 'Age', 'Sex']]
clf.fit(x, y)
importances = clf.feature_importances_
lst = list(x)
for i in range(x.shape[1]):
    print(lst[i], importances[i])

