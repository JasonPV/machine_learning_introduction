import pandas as pd

df = pd.read_csv('titanic.csv', index_col= 'PassengerId')
#number of men and women
print(df['Sex'].value_counts()['male'], end = ' ')
print(df['Sex'].value_counts()['female'])
#share of survivors
print(round((df['Survived'][df.Survived == 1].count()/df['Survived'].count()) * 100, 2))
# share of first-class passengers
print(round((df['Pclass'][df.Pclass == 1].count()/ df['Pclass'].count()) * 100, 2))
#average age and median age
print(round(df['Age'].mean(), 2), round(df['Age'].median(), 2))
#correlation
print(round(df[['SibSp', 'Parch']].corr(method='pearson')['SibSp']['Parch'], 2))
#most popular female name
d = {}
for f in df['Name']:
    str = f.split()
    if str[1] == 'Miss.':
        if str[2] in d.keys():
            d[str[2]] += 1
        else:
            d[str[2]] = 1
for i in d.keys():
    if d[i] == max(d.values()):
        print(i)