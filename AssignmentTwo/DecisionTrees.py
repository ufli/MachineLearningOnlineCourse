__author__ = 'ufli'

import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def toInt(str):
    return 0 if str == 'male' else 1


data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data = data[~np.isnan(data['Age'])]
data['Sex'] = map(lambda x: toInt(x), data['Sex'])

train_data = data[['Pclass', 'Age', 'Sex', 'Fare']].values
train_data = [value for value in train_data if not np.isnan(value[1])]

labels = data.Survived

clf = DecisionTreeClassifier(random_state=241)
clf.fit(train_data, labels)
importances = clf.feature_importances_
print(importances)
