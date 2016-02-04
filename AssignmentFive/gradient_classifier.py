__author__ = 'ufli'

import pandas
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler


def accuracy(prediction, labels):
    share = 0.
    for i in range(len(prediction)):
        if prediction[i] == labels.iloc[i]:
            share += 1
    return share / len(labels)


test = pandas.read_csv('test.csv', header=None)
train = pandas.read_csv('train.csv', header=None)

train_labels = train[0]
train_features = train[train.columns[1:]]

test_labels = test[0]
test_features = test[test.columns[1:]]

clf = Perceptron(random_state=241)
clf.fit(train_features, train_labels)
predictions = clf.predict(test_features)
acc_before = accuracy(predictions, test_labels)

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

clf.fit(train_features_scaled, train_labels)
predictions = clf.predict(test_features_scaled)
acc_after = accuracy(predictions, test_labels)

print(acc_after - acc_before)
