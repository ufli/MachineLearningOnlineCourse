__author__ = 'ufli'

import numpy
import pandas
import sklearn
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier


def accuracy(prediction, labels):
    share = 0.
    for i in range(len(prediction)):
        if prediction[i] == labels.iloc[i]:
            share += 1
    return share / len(labels)


data = pandas.read_csv('wine.data.txt', header=None)
lables = data[0]

features = data[data.columns[1:]]

kf = KFold(n=len(data), n_folds=5, shuffle=True, random_state=42)

accs_k = []
for k in xrange(1, 50):
    knn = KNeighborsClassifier(n_neighbors=k)
    accs = []
    for train_index, test_index in kf:
        knn.fit(features.iloc[train_index], lables.iloc[train_index])
        accs.append(accuracy(knn.predict(features.iloc[test_index]), lables.iloc[test_index]))
    accs_k.append(pandas.Series.mean(pandas.Series(accs)))

print(numpy.argmax(accs_k))
print(numpy.max(accs_k))

features_scaled = sklearn.preprocessing.scale(features)

accs_k_scaled = []
for k in xrange(1, 50):
    knn = KNeighborsClassifier(n_neighbors=k)
    accs = []
    for train_index, test_index in kf:
        knn.fit(features_scaled[train_index], lables.iloc[train_index])
        accs.append(accuracy(knn.predict(features_scaled[test_index]), lables.iloc[test_index]))
    accs_k_scaled.append(pandas.Series.mean(pandas.Series(accs)))

print(numpy.argmax(accs_k_scaled))
print(numpy.max(accs_k_scaled))
