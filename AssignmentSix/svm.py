__author__ = 'ufli'

import pandas
import sklearn.svm

data = pandas.read_csv('svm_data.csv', header=None)
lables = data[0]
features = data[data.columns[1:]]

print(lables)
print(features)

svm = sklearn.svm.SVC(C=100000, random_state=241)
svm.fit(features, lables)
print(svm.support_)
