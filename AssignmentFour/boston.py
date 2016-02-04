__author__ = 'ufli'

import numpy
import sklearn.datasets
import sklearn.neighbors
import sklearn.preprocessing
from sklearn.cross_validation import KFold


data = sklearn.datasets.load_boston()
features = data.data
lables = data.target

features_scaled = sklearn.preprocessing.scale(features)

kf = KFold(n=len(data['data']), n_folds=5, shuffle=True, random_state=42)

errors = []
for p in numpy.linspace(start=1, stop=10, num=200):
    regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
    error = sklearn.cross_validation.cross_val_score(regressor,
                                                     features_scaled,
                                                     lables,
                                                     scoring='mean_squared_error',
                                                     cv=kf)
    errors.append(numpy.mean(error))

print(numpy.argmax(errors))
print(errors[0])
