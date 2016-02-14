__author__ = 'ufli'

import sklearn.svm
import numpy as np
import sklearn.grid_search
from sklearn import datasets
import sklearn.feature_extraction.text
from sklearn.cross_validation import KFold

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

labels = newsgroups.target
features = sklearn.feature_extraction.text.TfidfVectorizer().fit_transform(newsgroups.data)

svm = sklearn.svm.SVC(random_state=241, kernel='linear')
kf = KFold(n=len(labels), n_folds=5, shuffle=True, random_state=241)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
gs = sklearn.grid_search.GridSearchCV(svm, grid, scoring='accuracy', cv=kf)
gs.fit(features, labels)

# c = gs.best_params_['C']

c = 10
svm = sklearn.svm.SVC(random_state=241, kernel='linear', C=c)
svm.fit(features, labels)
coefs = np.argsort(np.absolute(np.asarray(svm.coef_.todense())).reshape(-1))
idxs = coefs[-10:]

vect = sklearn.feature_extraction.text.TfidfVectorizer()
vect.fit_transform(newsgroups.data)

words = []
for idx in idxs:
    words.append(vect.get_feature_names()[idx])

words.sort()
print(words)


