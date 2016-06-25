'''
Created on 10 июня 2016 г.

@author: miroslvgoncarenko
'''

import numpy as np
import pandas

from sklearn.svm import SVC

from sklearn import datasets

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn import grid_search

#data = pandas.read_csv('svm-data.csv',header=None)

#y = data.ix[:,0]
#X = data.ix[:,1:]

#clf = SVC(C = 100000, kernel='linear',random_state=241)

#clf.fit(X,y)

#sp = clf.support_

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

#newsgroups_v = datasets.fetch_20newsgroups_vectorized(
#                    subset='all', 
#                    categories=['alt.atheism', 'sci.space']
#             )

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(newsgroups.data) #.toarray()
y = newsgroups.target

feature_mapping = vectorizer.get_feature_names()

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241, verbose = True)
gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)

best_C = np.Inf
best_accurancy = 0

for a in gs.grid_scores_:
    # a.mean_validation_score — оценка качества по кросс-валидации
    # a.parameters — значения параметров
    if (a.mean_validation_score > best_accurancy):
        best_accurancy = a.mean_validation_score
        best_C = a.parameters

clf = SVC(kernel='linear', random_state=241, C=best_C.get('C'))    
clf.fit(X, y)    

Shuffle = True