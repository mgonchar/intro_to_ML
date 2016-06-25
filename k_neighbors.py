'''
Created on 5 июня 2016 г.

@author: miroslvgoncarenko
'''

import pandas
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data = pandas.read_csv('wine.data')

classes = data.ix[:,0]
features = data.ix[:,1:]

Kfold_generator = KFold(n=177, shuffle=True, random_state=42, n_folds=5)

best_k = 1
best_cvs = 0

X = features.as_matrix()#scale(features.as_matrix())

for k in range(1,51):

    classifier = KNeighborsClassifier(n_neighbors=k)
    
    cvs = cross_val_score(estimator = classifier, X = X, y = classes.as_matrix(), scoring='accuracy', cv=Kfold_generator)
    
    m_cvs = np.mean(cvs)
    
    if (m_cvs > best_cvs):
        best_cvs = m_cvs
        best_k = k

shuffle = True