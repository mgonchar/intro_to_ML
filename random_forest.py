'''
Created on 16 июня 2016 г.

@author: miroslvgoncarenko
'''

import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import r2_score

data = pandas.read_csv('abalone.csv')

data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

y_age = data.ix[:,-1]

X = data.ix[:,:-1].as_matrix()

Kfold_generator = KFold(n=len(y_age), shuffle=True, random_state=1, n_folds=5)

best_n_trees = 0
best_score = -np.Inf
for i in range(1,51):
    forest = RandomForestRegressor(n_estimators = i, random_state=1)
    cvs = cross_val_score(estimator = forest, X = X, y = y_age, scoring='r2', cv=Kfold_generator)
    
    cvs_m = np.mean(cvs)
    print("n_trees: " + str(i) + " mean R2: " + str(cvs_m)+"\n")
    if cvs_m > best_score:
        best_score = cvs_m
        best_n_trees = i

Shuffle = True