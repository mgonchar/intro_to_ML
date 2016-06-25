'''
Created on 5 июня 2016 г.

@author: miroslvgoncarenko
'''

from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
import numpy as np

data = datasets.load_boston()

features = scale(data.data)

p = np.linspace(1.0,10.0,num=200)

Kfold_generator = KFold(n=177, shuffle=True, random_state=42, n_folds=5)

best_cvs = np.inf
best_p = 0

for p_cur in p:

    reg = KNeighborsRegressor(n_neighbors=5,weights='distance', p=p_cur)
    
    cvs = cross_val_score(estimator = reg, X = features, y = data.target, scoring='mean_squared_error', cv=Kfold_generator)
    
    cvs_m = np.mean(cvs)
    if np.abs(cvs_m) < np.abs(best_cvs):
        best_cvs = cvs_m
        best_p = p_cur
    

stopper = True