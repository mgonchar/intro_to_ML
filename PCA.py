'''
Created on 15 июня 2016 г.

@author: miroslvgoncarenko
'''

import numpy as np
import pandas
from sklearn.decomposition import PCA

close_prices = pandas.read_csv('close_prices.csv')
X_cp = close_prices.ix[:,1:]

pca_trnsf = PCA(n_components=10)

pca_trnsf.fit(X_cp)

first_axis = pca_trnsf.components_[0]
first_comp = np.dot(X_cp.as_matrix(),first_axis)

dj_index = pandas.read_csv('djia_index.csv')

X_dj = dj_index.ix[:,1:]

corr = np.corrcoef(np.array([np.squeeze(X_dj._get_values), first_comp]))

Shuffle = True