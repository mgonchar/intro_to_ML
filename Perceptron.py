'''
Created on 8 июня 2016 г.

@author: miroslvgoncarenko
'''

import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train = pandas.read_csv('perceptron-train.csv',header=None)
test  = pandas.read_csv('perceptron-test.csv',header=None)

train_y = train.ix[:,0]
train_X = train.ix[:,1:]

test_y = test.ix[:,0]
test_X = test.ix[:,1:]

clf = Perceptron(random_state=241)

clf.fit(train_X, train_y)

y_tmp_test = clf.predict(test_X)
tst_score = accuracy_score(test_y, y_tmp_test)

X_train_scaled = scaler.fit_transform(train_X)
X_test_scaled  = scaler.transform(test_X)

clf.fit(X_train_scaled, train_y)

y_tmp_test_fitted = clf.predict(X_test_scaled)
tst_score_fit = accuracy_score(test_y, y_tmp_test_fitted)

delta_quality = tst_score_fit - tst_score

Shuffle = True