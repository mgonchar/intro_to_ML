'''
Created on 18 июня 2016 г.

@author: miroslvgoncarenko
'''

import numpy as np
import pandas

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

def convert_to_prob(X):
    return np.power((np.exp(-X) + 1),-1)

data = pandas.read_csv('gbm-data.csv')

y = data.ix[:,0]
X = data.ix[:,1:].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

n_estimators = 250
#for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
#    gbc = GradientBoostingClassifier(n_estimators=n_estimators, verbose=True, random_state=241, learning_rate=learning_rate)
#    gbc.fit(X_train,y_train)
#    score_train = np.zeros((X_train.shape[0],n_estimators,), dtype=np.float64)
#    for i, el in enumerate(gbc.staged_decision_function(X_train)): 
#        score_train[:,i] = np.squeeze(el)
#    y_train_prob = convert_to_prob(score_train)
    
#    log_loss_train = np.zeros((n_estimators,), dtype=np.float64)
#    for i in range(0, n_estimators):
#        log_loss_train[i] = log_loss(y_train, y_train_prob[:,i])
        
    #gbc.predict(X_test)
    
#    score_test = np.zeros((X_test.shape[0],n_estimators,), dtype=np.float64)
#    for i, el in enumerate(gbc.staged_decision_function(X_test)): 
#        score_test[:,i] = np.squeeze(el)
#    y_test_prob = convert_to_prob(score_test)
    
#    log_loss_test = np.zeros((n_estimators,), dtype=np.float64)
#    for i in range(0, n_estimators):
#        log_loss_test[i] = log_loss(y_test, y_test_prob[:,i])
    
    #plt.figure()
    #plt.plot(log_loss_test, 'r', linewidth=2)
    #plt.plot(log_loss_train, 'g', linewidth=2)
    #plt.legend(['test', 'train'])
    #plt.show()
    
#    shfl = True
    
rfc = RandomForestClassifier(n_estimators=37, random_state=241)

rfc.fit(X_train, y_train)
y_forest_prob = rfc.predict_proba(X_test)

log_loss_forest = log_loss(y_test, y_forest_prob)

Shuffle = True
