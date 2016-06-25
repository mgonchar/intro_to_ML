'''
Created on 12 июня 2016 г.

@author: miroslvgoncarenko
'''

import numpy as np
import pandas
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score

def make_grad_step(X, y, w1,w2,k,C):
    l = len(y)
    sum1 = 0
    sum2 = 0
    
    for i in range(0,l):
        logistics_member = 1 - np.power((1 + np.exp(-y[i]*(w1*X[i,0] + w2*X[i,1]))), -1)
        sum1 = sum1 + y[i]*X[i,0]*logistics_member
        sum2 = sum2 + y[i]*X[i,1]*logistics_member
        
    w1_new = w1*(1 - k*C) + k*sum1/l
    w2_new = w2*(1 - k*C) + k*sum2/l
    return np.array([w1_new, w2_new])

def minimize_Q(X,y):
    norm_delta = np.power(10.0, -5)
    iter_max   = np.power(10,4)
    k          = 0.1
    C          = 10
    w          = np.array([0,0])
    
    for i in range(0,iter_max):
        w_new = make_grad_step(X, y, w[0], w[1], k, C)
        print("\n step " + str(i) + "\n w_"+str(i)+": "+ str(w[0]) +" "+str(w[1])+"\n w_new_" + str(i) +": "+ str(w_new[0]) +" "+str(w_new[1])+"\n")
        if distance.euclidean(w,w_new) < norm_delta :
            w = w_new
            break
        else:
            w = w_new
            
    return w

def a(X,w):
    return np.power(1 + np.exp(-w[0]*X[:,0]-w[1]*X[:,1]), -1)
        
data = pandas.read_csv('data-logistic.csv',header=None)

y = data.ix[:,0]
X = data.ix[:,1:].as_matrix()

w = minimize_Q(X,y)

score = roc_auc_score(y, a(X,w))

shuffle = True