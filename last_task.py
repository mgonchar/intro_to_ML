'''
Created on 25 июня 2016 г.

@author: miroslvgoncarenko
'''

import numpy as np
import pandas
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score

import time
import datetime

features_train = pandas.read_csv('./last_task_features.csv', index_col='match_id')
#features_test  = pandas.read_csv('./last_task_features_test.csv', index_col='match_id')

# find incomplete features
n_rows = features_train.shape[0]
filled = features_train.count()
idx = np.nonzero(filled - n_rows)
incomplete_features = features_train.columns.values[idx]

# fill gaps with zeros
features_train = features_train.fillna(0)

y_train = features_train['radiant_win'].as_matrix()

X_train = features_train.drop(['duration','radiant_win','tower_status_radiant','tower_status_dire','barracks_status_radiant','barracks_status_dire'],1).as_matrix()

Kfold_generator = KFold(n=len(y_train), shuffle=True, random_state=42, n_folds=5)

for learning_rate in [1, 0.5, 0.3, 0.2, 0.1]:
    for n_estimators in [10, 15, 20, 25, 30]:
        for scoring in ['accuracy', 'log_loss']:
            gbc = GradientBoostingClassifier(n_estimators=n_estimators, verbose=False, random_state=241, learning_rate=learning_rate)
            
            start_time = datetime.datetime.now()
            cvs = cross_val_score(estimator = gbc, X = X_train, y = y_train, scoring='accuracy', cv=Kfold_generator)
            time_spent = datetime.datetime.now() - start_time
        
            m_cvs = np.mean(cvs)
            
            # verbose
            print("CV finished with result: "+str(m_cvs)+" time spent: "+str(time_spent)+"\n\tlearning_rate: "+str(learning_rate)+"\n\tn_estimators: "+str(n_estimators)+"\n\tscoring: "+scoring+"\n")

            
    #        gbc.fit(X_train,y_train)
    #        score_train = np.zeros((X_train.shape[0],n_estimators,), dtype=np.float64)
    #        for i, el in enumerate(gbc.staged_decision_function(X_train)): 
    #            score_train[:,i] = np.squeeze(el)
    #        y_train_prob = convert_to_prob(score_train)
            
    #        log_loss_train = np.zeros((n_estimators,), dtype=np.float64)
    #        for i in range(0, n_estimators):
    #            log_loss_train[i] = log_loss(y_train, y_train_prob[:,i])
                
    #        gbc.predict(X_test)
            
    #        score_test = np.zeros((X_test.shape[0],n_estimators,), dtype=np.float64)
    #        for i, el in enumerate(gbc.staged_decision_function(X_test)): 
    #            score_test[:,i] = np.squeeze(el)
    #        y_test_prob = convert_to_prob(score_test)
            
    #        log_loss_test = np.zeros((n_estimators,), dtype=np.float64)
    #        for i in range(0, n_estimators):
    #            log_loss_test[i] = log_loss(y_test, y_test_prob[:,i])
            
            #plt.figure()
            #plt.plot(log_loss_test, 'r', linewidth=2)
            #plt.plot(log_loss_train, 'g', linewidth=2)
            #plt.legend(['test', 'train'])
            #plt.show()

shuffle = True