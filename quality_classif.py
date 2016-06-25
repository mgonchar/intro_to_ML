'''
Created on 13 июня 2016 г.

@author: miroslvgoncarenko
'''
import numpy as np
import pandas
from sklearn import metrics

def extractPrecission70Recall(ROC):
    i = 0
    while ROC[1][i] >= 0.7:
        i = i + 1
    return ROC[0][i]


bin_q_data = pandas.read_csv('quality_classification.csv')
y_true = bin_q_data.ix[:,0]
y_pred = bin_q_data.ix[:,1]

TP = np.dot(y_true,y_pred)
FP = len(np.nonzero(y_pred)[0]) - TP
zero_pred = np.where(y_pred == 0)[0]
FN = np.sum(y_true[zero_pred])
TN = len(zero_pred) - FN

Accuracy = np.round(metrics.accuracy_score(y_true,y_pred), 2)
Precision = np.round(metrics.precision_score(y_true,y_pred), 2)
Recall = np.round(metrics.recall_score(y_true,y_pred), 2)
F = np.round(metrics.f1_score(y_true,y_pred), 2)


#comp_data = pandas.read_csv('quality_scores.csv')
#y_true = comp_data['true'].as_matrix()
#score_logreg = comp_data['score_logreg'].as_matrix()
#score_svm = comp_data['score_svm'].as_matrix()
#score_knn = comp_data['score_knn'].as_matrix()
#score_tree = comp_data['score_tree'].as_matrix()

#score_logreg_roc_auc = metrics.roc_auc_score(y_true, score_logreg)
#score_svm_roc_auc = metrics.roc_auc_score(y_true, score_svm)
#score_knn_roc_auc = metrics.roc_auc_score(y_true, score_knn)
#score_tree_roc_auc = metrics.roc_auc_score(y_true, score_tree)

#roc_logreg = metrics.precision_recall_curve(y_true, score_logreg)
#roc_svm = metrics.precision_recall_curve(y_true, score_svm)
#roc_knn = metrics.precision_recall_curve(y_true, score_knn)
#roc_tree = metrics.precision_recall_curve(y_true, score_tree)

#recall_logreg = extractPrecission70Recall(roc_logreg)
#recall_svm = extractPrecission70Recall(roc_svm)
#recall_knn = extractPrecission70Recall(roc_knn)
#recall_tree = extractPrecission70Recall(roc_tree)


Shuffle = True