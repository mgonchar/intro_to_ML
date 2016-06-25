'''
Created on 13 июня 2016 г.

@author: miroslvgoncarenko
'''


import numpy as np
import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

data_train = pandas.read_csv('salary-train.csv')
data_test = pandas.read_csv('salary-test-mini.csv')

tmp = data_train['FullDescription'].tolist()

for i in range(0,len(tmp)):
    tmp[i] = tmp[i].lower()

data_train['FullDescription'] = tmp

tmp = data_test['FullDescription'].tolist()

for i in range(0,len(tmp)):
    tmp[i] = tmp[i].lower()

data_test['FullDescription'] = tmp

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

tfid_vec = TfidfVectorizer(min_df=5)
X_train_full_decr = tfid_vec.fit_transform(data_train['FullDescription'])
X_test_full_decr  = tfid_vec.transform(data_test['FullDescription'])

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ  = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train_full = hstack([X_train_full_decr, X_train_categ])
X_test_full  = hstack([X_test_full_decr, X_test_categ])

y_train = data_train['SalaryNormalized']

clf = Ridge(alpha=1, random_state=241)

clf.fit(X_train_full, y_train)

y_pred = clf.predict(X_test_full)

Shuffle = True