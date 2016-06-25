'''
Created on 28 мая 2016 г.

@author: miroslvgoncarenko
'''
import pandas
import numpy as np

from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

n_female = np.nonzero(data['Sex']=='female')
sv = data['Survived'].as_matrix()
tt = sv[n_female]

clf = DecisionTreeClassifier(random_state=241)

Pclass = data['Pclass']
Fare   = data['Fare']
Age    = data['Age']
Sex    = data['Sex'] == 'male'

Target = data['Survived']

for i in range(len(Pclass), 0,-1):
    if np.isnan(Pclass[i]) or np.isnan(Fare[i]) or np.isnan(Age[i]) or np.isnan(Sex[i]) or np.isnan(Target[i]):
        del Pclass[i]
        del Fare[i]
        del Age[i]
        del Sex[i]
        del Target[i]
    print(i)
        
X = np.vstack((Pclass.as_matrix(), Fare.as_matrix(), Age.as_matrix(), Sex.as_matrix()))
clf.fit(X.transpose(), np.array(Target.as_matrix()))

importances = clf.feature_importances_
