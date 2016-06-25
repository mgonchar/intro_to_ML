'''
Created on 22 мая 2016 г.

@author: miroslvgoncarenko
'''

import pandas
import numpy as np

import itertools
import operator

def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

a = data['Pclass']==1
a = np.count_nonzero(a)

n_female = np.nonzero(data['Sex']=='female')
names = data['Name'].as_matrix()
tt = names[n_female]

for i in range(0,len(tt)):
    if 'Mrs.' in tt[i] and '(' in tt[i]:
        nm = tt[i].split('(')
        tt[i] = nm[1].split(' ')[0]
    else:
        nm = tt[i].split(', ')
        tt[i] = nm[1].split(' ')[1]
    tt[i] = tt[i].replace(')','')
    print(i)
    print(tt[i])
    
print (most_common(tt))