# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:17:40 2014

@author: Sudalai Rajkumar S

This code is for selecting the features to run the final model
Feature sets 2 and 3 in the final model are selected based on this code
Feature selection uses stepwise forward feature selection algorithm that maximizes weighted gini coefficient
"""
from __future__ import division
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LogisticRegression, SGDClassifier, LinearRegression
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score

import pandas as pd
import numpy as np

def weighted_gini(act,pred,weight):
    df = pd.DataFrame({"act":act,"pred":pred,"weight":weight})
    df = df.sort('pred',ascending=False)
    df["random"] = (df.weight / df.weight.sum()).cumsum()
    total_pos = (df.act * df.weight).sum()
    df["cum_pos_found"] = (df.act * df.weight).cumsum()
    df["lorentz"] = df.cum_pos_found / total_pos
    #n = df.shape[0]
    #df["gini"] = (df.lorentz - df.random) * df.weight 
    #return df.gini.sum()
    gini = sum(df.lorentz[1:].values * (df.random[:-1])) - sum(df.lorentz[:-1].values * (df.random[1:]))
    return gini

def normalized_weighted_gini(act,pred,weight):
    return weighted_gini(act,pred,weight) / weighted_gini(act,act,weight)

data_path = "Path to data"

tr = np.load(data_path+"train.npy")
ts = np.load(data_path+"test.npy")
train_y = np.load(data_path+"train_y.npy")


### Feature selection using stepwise fashion based on cross validation ###
# This code will select one variable at a time from the given input variables using greedy approach which maximizes weighted gini metric #
print "Cross Validating.."
wt_gini = 0
kf = KFold(tr.shape[0], n_folds=5)
for i in xrange(tr.shape[1]):
    cv_gini_list=[]
    for dev_index, val_index in kf:
	tr_new = tr[:,[i]]
        #tr_new = tr[:,[0,2,3,5,6,7,40,157,245,288,305,310,312,321,323,338,372,378,i]]
        #tr_new = tr[:,[288, 334,  50, 359,  29, 238,  45, 369, 188, 183, 225, 370, 310,  40,  63, 321, 226, 119,   2, 300, 291, 157, 303, 214,  46, 282, 349, 155,  32, 120, 100, 264, 382, 331, 180, 302, 295, 312, 372, 1, 335, 385, 387, 378, 338, 381, 6, 5, 0, 3, i]]
        X_dev, X_val = tr_new[dev_index,:], tr_new[val_index,:]
        y_dev, y_val = train_y[dev_index], train_y[val_index]
        wt_dev, wt_val = tr[dev_index,1], tr[val_index,1]
        clf = Ridge()
        clf.fit(X_dev, y_dev)
        preds = clf.predict(X_val)
        
        cv_gini_list.append(normalized_weighted_gini(y_val,preds,wt_val))
    print cv_gini_list
    print np.mean(cv_gini_list)
    if np.mean(cv_gini_list) > wt_gini:
        wt_gini = np.mean(cv_gini_list)
        selected_index = i
    if i % 50 == 0:
        print "Processed : ",i
print wt_gini
print selected_index
