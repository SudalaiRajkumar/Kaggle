# -*- coding: utf-8 -*-
"""
Created on Jul 28 17:17:40 2014

@author: Sudalai Rajkumar S
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
sample = pd.read_csv(data_path+'sampleSubmission.csv')

### Three training sets are created based on different feature selection methodologies ###
### Set1 - Run univariate regression to get the top 30 features ###
feature_selector = SelectKBest(score_func=f_regression, k=30)
feature_selector.fit(tr, train_y)
tr1 = feature_selector.transform(tr)
ts1 = feature_selector.transform(ts)

### Set 2 & 3 - Features selected based on stepwise cross validation ( (tr2,ts2) and (tr3,ts3) )###
tr2 = tr[:,[288, 334,  50, 359,  29, 238,  45, 369, 188, 183, 225, 370, 310,  40,  63, 321, 226, 119,   2, 300, 291, 157, 303, 214,  46, 282, 349, 155,  32, 120, 100, 264, 382, 331, 180, 302, 295, 312, 372, 1, 335, 385, 387, 378, 338, 381, 6, 5, 0, 3]]
ts2 = ts[:,[288, 334,  50, 359,  29, 238,  45, 369, 188, 183, 225, 370, 310,  40,  63, 321, 226, 119,   2, 300, 291, 157, 303, 214,  46, 282, 349, 155,  32, 120, 100, 264, 382, 331, 180, 302, 295, 312, 372, 1, 335, 385, 387, 378, 338, 381, 6, 5, 0, 3]]

tr3 = tr[:,[0,2,3,5,6,7,40,157,245,288,305,310,312,321,323,338,372,378]]
ts3 = ts[:,[0,2,3,5,6,7,40,157,245,288,305,310,312,321,323,338,372,378]]

### Running ridge regression using all three train samples then make predictions on test set separately ###
# Model 1 #
clf = Ridge()
clf.fit(tr1, train_y)
preds1 = clf.predict(ts1)

# Model 2 #
clf = Ridge()
clf.fit(tr2, train_y)
preds2 = clf.predict(ts2)

# Model 3#
clf = Ridge()
clf.fit(tr3, train_y)
preds3 = clf.predict(ts3)

### Ensembling the models together ###
preds = (0.2*preds1) + (0.32*preds2)+ (0.48*preds3)

### Writing the outputs to out file ###
sample['target'] = preds
sample.to_csv('submission.csv', index = False)
