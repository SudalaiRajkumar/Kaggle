
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:17:40 2014

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

### Reading the Input files ###
data_path = "Path to data"
train = pd.read_csv(data_path+'train.csv')
test = pd.read_csv(data_path+'test.csv')

### creating dummy variables from categorical variables ###
train_var1 = pd.get_dummies(train['var1'])
test_var1 = pd.get_dummies(test['var1'])

train_var2 = pd.get_dummies(train['var2'])
test_var2 = pd.get_dummies(test['var2'])

train_var3 = pd.get_dummies(train['var3'])
test_var3 = pd.get_dummies(test['var3'])

train_var4 = pd.get_dummies(train['var4'])
test_var4 = pd.get_dummies(test['var4'])

train_var5 = pd.get_dummies(train['var5'])
test_var5 = pd.get_dummies(test['var5'])

train_var6 = pd.get_dummies(train['var6'])
test_var6 = pd.get_dummies(test['var6'])

train_var7 = pd.get_dummies(train['var7'])
test_var7 = pd.get_dummies(test['var7'])

train_var8 = pd.get_dummies(train['var8'])
test_var8 = pd.get_dummies(test['var8'])

train_var9 = pd.get_dummies(train['var9'])
test_var9 = pd.get_dummies(test['var9'])

### Stacking the dummy variables together with the numerical variables ###
train = np.hstack([train.iloc[:,11:19], train.iloc[:,20:], train_var1, train_var2, train_var3, train_var4, train_var5, train_var6, train_var7, train_var8, train_var9])
test = np.hstack([test.iloc[:,10:18], test.iloc[:,19:], test_var1, test_var2, test_var3, test_var4, test_var5, test_var6, test_var7, test_var8, test_var9])

### Replacing the missing values with zero ###
train = np.nan_to_num(np.array(train)).astype('float64')
test = np.nan_to_num(np.array(test)).astype('float64')

### Saving the outputs as .npy file ###
np.save("train.npy", train)
np.save("test.npy", test)
np.save("train_y.npy", train['target'].values)
