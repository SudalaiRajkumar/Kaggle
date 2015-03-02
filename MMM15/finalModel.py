# -*- coding: utf-8 -*-
"""
Created on Tue Mar 02 15:28:42 2015

@author: Sudalai Rajkumar S
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import cross_val_score

def runLogistic(X, y, test_X, C_val = 1, penalty_val='l1'):
        clf = LogisticRegression(C = C_val, penalty=penalty_val, random_state=0)
        clf.fit(X, y)
        scores = clf.predict_proba(test_X)[:,1]
        return scores

def runRF(X, y, test_X, estimator_val=200, max_depth_val=5, min_samples_val = 10):
        clf = RandomForestClassifier(n_estimators=estimator_val, max_depth = max_depth_val, min_samples_split= min_samples_val, random_state=0)
        clf.fit(X, y)
        scores = clf.predict_proba(test_X)[:,1]
        return scores


if __name__ == "__main__":
        data_path = "/home/sudalai/Others/Kaggle/MMM15/Data/"
        train_file = data_path + "train_v4.csv"
        test_file = data_path + "test_v4.csv"
        sub_file = "sub8.csv"

        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        X = train_data.iloc[:,:-1]
        y = train_data['DV'].astype('int')
        test_X = test_data.iloc[:,1:]
        id_val = test_data['id']

        scores = runLogistic(X, y, test_X, C_val=1, penalty_val='l1')
        #scores = runRF(X, y, test_X, estimator_val=200, max_depth_val=6, min_samples_val = 200)

        #print X.shape
        #print y.shape

        sub_file_handle = open(sub_file, 'w')
        sub_file_handle.write('id,pred\n')
        for i in xrange(len(scores)):
                sub_file_handle.write(str(id_val[i])+','+ str(scores[i]) +'\n')
        sub_file_handle.close()
