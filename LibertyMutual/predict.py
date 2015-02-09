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

data_path = "C:/Sudalai/Others/Comp/Kaggle/LibertyMutual/Data/"

#train = pd.read_csv(data_path+'train.csv')
#test = pd.read_csv(data_path+'test.csv')

#train_var1 = pd.get_dummies(train['var1'])
#test_var1 = pd.get_dummies(test['var1'])

#train_var2 = pd.get_dummies(train['var2'])
#test_var2 = pd.get_dummies(test['var2'])

#train_var3 = pd.get_dummies(train['var3'])
#test_var3 = pd.get_dummies(test['var3'])

#train_var4 = pd.get_dummies(train['var4'])
#test_var4 = pd.get_dummies(test['var4'])

#train_var5 = pd.get_dummies(train['var5'])
#test_var5 = pd.get_dummies(test['var5'])

#train_var6 = pd.get_dummies(train['var6'])
#test_var6 = pd.get_dummies(test['var6'])

#train_var7 = pd.get_dummies(train['var7'])
#test_var7 = pd.get_dummies(test['var7'])

#train_var8 = pd.get_dummies(train['var8'])
#test_var8 = pd.get_dummies(test['var8'])

#train_var9 = pd.get_dummies(train['var9'])
#test_var9 = pd.get_dummies(test['var9'])

#train = np.hstack([train.iloc[:,11:19], train.iloc[:,20:], train_var1, train_var2, train_var3, train_var4, train_var5, train_var6, train_var7, train_var8, train_var9])
#test = np.hstack([test.iloc[:,10:18], test.iloc[:,19:], test_var1, test_var2, test_var3, test_var4, test_var5, test_var6, test_var7, test_var8, test_var9])

#train = np.nan_to_num(np.array(train)).astype('float64')
#test = np.nan_to_num(np.array(test)).astype('float64')
#print train.shape
#print test.shape

#np.save("train.npy", train)
#np.save("test.npy", test)
#print ts.shape

#np.save("train_y.npy", train['target'].values)

#sys.exit()
tr = np.load(data_path+"train.npy")
ts = np.load(data_path+"test.npy")
train_y = np.load(data_path+"train_y.npy")
sample = pd.read_csv(data_path+'sampleSubmission.csv')

print tr.shape
#print ts.shape

#tr = train[['var11', 'var12', 'var13', 'var14', 'var15', 'var16', 'var17']]
#ts = test[['var11', 'var12', 'var13', 'var14', 'var15', 'var16', 'var17']]

#tr = tr.iloc[:,2:]
#ts = ts.iloc[:,2:]

#for k in xrange(2,30):
#feature_selector = SelectKBest(score_func=f_regression, k=30)
#feature_selector.fit(tr, train_y)
#tr1 = feature_selector.transform(tr)
#ts1 = feature_selector.transform(ts)

#tr = tr[:,[288, 334,  50, 359,  29, 238,  45, 369, 188, 183, 225, 370, 310,  40,  63, 321, 226, 119,   2, 300, 291, 157, 303, 214,  46, 282, 349, 155,  32, 120, 100, 264, 382, 331, 180, 302, 295, 312, 372, 1, 335, 385, 387, 378, 338, 381, 6, 5, 0, 3]]
#ts = ts[:,[288, 334,  50, 359,  29, 238,  45, 369, 188, 183, 225, 370, 310,  40,  63, 321, 226, 119,   2, 300, 291, 157, 303, 214,  46, 282, 349, 155,  32, 120, 100, 264, 382, 331, 180, 302, 295, 312, 372, 1, 335, 385, 387, 378, 338, 381, 6, 5, 0, 3]]

tr2 = tr[:,[288, 334,  50, 359,  29, 238,  45, 369, 188, 183, 225, 370, 310,  40,  63, 321, 226, 119,   2, 300, 291, 157, 303, 214,  46, 282, 349, 155,  32, 120, 100, 264, 382, 331, 180, 302, 295, 312, 372, 1, 335, 385, 387, 378, 338, 381, 6, 5, 0, 3]]
ts2 = ts[:,[288, 334,  50, 359,  29, 238,  45, 369, 188, 183, 225, 370, 310,  40,  63, 321, 226, 119,   2, 300, 291, 157, 303, 214,  46, 282, 349, 155,  32, 120, 100, 264, 382, 331, 180, 302, 295, 312, 372, 1, 335, 385, 387, 378, 338, 381, 6, 5, 0, 3]]

tr3 = tr[:,[0,2,3,5,6,7,40,157,245,288,305,310,312,321,323,338,372,378]]
ts3 = ts[:,[0,2,3,5,6,7,40,157,245,288,305,310,312,321,323,338,372,378]]

#print tr3.shape
#print ts.shape

#train_y_cat = train_y[:]
#train_y_cat[train_y_cat>0] = 1

#tr = np.nan_to_num(np.array(tr))
#ts = np.nan_to_num(np.array(ts))

"""
print "Cross Validating.."
#clf = Ridge()
#train_y[train_y>0]=1
wt_gini = 0
#whole_cv_list = []
kf = KFold(tr.shape[0], n_folds=5)
#for i in xrange(tr.shape[1]):
for i in xrange(1):
    cv_gini_list=[]
    for dev_index, val_index in kf:
        #tr_new = tr[:,[0,2,3,5,6,7,40,157,245,288,305,310,312,321,323,338,372,378,i]]
        #tr_new = tr[:,[288, 334,  50, 359,  29, 238,  45, 369, 188, 183, 225, 370, 310,  40,  63, 321, 226, 119,   2, 300, 291, 157, 303, 214,  46, 282, 349, 155,  32, 120, 100, 264, 382, 331, 180, 302, 295, 312, 372, 1, 335, 385, 387, 378, 338, 381, 6, 5, 0, 3]]
        #X_dev, X_val = np.array([tr[dev_index,i]]).T, np.array([tr[val_index,i]]).T
        #X_dev, X_val = tr_new[dev_index,:], tr_new[val_index,:]
        #X_dev, X_val = tr1[dev_index,:], tr1[val_index,:]
        y_dev, y_val = train_y[dev_index], train_y[val_index]
        wt_dev, wt_val = tr[dev_index,1], tr[val_index,1]
        #print X_dev.shape
#for i in xrange(1):
        #clf = Ridge()
        #clf.fit(X_dev, y_dev)
        #preds1 = clf.predict(X_val)
        
        X_dev, X_val = tr2[dev_index,:], tr2[val_index,:]
        clf = Ridge()
        clf.fit(X_dev, y_dev)
        preds2 = clf.predict(X_val)
    
        X_dev, X_val = tr3[dev_index,:], tr3[val_index,:]
        clf = Ridge()
        clf.fit(X_dev, y_dev)
        preds3 = clf.predict(X_val)
        
        preds = (0.4*preds2)+ (0.6*preds3)
        cv_gini_list.append(normalized_weighted_gini(y_val,preds,wt_val))
        #clf = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split = 1000, min_samples_leaf=20, random_state=0)
        #cv_scores = cross_val_score(clf, tr[:,[3,5,6,7,40,53,161,251,312,335,338,372,378, i]], train_y, cv=5, scoring = "mean_squared_error")
        #cv_scores = cross_val_score(clf, tr, train_y, cv=5, scoring = "roc_auc")
    #print c_value 
    #print cv_scores
    #print np.mean(cv_scores)
    #if abs(np.mean(cv_scores)) < min_rms:
    #    min_rms = abs(np.mean(cv_scores))
    #    selected_index = i
    print cv_gini_list
    print np.mean(cv_gini_list)
    #whole_cv_list.append(np.mean(cv_gini_list))
    if np.mean(cv_gini_list) > wt_gini:
        wt_gini = np.mean(cv_gini_list)
        selected_index = i
    if i % 50 == 0:
        print "Processed : ",i
print wt_gini
print selected_index
"""

"""
kf = KFold(tr.shape[0], n_folds=5)
f1_cv_list = []
roc_cv_list = []
for dev_index, val_index in kf:
    X_dev, X_val = tr[dev_index,:], tr[val_index,:]
    y_dev, y_val = train_y[dev_index], train_y[val_index]
    y_dev_cat, y_val_cat = train_y_cat[dev_index], train_y_cat[val_index]
    #y_dev_cat = y_dev[:]
    #y_val_cat = y_val[:]
    #y_dev_cat[y_dev_cat>0]=1
    #y_val_cat[y_val_cat>0]=1
    #clf = Ridge()
    #clf = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split = 1000, min_samples_leaf=20, random_state=0)
    clf = LogisticRegression(penalty='l2', class_weight='auto')   
    #clf = SGDClassifier(loss='log', alpha=0.00001, n_iter=50)
    clf.fit(X_dev, y_dev_cat)
    pred_y_val = clf.predict_proba(X_val)[:,1]
    #f1_err = f1_score(y_val_cat, pred_y_val)
    #f1_cv_list.append(f1_err)
    #print "f1",f1_err
    roc_err = roc_auc_score(y_val_cat, pred_y_val)
    roc_cv_list.append(roc_err)
    print "roc", roc_err
print roc_cv_list    
print np.mean(roc_cv_list)
print f1_cv_list
print np.mean(f1_cv_list)
"""    
    
#clf = Ridge()
#clf.fit(tr1, train_y)
#preds1 = clf.predict(ts1)

clf = Ridge()
clf.fit(tr2, train_y)
preds2 = clf.predict(ts2)

clf = Ridge()
clf.fit(tr3, train_y)
preds3 = clf.predict(ts3)

preds = (0.4*preds2)+ (0.6*preds3)

##preds[preds<0] = 0
sample['target'] = preds
sample.to_csv('submission23.csv', index = False)
