import sys
import pandas as pd
import numpy as np
import operator
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn import ensemble
from sklearn.metrics import roc_auc_score,log_loss
import xgboost as xgb


def getCountVar(compute_df, count_df, var_name, count_var="v1"):
        grouped_df = count_df.groupby(var_name, as_index=False)[count_var].agg('count')
	grouped_df.columns = [var_name, "var_count"]

	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(-1, inplace=True)
	return list(merged_df["var_count"])	

def create_feature_map(features):
        outfile = open('xgb.fmap', 'w')
        for i, feat in enumerate(features):
                outfile.write('{0}\t{1}\tq\n'.format(i,feat))
        outfile.close()


def getDVEncodeVar(compute_df, target_df, var_name, target_var="target", min_cutoff=5):
        grouped_df = target_df.groupby(var_name, as_index=False)["target"].agg(["mean", "count"])
	grouped_df.columns = ["target_mean", "count_var"]
	grouped_df.reset_index(level=var_name, inplace=True)
	grouped_df["count_var"][grouped_df["count_var"]<min_cutoff] = 0
	grouped_df["count_var"][grouped_df["count_var"]>=min_cutoff] = 1
	grouped_df["target_mean"] = grouped_df["target_mean"] * grouped_df["count_var"]

	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(-1, inplace=True)
        return list(merged_df["target_mean"])


def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None):
	params = {}
	params["objective"] = "binary:logistic"
	params['eval_metric'] = 'logloss'
	params["eta"] = 0.02
	params["min_child_weight"] = 1 
	params["subsample"] = 0.85
	params["colsample_bytree"] = 0.3
	params["silent"] = 1
	params["max_depth"] = 10
	params["seed"] = 232345
	#params["gamma"] = 0.5
	num_rounds = 600

	plst = list(params.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	
	if test_y is not None:
	        xgtest = xgb.DMatrix(test_X, label=test_y)
	        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
	        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=1000)
	else:
		xgtest = xgb.DMatrix(test_X)
		model = xgb.train(plst, xgtrain, num_rounds)
	
	if feature_names:
                        create_feature_map(feature_names)
                        importance = model.get_fscore(fmap='xgb.fmap')
                        importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
                        imp_df = pd.DataFrame(importance, columns=['feature','fscore'])
                        imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()
                        imp_df.to_csv("imp_feat.txt", index=False)
	
	pred_test_y = model.predict(xgtest)
	
	if test_y is not None:
	        loss = log_loss(test_y, pred_test_y)
		print loss
	
		return pred_test_y, loss  
	else:
		return pred_test_y

def prepData(var4_col="v52"):
	import datetime
	start_time = datetime.datetime.now()
	print "Start time : ", start_time

	print "Reading files.."
	train = pd.read_csv('../Data/train.csv')
	test = pd.read_csv('../Data/test.csv')
	print train.shape, test.shape

	print "Filling NA.."
	train = train.fillna(-1)
	test = test.fillna(-1)

	print "Label encoding.."
	cat_columns = ["v129", "v72", "v62", "v38"]
	for f in train.columns:
		if train[f].dtype=='object':
			print(f), len(np.unique(train[f].values))
			#if f != 'v22':
			cat_columns.append(f)
	        	lbl = preprocessing.LabelEncoder()
	        	lbl.fit(list(train[f].values) + list(test[f].values))
	        	train[f] = lbl.transform(list(train[f].values))
	        	test[f] = lbl.transform(list(test[f].values))
			new_train = pd.concat([ train[['v1',f]], test[['v1',f]] ])
			train["CountVar_"+str(f)] = getCountVar(train[['v1',f]], new_train[['v1', f]], f)
                	test["CountVar_"+str(f)] = getCountVar(test[['v1',f]], new_train[['v1',f]], f)
	cat_columns_copy = cat_columns[:]


	print "Encoding train...."
	for f in cat_columns:
		print f
		val_list = np.zeros(train.shape[0])
		folds_array = np.array( pd.read_csv("./xfolds.csv")["fold5"] )
        	for fold_index in xrange(1,6):
                	dev_index = np.where(folds_array != fold_index)[0]
                	val_index = np.where(folds_array == fold_index)[0]
			new_train = train[["v1", f, "target"]]
			dev, val = new_train.iloc[dev_index,:], new_train.iloc[val_index,:]
			enc_list =  np.array( getDVEncodeVar(val[["v1", f]], dev[["v1", f, "target"]], f, min_cutoff=0 ))
			val_list[val_index] = enc_list
		train["DVEncode_"+str(f)] =  val_list


	print "Encoding test.."
	for f in cat_columns:
		print f
		test["DVEncode_"+str(f)] = getDVEncodeVar(test[["v1", f]], train[["v1", f, "target"]], f, min_cutoff=0)



	print "Two way encoding.."
	other_cols=[]
	new_var_list = []
	cat_columns = [col for col in cat_columns_copy]
	for ind, var1 in enumerate(cat_columns):
	                rem_cols = cat_columns[ind+1:]
	                #if var1 in "v30":
	                #       break
	                for var2 in rem_cols:
	                        print var1, var2
	                        new_var = var1+"_"+var2
	
				train[new_var] = train[var1].astype("str") +"_" + train[var2].astype("str") 
				test[new_var] = test[var1].astype("str") + "_" + test[var2].astype("str") 
				#print train[new_var][:10]
				#print test[new_var][:10]
	
				lbl = preprocessing.LabelEncoder()
	                        lbl.fit(list(train[new_var].values) + list(test[new_var].values))
	                        train[new_var] = lbl.transform(list(train[new_var].values))
	                        test[new_var] = lbl.transform(list(test[new_var].values))
	
	                        new_train = pd.concat([ train[['v1',new_var]], test[['v1',new_var]] ])
	                        test["Count_"+new_var] = getCountVar(test[['v1',new_var]], new_train[['v1', new_var]], new_var)
	                        train["Count_"+new_var] = getCountVar(train[['v1',new_var]], new_train[['v1', new_var]], new_var)
	                        new_var_list.append(new_var)

        print "Train.."
        for f in new_var_list:
                print f
                val_list = np.zeros(train.shape[0])
                folds_array = np.array( pd.read_csv("./xfolds.csv")["fold5"] )
                for fold_index in xrange(1,6):
                        dev_index = np.where(folds_array != fold_index)[0]
                        val_index = np.where(folds_array == fold_index)[0]
                        new_train = train[["v1", f, "target"]]
                        dev, val = new_train.iloc[dev_index,:], new_train.iloc[val_index,:]
                        enc_list =  np.array( getDVEncodeVar(val[["v1", f]], dev[["v1", f, "target"]], f, min_cutoff=0 )  )
                        val_list[val_index] = enc_list
                train["DVEncode_"+str(f)] =  val_list

        print "Test.."
        for f in new_var_list:
                print f
                test["DVEncode_"+str(f)] = getDVEncodeVar(test[["v1", f]], train[["v1", f, "target"]], f, min_cutoff=0)
	train = train.drop(new_var_list, axis=1)
	test = test.drop(new_var_list, axis=1)





	print "Three way encoding.."
	other_cols=[]
	new_var_list = []
	var3 = "v22"
	cat_columns = [col for col in cat_columns_copy if col!= var3]
	for ind, var1 in enumerate(cat_columns):
	                rem_cols = cat_columns[ind+1:]
	                #if var1 in "v30":
	                #       break
	                for var2 in rem_cols:
	                        print var1, var2
	                        new_var = var1+"_"+var2+"_"+var3
	
				train[new_var] = train[var1].astype("str") +"_" + train[var2].astype("str") +"_" + train[var3].astype("str")
				test[new_var] = test[var1].astype("str") + "_" + test[var2].astype("str") + "_" + test[var3].astype("str")
				#print train[new_var][:10]
				#print test[new_var][:10]
	
				lbl = preprocessing.LabelEncoder()
	                        lbl.fit(list(train[new_var].values) + list(test[new_var].values))
	                        train[new_var] = lbl.transform(list(train[new_var].values))
	                        test[new_var] = lbl.transform(list(test[new_var].values))
	
	                        new_train = pd.concat([ train[['v1',new_var]], test[['v1',new_var]] ])
	                        test["Count_"+new_var] = getCountVar(test[['v1',new_var]], new_train[['v1', new_var]], new_var)
	                        train["Count_"+new_var] = getCountVar(train[['v1',new_var]], new_train[['v1', new_var]], new_var)
	                        new_var_list.append(new_var)

        print "Train.."
        for f in new_var_list:
                print f
                val_list = np.zeros(train.shape[0])
                folds_array = np.array( pd.read_csv("./xfolds.csv")["fold5"] )
                for fold_index in xrange(1,6):
                        dev_index = np.where(folds_array != fold_index)[0]
                        val_index = np.where(folds_array == fold_index)[0]
                        new_train = train[["v1", f, "target"]]
                        dev, val = new_train.iloc[dev_index,:], new_train.iloc[val_index,:]
                        enc_list =  np.array( getDVEncodeVar(val[["v1", f]], dev[["v1", f, "target"]], f, min_cutoff=0)  )
                        val_list[val_index] = enc_list
                train["DVEncode_"+str(f)] =  val_list

        print "Test.."
        for f in new_var_list:
                print f
                test["DVEncode_"+str(f)] = getDVEncodeVar(test[["v1", f]], train[["v1", f, "target"]], f, min_cutoff=0)
	train = train.drop(new_var_list, axis=1)
	test = test.drop(new_var_list, axis=1)





	
	print "Four way encoding.."
	other_cols=[]
	new_var_list = []
	for var4_col in ["v52", "v66", "v24", "v56", "v125", "v30"]:
	        var1_cols = ["v22"]
		var4 = var4_col
		other_cols.append(var4)
	        cat_columns = [col for col in cat_columns_copy if col not in var1_cols if col not in other_cols]
	        for var1 in var1_cols:
		    for ind, var2 in enumerate(cat_columns):
	                rem_cols = cat_columns[ind+1:]
	                #if var1 in "v30":
	                #       break
	                for var3 in rem_cols:
	                        print var1, var4, var2, var3
	                        new_var = var1+"_"+var4+"_"+var2+"_"+var3
	
				train[new_var] = train[var1].astype("str") +"_" + train[var2].astype("str") + "_"+ train[var3].astype("str") + "_" +train[var4].astype("str")
				test[new_var] = test[var1].astype("str") + "_" + test[var2].astype("str") + "_" +test[var3].astype("str") + "_" + test[var4].astype("str")
				#print train[new_var][:10]
				#print test[new_var][:10]
	
				lbl = preprocessing.LabelEncoder()
	                        lbl.fit(list(train[new_var].values) + list(test[new_var].values))
	                        train[new_var] = lbl.transform(list(train[new_var].values))
	                        test[new_var] = lbl.transform(list(test[new_var].values))
	
	                        new_train = pd.concat([ train[['v1',new_var]], test[['v1',new_var]] ])
	                        test["Count_"+new_var] = getCountVar(test[['v1',new_var]], new_train[['v1', new_var]], new_var)
	                        train["Count_"+new_var] = getCountVar(train[['v1',new_var]], new_train[['v1', new_var]], new_var)
	                        new_var_list.append(new_var)

        print "Train.."
        for f in new_var_list:
                print f
                val_list = np.zeros(train.shape[0])
                folds_array = np.array( pd.read_csv("./xfolds.csv")["fold5"] )
                for fold_index in xrange(1,6):
                        dev_index = np.where(folds_array != fold_index)[0]
                        val_index = np.where(folds_array == fold_index)[0]
                        new_train = train[["v1", f, "target"]]
                        dev, val = new_train.iloc[dev_index,:], new_train.iloc[val_index,:]
                        enc_list =  np.array( getDVEncodeVar(val[["v1", f]], dev[["v1", f, "target"]], f, min_cutoff=2)  )
                        val_list[val_index] = enc_list
                train["DVEncode_"+str(f)] =  val_list

        print "Test.."
        for f in new_var_list:
                print f
                test["DVEncode_"+str(f)] = getDVEncodeVar(test[["v1", f]], train[["v1", f, "target"]], f, min_cutoff=2)
	train = train.drop(new_var_list, axis=1)
	test = test.drop(new_var_list, axis=1)


	print "Five way encoding.."
        new_var_list = []
        for var4_col, var5_col in [["v52", "v66"], ["v24", "v56"], ["v125", "v30"], ["v52","v56"], ["v71", "v91"], ["v112","v113"]]:
                var1_cols = ["v22"]
                var4 = var4_col
                var5 = var5_col
                cat_columns = [col for col in cat_columns_copy if col not in var1_cols if col != var4 if col!=var5]
                for var1 in var1_cols:
                    for ind, var2 in enumerate(cat_columns):
                        rem_cols = cat_columns[ind+1:]
                        #if var1 in "v30":
                        #       break
                        for var3 in rem_cols:
                                print var1, var4, var5, var2, var3
                                new_var = var1+"_"+var4+"_"+var5+"_"+var2+"_"+var3

                                train[new_var] = train[var1].astype("str") + "_"+ train[var2].astype("str") + "_"+train[var3].astype("str") + "_"+ train[var4].astype("str") + "_"+ train[var5].astype("str")
                                test[new_var] = test[var1].astype("str") + "_"+ test[var2].astype("str") + "_"+ test[var3].astype("str") + "_"+ test[var4].astype("str") + "_"+ test[var5].astype("str")

                                lbl = preprocessing.LabelEncoder()
                                lbl.fit(list(train[new_var].values) + list(test[new_var].values))
                                train[new_var] = lbl.transform(list(train[new_var].values))
                                test[new_var] = lbl.transform(list(test[new_var].values))

                                new_train = pd.concat([ train[['v1',new_var]], test[['v1',new_var]] ])
                                test["Count_"+new_var] = getCountVar(test[['v1',new_var]], new_train[['v1', new_var]], new_var)
                                train["Count_"+new_var] = getCountVar(train[['v1',new_var]], new_train[['v1', new_var]], new_var)
                                new_var_list.append(new_var)

                print "Train.."
                for f in new_var_list:
                        print f
                        val_list = np.zeros(train.shape[0])
                        folds_array = np.array( pd.read_csv("./xfolds.csv")["fold5"] )
                        for fold_index in xrange(1,6):
                                dev_index = np.where(folds_array != fold_index)[0]
                                val_index = np.where(folds_array == fold_index)[0]
                                new_train = train[["v1", f, "target"]]
                                dev, val = new_train.iloc[dev_index,:], new_train.iloc[val_index,:]
                                enc_list =  np.array( getDVEncodeVar(val[["v1", f]], dev[["v1", f, "target"]], f, min_cutoff=2)  )
                                val_list[val_index] = enc_list
                        train["DVEncode_"+str(f)] =  val_list

                print "Test.."
                for f in new_var_list:
                        print f
                        test["DVEncode_"+str(f)] = getDVEncodeVar(test[["v1", f]], train[["v1", f, "target"]], f, min_cutoff=2)




	

	train = train.drop(new_var_list, axis=1)
	test = test.drop(new_var_list, axis=1)
	train.to_csv("train_5levelenc_withint.csv", index=False)
	test.to_csv("test_5levelenc_withint.csv", index=False)
		
	end_time = datetime.datetime.now()
	print "End time : ",end_time

	print end_time - start_time

def runET(train_X, train_y, test_X, test_y=None, validation=1, n_est_val=50, depth_val=None, split_val=2, leaf_val=1, feat_val='auto', jobs_val=4, random_state_val=0):
        clf = ensemble.ExtraTreesClassifier(
                n_estimators = n_est_val,
                max_depth = depth_val,
                min_samples_split = split_val,
                min_samples_leaf = leaf_val,
                max_features = feat_val,
                criterion='entropy',
                n_jobs = jobs_val,
                random_state = random_state_val)
        clf.fit(train_X, train_y)
        pred_train_y = clf.predict_proba(train_X)[:,1]
        pred_test_y = clf.predict_proba(test_X)[:,1]

        if validation:
                train_loss = log_loss(train_y, pred_train_y)
                loss = log_loss(test_y, pred_test_y)
                print "Train, Test loss : ", train_loss, loss
                return pred_test_y, loss
        else:
                return pred_test_y


def prepModel(var4_col="v52"):
        print "Reading files.."
        train = pd.read_csv('./train_5levelenc_withint.csv')
        test = pd.read_csv('./test_5levelenc_withint.csv')
        print train.shape, test.shape

        print "Getting DV and ID.."
        train_y = train.target.values
        train_ID = train.ID.values
        test_ID = test.ID.values
        train = train.drop(['ID', "target"], axis=1)
        test = test.drop(['ID'], axis=1)

        print "Filling NA.."
        train = train.fillna(-1)
        test = test.fillna(-1)

	feat_names = list(train.columns)
        print "Converting to array.."
        train = np.array(train)
        test = np.array(test)
        print train.shape, test.shape

	assert train.shape[1] == test.shape[1]
	print "Cross validating.."
        cv_scores = []
        train_preds = np.zeros(train.shape[0])
        folds_array = np.array( pd.read_csv("./xfolds.csv")["fold5"] )
        for fold_index in xrange(1,6):
                dev_index = np.where(folds_array != fold_index)[0]
                val_index = np.where(folds_array == fold_index)[0]
                dev_X, val_X = train[dev_index,:], train[val_index,:]
                dev_y, val_y = train_y[dev_index], train_y[val_index]

                #preds, loss = runXGB(dev_X, dev_y, val_X, val_y, feature_names=feat_names)
                #for feat in [60, 100, 150]:
                preds, loss = runET(dev_X, dev_y, val_X, val_y, validation=1, n_est_val=500, depth_val=40, split_val=4, leaf_val=2, feat_val=180, jobs_val=4, random_state_val=98751)
                #print feat, loss
                cv_scores.append(loss)
                print cv_scores
                train_preds[val_index] = preds
        print cv_scores, np.mean(cv_scores)

        out_df = pd.DataFrame({"ID":train_ID})
        out_df["et1_srk_5levelenc_withint"] = train_preds
        out_df.to_csv("prval_et1_srk_5levelenc_withint.csv", index=False)

	print "Final model.."
	preds = runET(train, train_y, test, validation=0, n_est_val=500, depth_val=40, split_val=4, leaf_val=2, feat_val=180, jobs_val=4, random_state_val=98751)
	out_df = pd.DataFrame({"ID":test_ID})
        out_df["et1_srk_5levelenc_withint"] = preds
        out_df.to_csv("prfull_et1_srk_5levelenc_withint.csv", index=False)



def prepModelXGB(var4_col="v52"):
        print "Reading files.."
        train = pd.read_csv('./train_5levelenc_withint.csv')
	print train.shape

        print "Getting DV and ID.."
        train_y = train.target.values
        train_ID = train.ID.values
        train = train.drop(['ID', "target"], axis=1)

        print "Filling NA.."
        train = train.fillna(-1)

	feat_names = list(train.columns)
        print "Converting to array.."
        train = np.array(train)
	print train.shape

	
	print "Cross validating.."
        cv_scores = []
        train_preds = np.zeros(train.shape[0])
        folds_array = np.array( pd.read_csv("./xfolds.csv")["fold5"] )
        for fold_index in xrange(1,6):
                dev_index = np.where(folds_array != fold_index)[0]
                val_index = np.where(folds_array == fold_index)[0]
                dev_X, val_X = train[dev_index,:], train[val_index,:]
                dev_y, val_y = train_y[dev_index], train_y[val_index]

                preds, loss = runXGB(dev_X, dev_y, val_X, val_y, feature_names=feat_names)
                #for feat in [60, 100, 150]:
                #preds, loss = runET(dev_X, dev_y, val_X, val_y, validation=1, n_est_val=600, depth_val=40, split_val=4, leaf_val=2, feat_val=180, jobs_val=4, random_state_val=8111)
                #print feat, loss
                cv_scores.append(loss)
                print cv_scores
                train_preds[val_index] = preds
        print cv_scores, np.mean(cv_scores)

        out_df = pd.DataFrame({"ID":train_ID})
        out_df["xg1_srk_5levelenc_withint"] = train_preds
        out_df.to_csv("prval_xg1_srk_5levelenc_withint.csv", index=False)

	import gc
	del dev_X
	del val_X
	gc.collect()


	print "Final model.."
        test = pd.read_csv('./test_5levelenc_withint.csv')
        print train.shape, test.shape
        test_ID = test.ID.values
        test = test.drop(['ID'], axis=1)
        test = test.fillna(-1)
        test = np.array(test)
        print train.shape, test.shape

	assert train.shape[1] == test.shape[1]
	#preds = runET(train, train_y, test, validation=0, n_est_val=600, depth_val=40, split_val=4, leaf_val=2, feat_val=180, jobs_val=4, random_state_val=8111)
	preds = runXGB(train, train_y, test, feature_names=feat_names)
	out_df = pd.DataFrame({"ID":test_ID})
        out_df["xg1_srk_5levelenc_withint"] = preds
        out_df.to_csv("prfull_xg1_srk_5levelenc_withint.csv", index=False)


if __name__ == "__main__":
	#for var4_col_name in ["v52", "v66", "v24", "v56", "v125", "v30"]:
	for var4_col_name in ["v30"]:
		try:
			prepData(var4_col_name)
			prepModelXGB(var4_col_name)
			prepModel(var4_col_name)
		except Exception,e:
			print e
			pass
