import sys
import csv
import operator
import numpy as np
import pandas as pd
import scipy as sp
import cPickle as pkl
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn import ensemble, preprocessing
from sklearn.metrics import mean_squared_error, roc_auc_score
#sys.path.append("/home/sudalai/Softwares/xgboost-master/wrapper/")
sys.path.append("/home/sudalai/Softwares/XGB_pointfour/xgboost-master/wrapper/")
import xgboost as xgb

np.random.seed(12345)
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Dropout
from keras.constraints import maxnorm
from keras.utils import np_utils
from keras import regularizers
from keras.layers.advanced_activations import PReLU


def multiclassLogLoss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss


def runNN(train_X, train_y, test_X=None, test_y=None):
        sc = preprocessing.StandardScaler()
        train_X = sc.fit_transform(train_X)
        #test_X = sc.transform(test_X)

        train_y = np_utils.to_categorical(train_y, 38)

        model = Sequential()
	#model.add(Dropout(0.2))

        model.add(Dense(600, input_shape=(train_X.shape[1],), init='he_uniform', W_regularizer=regularizers.l1(0.002)))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
	#model.add(BatchNormalization())

        model.add(Dense(600, init='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
	#model.add(BatchNormalization())

        #model.add(Dense(100, init='he_uniform'))
        #model.add(Activation('relu'))
        #model.add(Dropout(0.5))

        model.add(Dense(38, init='he_uniform'))
        model.add(Activation('softmax'))

        #sgd_opt = SGD(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer='adagrad')

	#for i in xrange(500):
        model.fit(train_X, train_y, batch_size=256, nb_epoch=200, validation_split=0.03, verbose=2, shuffle=True)
	#preds = model.predict(test_X, verbose=0)
        #print "Test preds shape : ",preds.shape
	#loss = multiclassLogLoss(test_y, preds)
	#print "At",(i+1)*2, "Epochs, Loss is : ", loss
        #print "ROC AUC score : ", metrics.roc_auc_score(test_y, preds)

	return model, sc

if __name__ == "__main__":
        # setting the input path and reading the data into dataframe #
        print "Reading data.."
        data_path = "../Data/"
	train_X = pd.read_csv(data_path + "train_mod_v2.csv")

	print "Getting target and id"
        train_y = np.array(train_X["DV"])
        train_id = np.array(train_X["VisitNumber"])
	
	print "Dropping columns"
        drop_columns = ["DV"]
        train_X.drop(drop_columns+["VisitNumber"], axis=1, inplace=True)
	#test_X.drop(["VisitNumber"], axis=1, inplace=True)

	print "Converting to array"
        train_X = np.array(train_X)
	print "Train shape : ", train_X.shape 

	print "Building model.."
	model, scaler = runNN(train_X, train_y)
	del train_X
	import gc
	gc.collect()

	print "Working on test data.."
	test_X = pd.read_csv(data_path + "test_mod_v2.csv")
	test_id = np.array(test_X["VisitNumber"])
	test_X.drop(["VisitNumber"], axis=1, inplace=True)
	test_X = np.array(test_X)
	test_X = scaler.transform(test_X)

	print "Getting preds.."
	preds = model.predict(test_X, verbose=0)
	
	sample = pd.read_csv(data_path + "sample_submission.csv")
        preds = pd.DataFrame(preds, index=test_id, columns=sample.columns[1:])
        preds.to_csv("sub_nn.csv", index_label="VisitNumber")
