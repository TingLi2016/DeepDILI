#!/account/tli/anaconda3/bin/python


import time
start_time = time.time()

import sys
var=sys.argv[1]

###Loading packages
import warnings
warnings.filterwarnings('ignore')

import os
from os import listdir
from os.path import isfile, join
import itertools
import math
import keras
import pandas as pd
import numpy as np


from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score,balanced_accuracy_score


from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf
from keras import optimizers
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(6)

he_normal = initializers.he_normal()


def model_predict(X, y, model, col_name):
    y_pred = model.predict(X)
    y_pred_class = np.where(y_pred > 0.5, 1, 0)
    pred_result = pd.DataFrame()
    pred_result['id'] = y.index
    pred_result['y_true'] = y.values
    pred_result['prob_'+col_name] = y_pred
    pred_result['class_'+col_name] = y_pred_class

    result=measurements(y, y_pred_class, y_pred)
    return pred_result, result

def measurements(y_test, y_pred, y_pred_prob):
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob) 
    sensitivity = metrics.recall_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    specificity = TN/(TN+FP)
    npv = TN/(TN+FN)
    return [TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc, balanced_accuracy]
  

def dim_reduce(df, test_df, col_name1):
    
    X = df.iloc[:, 3:]
    y = df.loc[:, 'y_true']
    X_test = test_df.iloc[:, 3:]
    y_test = test_df.loc[:, 'y_true']

    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    X_test = sc.transform(X_test)    
    
    ###parameters
    model_path = '/account/tli/CDER/script/train_validation_test/fp_extra/maccs/maccs_best_model.h5'

    ###load model
    best_model = load_model(model_path)
    
    ### predict test set
    test_class, test_result = model_predict(X_test, y_test, best_model, col_name1)
    train_class, train_result= model_predict(X, y, best_model, col_name1)

    return test_class, test_result, train_class, train_result 

def sep_performance(df):
    cols = ['TN', 'FP', 'FN', 'TP', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'MCC', 'Banlanced_accuracy']
    for i, col in enumerate(cols):
        if i == 0:
            df[col] = df.value.str.split(',').str[i].str.split('[').str[1].values
        elif i == len(cols)-1:
            df[col] = df.value.str.split(',').str[i].str.split(']').str[0].values
        else:
            df[col] = df.value.str.split(',').str[i].values

    for i, col in enumerate(cols):
        if i < 4:
            df[col] = df[col].astype(int)
        else:
            df[col] = df[col].astype(float)
            df[col] = round(df[col], 3)
    del df['value']
            
    return df

def reform_result(results):
    df = pd.DataFrame(data=results.items())
    ###reform the result data format into single colum
    df = df.rename(columns={0:'name', 1:'value'})
    df['name'] = df['name'].astype('str')
    df['value'] = df['value'].astype('str')
    df = sep_performance(df)
    return df


###data
path='/account/tli/CDER/results/check/data'
data = pd.read_csv(path+'/validation/validation_probabilities_'+var+'.csv')
test = pd.read_csv(path+'/test/test_probabilities_'+var+'.csv')
print('data: ' ,data.shape)
print('test: ' ,test.shape)


base_path='/account/tli/CDER/results/check/result'
path1 = base_path + '/weights'
path2 = base_path + '/validation_class'
path3 = base_path + '/validation_performance'
path4 = base_path + '/test_class'
path5 = base_path + '/test_performance'


#initial performance dictionary
test_results={}
train_results={}

col_name2 = 'feature_' + var
###get the prediction
test_class, test_result, train_class, train_result  = dim_reduce(data, test, col_name2)

test_results[col_name2]=test_result
test_class.to_csv(path4+'/test_'+col_name2+'.csv')

train_results[col_name2]=train_result
train_class.to_csv(path2+'/validation_'+col_name2+'.csv')

reform_result(test_results).to_csv(path5+'/dnn_test_para_'+ var +'.csv')
reform_result(train_results).to_csv(path3+'/dnn_validation_para_'+ var+'.csv')

K.clear_session()
tf.reset_default_graph() 

print("--- %s seconds ---" % (time.time() - start_time))
