#!/account/tli/anaconda3/bin/python

import sys
var=sys.argv[1]


import time
start_time = time.time()


###Loading packages
import os
import numpy as np
import pandas as pd
import math
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import class_weight
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, roc_curve

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from numpy.random import seed
seed(1)


import itertools


def measurements(y_test, y_pred, y_pred_prob):  
    acc = metrics.accuracy_score(y_test, y_pred)
    sensitivity = metrics.recall_score(y_test, y_pred)
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    specificity = TN/(TN+FP)
    precision = metrics.precision_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    npv = TN/(TN+FN)       
    return [TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc]

def model_predict(X, y, model, col_name):
    y_pred_prob = model.predict_proba(X)
    # keep probabilities for the positive outcome only
    y_pred_prob = y_pred_prob[:, 1]
    y_pred_class = np.where(y_pred_prob > 0.5, 1, 0)

    ###create dataframe
    pred_result = pd.DataFrame()
    pred_result['id'] = y.index
    pred_result['y_true'] = y.values
    pred_result['prob_'+col_name] = y_pred_prob
    pred_result['class_'+col_name] = y_pred_class
    
    performance =measurements(y, y_pred_class, y_pred_prob)

    return pred_result, performance


###import trainind data index (which was used in the training)
train_index = pd.read_csv('/account/tli/CDER/data/fps_extra/train_index.csv')

data = pd.read_csv('/account/tli/CDER/data/fps_extra/dili_maccs.csv',low_memory=False)

X, y = data[data.Usage == 'training_set_1'].iloc[:, 7:], data[data.Usage == 'training_set_1']['DILI_label']
X_val, y_val = data[data.Usage == 'training_set_2'].iloc[:, 7:], data[data.Usage == 'training_set_2']['DILI_label']
#X_test, y_test = data[data.Usage == 'test'].iloc[:, 7:], data[data.Usage == 'test']['DILI_label']

###external dataset
external = pd.read_csv('/account/tli/CDER/data/external_dataset/external_maccs.csv')
cols = X.columns
X_test, y_test = external[cols], external['DILI_label']


print(data.shape)
print(X.shape)
print(X_val.shape)
print(X_test.shape)

base_path = '/account/tli/CDER/results/check/knn/knn_' + var

path10 = base_path + '/training_performance'
path20 = base_path + '/validation_performance'
path30 = base_path + '/test_performance'

path1 = base_path + '/training_class'
path2 = base_path + '/validation_class'
path3 = base_path + '/test_class'

###make the directory
os.mkdir(base_path)
os.mkdir(path10)
os.mkdir(path20)
os.mkdir(path30)

os.mkdir(path1)
os.mkdir(path2)
os.mkdir(path3)



#initial performance dictionary
validation_results={}
test_results={}

pred_val_df = pd.DataFrame()
pred_test_df = pd.DataFrame()

for col in train_index.columns[9:].values:
    print(col)
    ###get train dataset
    train_sample_index = train_index[train_index[col] == 1].index.unique()
    X_train = X.iloc[train_sample_index, :]
    y_train = y.iloc[train_sample_index]

    X_val_s = X_val
    X_test_s = X_test

    ### define column name
    col_name = 'knn_'+'seed_'+col+'_paras_'+var+'_K_'+str(11)
    col_name2 = 'knn_'+'paras_'+var

    ###create classifier
    clf = KNeighborsClassifier(n_neighbors=11)
    clf.fit(X_train, y_train)


    ### predict validation results
    validation_class, validation_result=model_predict(X_val_s, y_val, clf, col_name)
    validation_results[col_name]=validation_result

    ### predict test results
    test_class, test_result=model_predict(X_test_s, y_test, clf, col_name)
    test_results[col_name]=test_result

    pred_val_df = pd.concat([pred_val_df, validation_class], axis=1, sort=False)
    pred_test_df = pd.concat([pred_test_df, test_class],axis=1, sort=False)


###save the result of validation results
pred_val_df.to_csv(path2+'/validation_'+col_name2+'.csv')
pd.DataFrame(data=validation_results.items()).to_csv(path20+'/validation_'+col_name2+'.csv')
pred_test_df.to_csv(path3+'/test_'+col_name2+'.csv')
pd.DataFrame(data=test_results.items()).to_csv(path30+'/test_'+col_name2+'.csv')

print("--- %s seconds ---" % (time.time() - start_time))           