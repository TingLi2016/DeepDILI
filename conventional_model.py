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
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


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

def sep_performance(df):
    cols = ['TN', 'FP', 'FN', 'TP', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'MCC']
    #cols = ['TN', 'FP', 'FN', 'TP', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'MCC']
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


tmp = pd.read_csv('/account/tli/CDER/data/org_data/QSAR_year_338_pearson_0.9.csv',low_memory=False)
cols = tmp.columns[5:]
data = tmp[['DILI_label','final_year', *cols]]
print(data.shape)


X,  y = data[data.final_year<1997].iloc[:,2:], data[data.final_year<1997]['DILI_label']
X_test, y_test = data[data.final_year>=1997].iloc[:,2:], data[data.final_year>=1997]['DILI_label']

print(X.shape)
print(X_test.shape)

###create directory
base_path = '/account/tli/CDER/results/sequence_feature_selection/conventional/conventional' + var

path30 = base_path + '/test_performance'
path3 = base_path + '/test_class'

###make the directory
os.mkdir(base_path)
os.mkdir(path30)
os.mkdir(path3)


#initial performance dictionary
test_results={}

#pred_val_df = pd.DataFrame()
pred_test_df = pd.DataFrame()


### scale the input
sc = MinMaxScaler()
sc.fit(X)
X = sc.transform(X)
X_test = sc.transform(X_test)

##KNN
###fit model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X, y)
###predict test results
knn_class, knn_result=model_predict(X_test, y_test, knn, 'knn')
test_results['knn']=knn_result
pred_test_df = pd.concat([pred_test_df, knn_class],axis=1, sort=False)

##LR
###fit model
lr = LogisticRegression(C=0.1, max_iter=300)
lr.fit(X, y)
###predict test results
lr_class, lr_result=model_predict(X_test, y_test, lr, 'lr')
test_results['lr']=lr_result
pred_test_df = pd.concat([pred_test_df, lr_class], axis=1, sort=False)

##SVM
###fit model
svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True,  random_state=1)
svm.fit(X, y)
###predict test results
svm_class, svm_result=model_predict(X_test, y_test, svm, 'svm')
test_results['svm']=svm_result
pred_test_df = pd.concat([pred_test_df, svm_class], axis=1, sort=False)

##RF
###fit model
rf = RandomForestClassifier(random_state=1, n_estimators=700, max_depth=11,  min_samples_leaf=5)
rf.fit(X, y)
###predict test results
rf_class, rf_result=model_predict(X_test, y_test, rf, 'rf')
test_results['rf']=rf_result
pred_test_df = pd.concat([pred_test_df, rf_class], axis=1, sort=False)

##XGBoost
###fit model
xgboost = XGBClassifier(learning_rate=0.01, n_estimators=700, max_depth=11, subsample=0.7)
xgboost.fit(X, y)
###predict test results
xgboost_class, xgboost_result=model_predict(X_test, y_test, xgboost, 'xgboost')
test_results['xgboost']=xgboost_result
pred_test_df = pd.concat([pred_test_df, xgboost_class], axis=1, sort=False)

col_name2 = 'conventional_'+ var

df = pd.DataFrame(data=test_results.items())
df = df.rename(columns={0:'name', 1:'value'})
df['name'] = df['name'].astype('str')
df['value'] = df['value'].astype('str')

df = sep_performance(df)


###save the result of validation results
pred_test_df.to_csv(path3+'/test_'+col_name2+'.csv')
df.to_csv(path30+'/test_'+col_name2+'.csv')

print("--- %s seconds ---" % (time.time() - start_time))      