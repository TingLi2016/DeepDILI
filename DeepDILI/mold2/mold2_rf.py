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


tmp0 = pd.read_csv('/account/tli/CDER/data/org_data/important_features_order.csv')
features = tmp0.feature.unique()

tmp = pd.read_csv('/account/tli/CDER/data/org_data/QSAR_year_338_pearson_0.9.csv',low_memory=False)
print("original tmp: ", tmp.shape) 
#tmp = tmp.drop(tmp.columns[int(var)], axis=1) 
tmp = tmp.drop([*features[:135]], axis=1)
print("changed tmp: ", tmp.shape) 
cols = tmp.columns[5:]
data = tmp[['DILI_label','final_year', *cols]]
print(data.shape)



X_org,  y_org = data[data.final_year<1997].iloc[:,2:], data[data.final_year<1997]['DILI_label']
X, X_val, y, y_val = train_test_split(X_org,  y_org, test_size=0.2, stratify=y_org, random_state=7)

X_test, y_test = data[data.final_year>=1997].iloc[:,2:], data[data.final_year>=1997]['DILI_label']

###external dataset
#external = pd.read_csv('/account/tli/CDER/data/external_dataset/external_240_mold2.csv')
#X_test, y_test = external[cols], external['DILI_label']

###drugbank dataset
#external = pd.read_csv('/account/tli/CDER/data/external_dataset/drugbank_without_training.csv')
#set a fake DILI label for running the script
#external['DILI_label'] = np.where(external.index < 4000, 0, 1)
#X_test, y_test = external[cols], external['DILI_label']

###covid dataset
#external = pd.read_csv('/account/tli/CDER/data/external_dataset/covid_without_training.csv')
#set a fake DILI label for running the script
#external['DILI_label'] = np.where(external.index < 7, 0, 1)
#X_test, y_test = external[cols], external['DILI_label']

print(X_org.shape)
print(X.shape)
print(X_test.shape)


base_path = '/account/tli/CDER/results/check/rf/rf_' + var

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
train_results={}
validation_results={}
test_results={}

pred_val_df = pd.DataFrame()
pred_test_df = pd.DataFrame()


for i in range(20):
    #pred_df = pd.DataFrame()
    skf = StratifiedKFold(n_splits=5, random_state=i, shuffle=True)
    j = 0
    for train_index, validation_index in skf.split(X, y):
        ###get train, validation dataset
        X_train, X_validation = X.iloc[train_index,:], X.iloc[validation_index,:]
        y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]
        
        ### scale the input
        sc = MinMaxScaler()
        #sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_validation = sc.transform(X_validation)
        X_val_s = sc.transform(X_val)
        X_test_s = sc.transform(X_test)

        ### define column name 
        col_name = 'rf_'+'seed_'+str(i)+'_skf_'+str(j)+'_paras_'+var
        col_name1 = 'rf_'+'seed_'+str(i)+'_paras_'+var
        col_name2 = 'rf_'+'_paras_'+var


        ### create and fit model
        clf = RandomForestClassifier(random_state=1, n_estimators=700, max_depth=11,  min_samples_leaf=5, class_weight='balanced', bootstrap = True, max_features='log2')
        clf.fit(X_train, y_train)

        ### predict validation results
        train_class, train_result=model_predict(X_validation, y_validation, clf, col_name)
        train_results[col_name]=train_result
        
        ### predict validation results
        validation_class, validation_result=model_predict(X_val_s, y_val, clf, col_name)
        validation_results[col_name]=validation_result

        ### predict test results
        test_class, test_result=model_predict(X_test_s, y_test, clf, col_name)
        test_results[col_name]=test_result
        
        pred_val_df = pd.concat([pred_val_df, validation_class], axis=1, sort=False)
        pred_test_df = pd.concat([pred_test_df, test_class],axis=1, sort=False)
        j += 1
        train_class.to_csv(path1+'/train_'+col_name+'.csv')

###save the result of validation results
pd.DataFrame(data=train_results.items()).to_csv(path10+'/train_'+col_name2+'.csv')
pred_val_df.to_csv(path2+'/validation_'+col_name2+'.csv')
pd.DataFrame(data=validation_results.items()).to_csv(path20+'/validation_'+col_name2+'.csv')
pred_test_df.to_csv(path3+'/test_'+col_name2+'.csv')
pd.DataFrame(data=test_results.items()).to_csv(path30+'/test_'+col_name2+'.csv')

print("--- %s seconds ---" % (time.time() - start_time))           