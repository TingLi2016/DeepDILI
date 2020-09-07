#!/account/tli/anaconda3/bin/python

import sys
var=sys.argv[1]

import time
start_time = time.time()

###Loading packages
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import math
import keras
import itertools

from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

import tensorflow as tf
from keras import optimizers
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2
from keras.layers import Dense, Activation,  Dropout, BatchNormalization
from keras.models import Sequential, load_model
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(6)

he_normal = initializers.he_normal()

# define base model
def create_model(layer, n_dim, node, dropout, activation, optimizer):

    # create model
    NN_model = Sequential()

    # The Input Layer :
    NN_model.add(Dense(node, input_dim = n_dim, kernel_initializer=he_normal,  activation=activation, kernel_regularizer=l2(0.001), activity_regularizer=l2(0.001)))
    NN_model.add(BatchNormalization())
    NN_model.add(Dropout(dropout))


    # The Hidden Layers :
    layer -= 1
    while (layer > 0):
        NN_model.add(Dense(node, kernel_initializer=he_normal, activation=activation))
        NN_model.add(BatchNormalization())
        NN_model.add(Dropout(dropout))
        layer -= 1


    # The Output Layer :
    NN_model.add(Dense(1, activation='sigmoid'))

    # Compile model
    NN_model.model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    return NN_model

###fit model
def fit_model(X_train, y_train, X_validation, y_validation, n, model_path, model, batch_size):
    ###balanced class weight
    class_weights = {0:0.5,1:0.5}
    es = EarlyStopping(monitor='acc', mode='max', verbose=1, patience=5)
    ###define checkpoint for the best model
    checkpoint = ModelCheckpoint(model_path, verbose=1, monitor='acc',save_best_only=True, mode='max')
    ###fit model
    model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=n, batch_size=batch_size, class_weight=class_weights, callbacks=[checkpoint, es])
    best_model = load_model(model_path)
    return best_model

def model_predict(X, y, model, col_name):
    y_pred = model.predict(X)
    y_pred_class = np.where(y_pred > 0.5, 1, 0)
    pred_result = pd.DataFrame()
    pred_result['id'] = y.index
    pred_result['y_true'] = y.values
    pred_result['prob_'+col_name] = y_pred
    pred_result['class_'+col_name] = y_pred_class

    TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc=measurements(y, y_pred_class, y_pred)
    return pred_result, [TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc]

def measurements(y_test, y_pred, y_pred_prob):
    acc = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    mcc = metrics.matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test.ravel(), y_pred_prob.ravel()) 
    sensitivity = metrics.recall_score(y_test, y_pred)
    
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    specificity = TN/(TN+FP)
    npv = TN/(TN+FN+1.e-10)
    return TN, FP, FN, TP, acc, auc, sensitivity, specificity, precision, npv, f1, mcc

def sep_performance(df):
    cols = ['TN', 'FP', 'FN', 'TP', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'MCC']
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
base_path = '/account/tli/CDER/results/sequence_feature_selection/dnn/dnn' + var

path30 = base_path + '/test_performance'

path3 = base_path + '/test_class'
path4 = base_path + '/weights'

###make the directory
os.mkdir(base_path)
os.mkdir(path30)

os.mkdir(path3)
os.mkdir(path4)

#initial performance dictionary
test_results={}

pred_test_df = pd.DataFrame()


### scale the input
sc = MinMaxScaler()
sc.fit(X)
X = sc.transform(X)
X_test = sc.transform(X_test)

###parameters
col_name2 = 'dnn_'+var
model_path = path4 + '/' +col_name2 + '_weights.h5'

optimizer = optimizers.Adam(lr=para[2])
activation = para[5]
print(activation)

###create and fit model
model = create_model(4, X.shape[1], 128, 0.2, 'tanh', optimizers.Adam(lr=0.0001))
best_model = fit_model(X, y, X_test, y_test, 100, model_path, model, para[3])


### predict test results
test_class, test_result=model_predict(X_test, y_test, best_model, 'dnn')
test_results['dnn']=test_result


test_class.to_csv(path3+'/test_'+col_name2+'.csv')

df = pd.DataFrame(data=test_results.items())
df = df.rename(columns={0:'name', 1:'value'})
df['name'] = df['name'].astype('str')
df['value'] = df['value'].astype('str')

df = sep_performance(df)
df.to_csv(path30+'/test_'+col_name2+'.csv')

print("--- %s seconds ---" % (time.time() - start_time))   