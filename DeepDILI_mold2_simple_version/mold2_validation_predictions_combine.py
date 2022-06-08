import pandas as pd
import numpy as np


import os
from os import listdir
from os.path import isfile, join
import itertools
from functools import reduce

def combine_validation_probabilities(base_path, mcc, probability_path, var):
    
    ### the path for five different base classifiers
    knn_base_path = base_path + '/knn/validation_class' 
    lr_base_path = base_path +  '/lr/validation_class'
    svm_base_path = base_path + '/svm/validation_class'
    rf_base_path = base_path + '/rf/validation_class'
    xgboost_base_path = base_path + '/xgboost/validation_class'


    ###get the seed
    a = 0.128
    b = 0.351

    seed_knn = mcc[(mcc.knn_MCC >= a)&(mcc.knn_MCC <= b)].seed.unique()
    seed_lr = mcc[(mcc.lr_MCC >= a)&(mcc.lr_MCC <= b)].seed.unique()
    seed_svm = mcc[(mcc.svm_MCC >= a)&(mcc.svm_MCC <= b)].seed.unique()
    seed_rf = mcc[(mcc.rf_MCC >= a)&(mcc.rf_MCC <= b)].seed.unique()
    seed_xgboost = mcc[(mcc.xgboost_MCC >= a)&(mcc.xgboost_MCC <= b)].seed.unique()


    print('knn: ', len(seed_knn))
    print('lr: ', len(seed_lr))
    print('svm: ', len(seed_svm))
    print('rf: ', len(seed_rf))
    print('xgboost: ', len(seed_xgboost))

    tmp = pd.read_csv(join(knn_base_path, 'validation_knn_paras_'+var+'_K_7.csv'))
    knn = tmp[['id', 'y_true']]
    for i, seed in enumerate(seed_knn):    
        col1 = [col for col in tmp.columns if 'prob_knn_seed_'+str(seed) in col]
        knn['knn_seed_'+str(seed)]=tmp[[*col1]]
    #knn.to_csv(join(path+'/knn_train_probabilities.csv'))


    tmp = pd.read_csv(join(lr_base_path, 'validation_lr_paras_'+var+'.csv'))
    lr = tmp[['id', 'y_true']]
    for i, seed in enumerate(seed_lr):
        col1 = [col for col in tmp.columns if 'prob_lr_seed_'+str(seed) in col]
        #print(col1)
        #print(tmp.columns)
        lr['lr_seed_'+str(seed)]=tmp[[*col1]]
    #lr.to_csv(join(path+'/lr_train_probabilities.csv'))


    tmp = pd.read_csv(join(svm_base_path, 'validation_svm_paras_'+var+'.csv'))
    svm = tmp[['id', 'y_true']]
    for i, seed in enumerate(seed_svm):
        col1 = [col for col in tmp.columns if 'prob_svm_seed_'+str(seed) in col]
        svm['svm_seed_'+str(seed)]=tmp[[*col1]]
    #svm.to_csv(join(path+'/svm_train_probabilities.csv'))


    tmp = pd.read_csv(join(rf_base_path, 'validation_rf__paras_'+var+'.csv'))
    rf = tmp[['id', 'y_true']]
    for i, seed in enumerate(seed_rf):
        col1 = [col for col in tmp.columns if 'prob_rf_seed_'+str(seed) in col]
        rf['rf_seed_'+str(seed)]=tmp[[*col1]]
    #rf.to_csv(join(path+'/rf_train_probabilities.csv'))


    tmp = pd.read_csv(join(xgboost_base_path, 'validation_xgboost_paras_'+var+'.csv'))
    xgboost = tmp[['id', 'y_true']]
    for i, seed in enumerate(seed_xgboost):
        col1 = [col for col in tmp.columns if 'prob_xgboost_seed_'+str(seed) in col]
        xgboost['xgboost_seed_'+str(seed)]=tmp[[*col1]]
    #xgboost.to_csv(join(path+'/xgboost_train_probabilities.csv'))


    del lr['y_true']
    del svm['y_true']
    del rf['y_true']
    del xgboost['y_true']


    data = reduce(lambda x,y: pd.merge(x,y, on='id', how='left'), [knn, lr, svm, rf, xgboost])
    data.to_csv(join(probability_path+'/validation_probabilities_'+var+'.csv'))

