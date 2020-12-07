import pandas as pd
import numpy as np

import sys

var=sys.argv[1]


import os
from os import listdir
from os.path import isfile, join
import itertools
from functools import reduce


base_path = '/account/tli/CDER/results/check'

knn_base_path = base_path + '/knn/knn_' + var + '/test_class'
lr_base_path = base_path +  '/lr/lr_' + var + '/test_class'
svm_base_path = base_path + '/svm/svm_' + var + '/test_class'
rf_base_path = base_path + '/rf/rf_' + var + '/test_class'
xgboost_base_path = base_path + '/xgboost/xgboost_' + var + '/test_class'


path='/account/tli/CDER/results/check/data/test'


###get the seed
mcc=pd.read_csv('/account/tli/CDER/data/fps_extra/selected_models/selected_mol2vec_mcc.csv')

seed_knn = mcc[mcc.model == 'knn'].seed.unique()
seed_lr = mcc[mcc.model == 'lr'].seed.unique()
seed_svm = mcc[mcc.model == 'svm'].seed.unique()
seed_rf = mcc[mcc.model == 'rf'].seed.unique()
seed_xgboost = mcc[mcc.model == 'xgboost'].seed.unique()


print('knn: ', len(seed_knn))
print('lr: ', len(seed_lr))
print('svm: ', len(seed_svm))
print('rf: ', len(seed_rf))
print('xgboost: ', len(seed_xgboost))



tmp = pd.read_csv(join(knn_base_path, 'test_knn_paras_'+var+'.csv'))
knn = tmp[['id', 'y_true']]
for i, seed in enumerate(seed_knn):    
    col1 = [col for col in tmp.columns if 'prob_knn_seed_'+str(seed) in col]
    knn['knn_seed_'+str(seed)]=tmp[[*col1]]
#knn.to_csv(join(path+'/knn_train_probabilities.csv'))


tmp = pd.read_csv(join(lr_base_path, 'test_lr_paras_'+var+'.csv'))
lr = tmp[['id', 'y_true']]
for i, seed in enumerate(seed_lr):
    col1 = [col for col in tmp.columns if 'prob_lr_seed_'+str(seed) in col]
    #print(col1)
    #print(tmp.columns)
    lr['lr_seed_'+str(seed)]=tmp[[*col1]]
#lr.to_csv(join(path+'/lr_train_probabilities.csv'))


tmp = pd.read_csv(join(svm_base_path, 'test_svm_paras_'+var+'.csv'))
svm = tmp[['id', 'y_true']]
for i, seed in enumerate(seed_svm):
    col1 = [col for col in tmp.columns if 'prob_svm_seed_'+str(seed) in col]
    svm['svm_seed_'+str(seed)]=tmp[[*col1]]
#svm.to_csv(join(path+'/svm_train_probabilities.csv'))


tmp = pd.read_csv(join(rf_base_path, 'test_rf__paras_'+var+'.csv'))
rf = tmp[['id', 'y_true']]
for i, seed in enumerate(seed_rf):
    col1 = [col for col in tmp.columns if 'prob_rf_seed_'+str(seed) in col]
    rf['rf_seed_'+str(seed)]=tmp[[*col1]]
#rf.to_csv(join(path+'/rf_train_probabilities.csv'))


tmp = pd.read_csv(join(xgboost_base_path, 'test_xgboost_paras_'+var+'.csv'))
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
data.to_csv(join(path+'/test_probabilities_'+var+'.csv'))


