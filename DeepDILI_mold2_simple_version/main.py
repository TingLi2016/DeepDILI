#!/account/tli/anaconda3/bin/python

import time
start_time = time.time()

### Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

### import scripts
import mold2_knn
import mold2_lr
import mold2_svm
import mold2_rf
import mold2_xgboost

import mold2_validation_predictions_combine
import mold2_test_predictions_combine

import mold2_DeepDILI

### please update the following path 
features = pd.read_csv('/account/tli/CDER/script/train_validation_test/mold2/mold2_download_github_two/important_features_order.csv').feature.unique() # path for important_features_order.csv
data = pd.read_csv('/account/tli/CDER/script/train_validation_test/mold2/mold2_download_github_two/QSAR_year_338_pearson_0.9.csv',low_memory=False)# path for QSAR_year_338_pearson_0.9.csv
test_data = data[data.final_year>=1997] 
#test_data = pd.read_csv('/account/tli/CDER/script/train_validation_test/mold2/mold2_download_github_three/external_mold2.csv')# path for external_mold2.csv (This is the external validation set)

data_split = pd.read_csv('/account/tli/CDER/script/train_validation_test/mold2/mold2_download_github_two/data_split.csv')# path for data_split.csv
mcc = pd.read_csv('/account/tli/CDER/script/train_validation_test/mold2/mold2_download_github_two/combined_score.csv') # path for combined_score.csv

base_path = '/account/tli/CDER/results/check' # path for base classifiers
probability_path = '/account/tli/CDER/results/check/probabilities_output' # path for the combined probabilities (model-level representations)
name = 'test' # can be any name 

model_path = '/account/tli/CDER/script/train_validation_test/mold2/mold2_download_github_two/mold2_best_model.h5' # path for mold2_best_model.h5
result_path = '/account/tli/CDER/results/check/result' # path for the final DeepDILI predictions

### run the scripts
mold2_knn.generate_baseClassifiers(features, data, test_data, data_split, name, base_path)
mold2_lr.generate_baseClassifiers(features, data, test_data, data_split, name, base_path)
mold2_svm.generate_baseClassifiers(features, data, test_data, data_split, name, base_path)
mold2_rf.generate_baseClassifiers(features, data, test_data, data_split, name, base_path)
mold2_xgboost.generate_baseClassifiers(features, data, test_data, data_split, name, base_path)

mold2_validation_predictions_combine.combine_validation_probabilities(base_path, mcc, probability_path, name)
mold2_test_predictions_combine.combine_test_probabilities(base_path, mcc, probability_path, name)

mold2_DeepDILI.dili_prediction(probability_path, name, model_path, result_path)


print("--- %s seconds ---" % (time.time() - start_time))

