#!/account/tli/anaconda3/bin/python

import time
start_time = time.time()

### Warnings
import warnings
warnings.filterwarnings('ignore')

import os
import argparse
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

def mkdir_if_missing(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def main(data_path: str, base_path: str, name: str):
    features = pd.read_csv(os.path.join(data_path,'important_features_order.csv')).feature.unique() # path for important_features_order.csv
    data = pd.read_csv(os.path.join(data_path,'QSAR_year_338_pearson_0.9.csv'),low_memory=False)# path for QSAR_year_338_pearson_0.9.csv
    test_data = data[data.final_year>=1997] 
    #test_data = pd.read_csv(os.path.join(data_path,'data_split.csv')# path for data_split.csv

    data_split = pd.read_csv(os.path.join(data_path,'data_split.csv'))# path for data_split.csv
    mcc = pd.read_csv(os.path.join(data_path,'combined_score.csv')) # path for combined_score.csv

    model_path = os.path.join(data_path,'mold2_best_model.h5') # path for mold2_best_model.h5

    #base_path = '/account/tli/CDER/results/check' # path for base classifiers
    probability_path = os.path.join(base_path, 'probabilities_output') # path for the combined probabilities (model-level representations)
    # mkdir_if_missing(probability_path)

    result_path = os.path.join(base_path,'result') # path for the final DeepDILI predictions
    # mkdir_if_missing(result_path)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep DILI')
    parser.add_argument(
            '--data_path', 
            default='.',
            type=str, help='Data directory')
    parser.add_argument(
            '--base_path', 
            default='./test',
            type=str, help='base path')
    parser.add_argument(
            '--name', 
            default='test',
            type=str, help='Any text')
    args = parser.parse_args()
    # mkdir_if_missing(args.base_path)
    os.system("chmod +x {}".format("creat_dir.sh"))
    os.system("./{} {}".format("creat_dir.sh", args.base_path))
    main(args.data_path, args.base_path, args.name)