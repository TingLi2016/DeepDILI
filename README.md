### Conventional_DNN
This folder includes three conventional Deep Neural Network (DNN) models developed from Mold2, Mol2vec and MACCS. For each conventional dnn model folder, it provides three profiles, including dataset (.csv), conventional dnn model (.h5) and python script (.py). Make sure repalce the file path of the dataset and model according to your directory on the python script.   

### DeepDILI
This folder includes three DeepDILI models developed from Mold2, Mol2vec and MACCS. In each folder, you can also find the readme.txt file for specific instruction about how to runing the model. 
#### - Mold2 instruction
Three major steps to run mold2_DeepDILI:
0, Create directories to save data
1, Develop base classifiers;
2, Collected the probability output from the selected base classifiers;
3, Fit into the neural network

**Step 0**:
You can run the bash script(creat_dir.sh) to create directory or built diretories by yourself.

**Step 1**: 
Update the directory of QSAR_year_338_pearson_0.9.csv and important_features_order.csv in the following scripts. 
- mold2_knn.py 
* mold2_lr.py
* mold2_svm.py
* mold2_rf.py
* mold2_xgboost.py

Runing example: python mold2_knn.py test ("test" is a name for your work, it can be any word, make sure it is consistent with all the following runing script)

**Step 2**:
Update the directory of combined_score.csv and the directory to collect the output from the base calssifiers in the following scripts.
- mold2_test_predictions_combine.py
- mold2_validation_predictions_combine.py 

Runing example: python mold2_test_predictions_combine.py test

**Step 3**:
Update the directory of mold2_best_model.h5, the step 2 results directory and directory to save your predicitons in the mold2_DeepDILI.py. 
Runing example: python mold2_DeepDILI.py test

#### - Mol2vec instruction
Three major steps to run mol2vec_DeepDILI:
0, Create directories to save data
1, Develop base classifiers;
2, Collected the probability output from the selected base classifiers;
3, Fit into the neural network

Step 0:
You can refer to the bash script(creat_dir.sh) to create directory or built diretories by yourself.

Step 1: 
Update the directory of dili_1009_mol2vec.csv and train_index_1009.csv in the following scripts. 
- mol2vec_knn.py 
- mol2vec_lr.py
- mol2vec_svm.py
- mol2vec_rf.py
- mol2vec_xgboost.py

Runing example: python mol2vec_knn.py test ("test" is a name for your work, it can be any word, make sure it is consistent with all the following runing script)

Step 2:
Update the directory of selected_mol2vec_mcc.csv and the directory to collect the output from the base calssifiers in the following scripts.
- mol2vec_test_predictions_combine.py
- mol2vec_validation_predictions_combine.py 

Runing example: python mol2vec_test_predictions_combine.py test

Step 3:
Update the directory of mol2vec_best_model.h5, the step 2 results directory and directory to save your predicitons in the mol2vec_DeepDILI.py. 
Runing example: python mol2vec_DeepDILI.py test

#### - MACCS instruction
Three major steps to run maccs_DeepDILI:
0, Create directories to save data
1, Develop base classifiers;
2, Collected the probability output from the selected base classifiers;
3, Fit into the neural network

Step 0:
You can refer to the bash script(creat_dir.sh) to create directory or built diretories by yourself.

Step 1: 
Update the directory of dili_1009_maccs.csv and train_index_1009.csv in the following scripts. 
- maccs_knn.py 
- maccs_lr.py
- maccs_svm.py
- maccs_rf.py
- maccs_xgboost.py

Runing example: python maccs_knn.py test ("test" is a name for your work, it can be any word, make sure it is consistent with all the following runing script)

Step 2:
Update the directory of selected_maccs_mcc.csv and the directory to collect the output from the base calssifiers in the following scripts.
- maccs_test_predictions_combine.py
- maccs_validation_predictions_combine.py 

Runing example: python maccs_test_predictions_combine.py test

Step 3:
Update the directory of maccs_best_model.h5, the step 2 results directory and directory to save your predicitons in the maccs_DeepDILI.py. 
Runing example: python maccs_DeepDILI.py test


### Full_DeepDILI
We use the full dataset(1009 compounds) as the training set to develop the DeepDILI model with Mold2. It includes three tables (.csv), one model (.h5), and one python script. Please make sure update the file (.csv and .h5) path on the python script. To screening your interested compounds, please update the test set path with your compounds' Mold2 descriptors.
