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
-maccs_test_predictions_combine.py
-maccs_validation_predictions_combine.py 
Runing example: python maccs_test_predictions_combine.py test

Step 3:
Update the directory of maccs_best_model.h5, the step 2 results directory and directory to save your predicitons in the maccs_DeepDILI.py. 
Runing example: python maccs_DeepDILI.py test
