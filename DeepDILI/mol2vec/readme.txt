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
-mol2vec_test_predictions_combine.py
-mol2vec_validation_predictions_combine.py 
Runing example: python mol2vec_test_predictions_combine.py test

Step 3:
Update the directory of mol2vec_best_model.h5, the step 2 results directory and directory to save your predicitons in the mol2vec_DeepDILI.py. 
Runing example: python mol2vec_DeepDILI.py test
