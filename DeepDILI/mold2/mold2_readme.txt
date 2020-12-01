### Three major steps to run mold2_DeepDILI:
0, Create directories to save data
1, Develop base classifiers;
2, Collected the probability output from the selected base classifiers;
3, Fit into the neural network

Step 0:
You can run the bash script(creat_dir.sh) to create directory or built diretories by yourself.

Step 1: 
Update the directory of QSAR_year_338_pearson_0.9.csv and important_features_order.csv in the following scripts. 
- mold2_knn.py 
- mold2_lr.py
- mold2_svm.py
- mold2_rf.py
- mold2_xgboost.py
Runing example: python mold2_knn.py test ("test" is a name for your work, it can be any word, make sure it is consistent with all the following runing script)

Step 2:
Update the directory of combined_score.csv and the directory to collect the output from the base calssifiers in the following scripts.
-mold2_test_predictions_combine.py
-mold2_validation_predictions_combine.py 
Runing example: python mold2_test_predictions_combine.py test

Step 3:
Update the directory of mold2_best_model.h5, the step 2 results directory and directory to save your predicitons in the mold2_DeepDILI.py. 
Runing example: python mold2_DeepDILI.py test
