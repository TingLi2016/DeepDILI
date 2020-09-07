# DeepDILI

  ### Base classifier development
  - lr.py generates 100 base classifiers with the methods of Logistic Regression (LR)
  - knn.py generates 100 base classifiers with the methods of kNN
  - svm.py generates 100 base classifiers with the methods of Support Vector Machine (SVM)
  - rf.py generates 100 base classifiers with the methods of Random Forest (RF)
  - xgboost.py generates 100 base classifiers with the methods of eXtreme Gradient Boosting (XGBoost)

  ### Meta classifier development
  - meta_classifier_dnn.py is the meta classifier


# Conventional machine learning classifiers

  -conventional_model.py includes five single (LR, kNN, SVM, RF, and XGBoost) conventional models to evaluate the test set independently. The result was compared with DeepDILI result.
  -deep_dnn.py is a six layer neural network classifier to evaluate the test set. The result was compared with DeepDILI result. 
  
# Best model parameters
  -best_model.h5 is the DeepDILI parameters file
