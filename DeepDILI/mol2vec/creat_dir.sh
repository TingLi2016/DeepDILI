#!/bin/bash

echo "[start]"
echo `date`

###build separate directory


base_path0='/account/tli/CDER/results/check'

echo "make base classifiers directory"
mkdir -p $base_path0
mkdir -p $base_path0/knn
mkdir -p $base_path0/lr
mkdir -p $base_path0/svm
mkdir -p $base_path0/rf
mkdir -p $base_path0/xgboost

echo "make probability directory"
mkdir -p $base_path0/data/test
mkdir -p $base_path0/data/validation


base_path=$base_path0/result
echo "make dnn directory"

mkdir -p $base_path
#mkdir -p $base_path/logs

mkdir -p $base_path/validation_performance
mkdir -p $base_path/test_performance

mkdir -p $base_path/validation_class
mkdir -p $base_path/test_class

#mkdir -p $base_path/weights


echo "[end]"
echo `date`
