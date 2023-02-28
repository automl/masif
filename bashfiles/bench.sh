#!/bin/bash
#source ~/miniconda3/bin/activate gravitas

# rbv2 benchmark is exactly the same for all subbenchmarks -- hence ranger only
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=ranger model.factor=2 seed=0 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=ranger model.factor=2 seed=1 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=ranger model.factor=2 seed=2 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=ranger model.factor=2 seed=3 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=ranger model.factor=2 seed=4 dataset.metric=f1


python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=xgboost model.factor=2 seed=0 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=xgboost model.factor=2 seed=1 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=xgboost model.factor=2 seed=2 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=xgboost model.factor=2 seed=3 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=xgboost model.factor=2 seed=4 dataset.metric=f1


python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=aknn model.factor=2 seed=0 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=aknn model.factor=2 seed=1 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=aknn model.factor=2 seed=2 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=aknn model.factor=2 seed=3 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=aknn model.factor=2 seed=4 dataset.metric=f1


python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=rpart model.factor=2 seed=0 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=rpart model.factor=2 seed=1 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=rpart model.factor=2 seed=2 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=rpart model.factor=2 seed=3 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=rpart model.factor=2 seed=4 dataset.metric=f1


python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=svm model.factor=2 seed=0 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=svm model.factor=2 seed=1 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=svm model.factor=2 seed=2 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=svm model.factor=2 seed=3 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=svm model.factor=2 seed=4 dataset.metric=f1


python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=super model.factor=2 seed=0 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=super model.factor=2 seed=1 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=super model.factor=2 seed=2 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=super model.factor=2 seed=3 dataset.metric=f1
python main.py +experiment=successive_halving_trainsize dataset_raw.enable=True dataset_raw.bench=super model.factor=2 seed=4 dataset.metric=f1



# lcbench Train/val_accuracy: but for both factors 2 & 3
python main.py +experiment=successive_halving_epoch_raw_lcbench model.factor=2 dataset_raw.enable=True seed=0
python main.py +experiment=successive_halving_epoch_raw_lcbench model.factor=2 dataset_raw.enable=True seed=1
python main.py +experiment=successive_halving_epoch_raw_lcbench model.factor=2 dataset_raw.enable=True seed=2
python main.py +experiment=successive_halving_epoch_raw_lcbench model.factor=2 dataset_raw.enable=True seed=3
python main.py +experiment=successive_halving_epoch_raw_lcbench model.factor=2 dataset_raw.enable=True seed=4


python main.py +experiment=successive_halving_epoch_raw_lcbench model.factor=3 dataset_raw.enable=True seed=0
python main.py +experiment=successive_halving_epoch_raw_lcbench model.factor=3 dataset_raw.enable=True seed=1
python main.py +experiment=successive_halving_epoch_raw_lcbench model.factor=3 dataset_raw.enable=True seed=2
python main.py +experiment=successive_halving_epoch_raw_lcbench model.factor=3 dataset_raw.enable=True seed=3
python main.py +experiment=successive_halving_epoch_raw_lcbench model.factor=3 dataset_raw.enable=True seed=4

