#!/bin/bash

cd ..

for MODEL in 'random' # 'parametric_best_lc ' # 'successivehalving' # 'random' 'successivehalving'
#for MODEL in 'masif_tmlr'  'masif_sh_scheduler'
do
    for DATASET in 'task_set' 'synthetic_func'  'lcbench' 'openml_alex' # 'synthetic_func' 'task_set' # 'openml'
    do
        for FOLDIDX in {0..5}
        do
            python main.py +experiment=random dataset=$DATASET  wandb.mode=online train_test_split.fold_idx=$FOLDIDX wandb.group='random_baeline'
        done
    done
done
