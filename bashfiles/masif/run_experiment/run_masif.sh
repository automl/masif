
# MASIF PARAMETERS, for the other baselines not required
MODEL='masif_h_iclr' 
REDUCE='None'
PE='pe_g'
FLC='None'
GUIDED='d_meta_guided'

# run MASIF
for DATASET in 'task_set' 'synthetic_func' 'lcbench' 'openml' 
do
   for SEED in {0 .. 4}
   do
       for FOLD_IDX in {0 .. 9}
       do
            python main.py +experiment=${MODEL} dataset=${DATASET} wandb.mode=online seed=$SEED train_test_split.fold_idx=${FOLD_IDX} +model.model_opts=[${REDUCE},${PE},${FLC},${GUIDED}]
       done
   done
done


# Run BASELINES that requries meta features. 
for MODEL in 'lcnet' 'random' 'successivehalving'  'augmented_algorithm_selection' 'algorithm_selection'
do
    for DATASET in 'lcbench' 'openml' 
    do
        for SEED in {0 .. 4}
        do
            for FOLD_IDX in {0 .. 9}
            do
                    python main.py +experiment=${MODEL} dataset=${DATASET} wandb.mode=online seed=$SEED train_test_split.fold_idx=${FOLD_IDX}
            done
        done
    done
done

# Run baselines that do not require meta features
for MODEL in 'random' 'successivehalving'
do
    for DATASET in 'task_set' 'synthetic_func' 'lcbench' 'openml' 
    do
        for SEED in {0 .. 4}
        do
            for FOLD_IDX in {0 .. 9}
            do
                    python main.py +experiment=${MODEL} dataset=${DATASET} wandb.mode=online seed=$SEED train_test_split.fold_idx=${FOLD_IDX}
            done
        done
    done
done
