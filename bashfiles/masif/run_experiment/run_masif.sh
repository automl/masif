MODEL='masif_h_iclr' 
DATASET='task_set' # can be 'synthetic_func' 'lcbench' 'openml'
SEED=0
FOLD_IDX=0 # can be {0 .. 10}

# MASIF PARAMETERS, for the other baselines not required
REDUCE='None'
PE='pe_g'
FLC='None'
GUIDED='d_meta_guided'
 
python main.py +experiment=${MODEL} dataset=${DATASET} wandb.mode=online seed=$SEED train_test_split.fold_idx=${FOLD_IDX} +model.model_opts=[${REDUCE},${PE},${FLC},${GUIDED}]
