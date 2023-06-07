#for DATASET in 'task_set' # 'openml_alex' 'task_set' # 'synthetic_func' 'lcbench'
#do
#    for FOLDIDX in {0..9}
#    do
#        for DROPOUT in 0.0 0.1 0.3 0.4 0.5
#        do
#            sbatch masif.sh 'masif_h_iclr'  $DATASET $FOLDIDX "dropout=${DROPOUT}"
#        done
#    done
#done


#for DATASET in 'task_set' # 'openml_alex' 'task_set' # 'synthetic_func' 'lcbench'
#do
#    for FOLDIDX in {0..9}
#    do
#        for LR in 0.0001 0.01 0.1
#        do
#            sbatch masif.sh 'masif_h_iclr'  $DATASET $FOLDIDX "lr=${LR}"
#        done
#    done
#done


#for DATASET in 'task_set' # 'openml_alex' 'task_set' # 'synthetic_func' 'lcbench'
#do
#    for FOLDIDX in {0..9}
#    do
#        for LR in 0.0001 0.001 0.01 0.1
#        do
#            sbatch masif.sh 'masif_h_iclr'  $DATASET $FOLDIDX "lr=${LR} optimizer@trainer.trainerobj.optimizer=adamw"
#        done
#    done
#done



#for DATASET in 'task_set' # 'openml_alex' 'task_set' # 'synthetic_func' 'lcbench'
#do
#    for FOLDIDX in {0..9}
#    do
#        for NTFLAYERS in 1 3 4 
#        do
#            sbatch masif.sh 'masif_h_iclr'  $DATASET $FOLDIDX "ntflayers=${NTFLAYERS}"
#        done
#    done
#done

#for DATASET in 'task_set' # 'openml_alex' 'lcbench'  'synthetic_func' 'task_set' 
#do
#    for FOLDIDX in {0..9}
#    do
#        for NHEADS in 4 8
#        do
#            for DMODEL in 64 128 256
#            do
#                sbatch masif.sh 'masif_h_iclr'  $DATASET $FOLDIDX "dmodel=${DMODEL} +nhead=${NHEADS}"
#            done
#        done
#    done
#done



for DATASET in 'openml_alex' 'lcbench'  'synthetic_func' # 'task_set' 
do
	for FOLDIDX in {0..9}
	do
	    sbatch masif.sh 'masif_h_iclr'  $DATASET $FOLDIDX "dropout=0.4 +lr=0.001 +ntflayers=4 optimizer@trainer.trainerobj.optimizer=adamw"
	done
done

