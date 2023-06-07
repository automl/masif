#! /bin/bash

#SBATCH --mail-type=FAIL
#SBATCH --job-name=imfas
#SBATCH --time=11:00:00
#SBATCH --output=slurm-%j-out.txt
#SBATCH --partition=cpu_short

#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M 

#SBATCH --array=1-5

set -e

working_dir=~/Project/masif/MASIF
cd $working_dir

MODEL=$1
DATASET=$2
FOLD_IDX=$3
ADDITIONAL_ARGUMENT=$4

echo $MODEL
echo $DATASET
echo $FOLD_IDX
echo $ADDITIONAL_ARGUMENT

source /home/${USER}/anaconda/tmp/bin/activate masif

srun python main.py +experiment=${MODEL} dataset=${DATASET} wandb.mode=online seed=$SLURM_ARRAY_TASK_ID train_test_split.fold_idx=${FOLD_IDX} +model.model_opts=['reduce','pe_g','d_meta_guided'] +${ADDITIONAL_ARGUMENT}
