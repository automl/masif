#!/bin/bash

python main.py +experiment=baseline_sh model.eta=2 seed=1 \
 wandb.job_type=sh_eta2 \
 dataset=lcbench  dataset.dataset_meta=lcbench_minimal dataset.algo_meta=lcbench_minimal



python main.py +experiment=baseline_sh model.eta=3 seed=1 \
 wandb.job_type=sh_eta3 \
 dataset=lcbench  dataset.dataset_meta=lcbench_minimal dataset.algo_meta=lcbench_minimal