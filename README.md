# MASIF: Meta-learned Algorithm Selection using Implicit Fidelity Information

Selecting a well-performing algorithm for a given task or dataset can be time-consuming and
tedious, but is nevertheless required for the successful day-to-day business of developing new
AI applications. Algorithm Selection (AS) mitigates this through a meta-model leveraging
meta-information about tasks. However, most of the classical AS methods are error-prone
because they aim at describing a task by either statistical meta-features of the dataset
or cheap evaluations of proxy algorithms, called landmarks. Similarly, other approaches
solely based on partial learning curves of the candidate algorithms have been parametric
and/or myopic. In this work, we extend the classical AS data setup to include multi-fidelity
information and demonstrate how meta-learning on algorithms’ learning behaviour allows
us to exploit test-time evidence effectively and combat myopia significantly. We further
postulate a budget-regret trade-off w.r.t. the selection process. Using this novel setup, we
derive a new class of algorithm selectors that actively gather online evidence in the form of
the candidate algorithms’ partial learning curves. Our new selector masif leverages a
transformer-based encoder to interpret the set of learning curves with varying lengths jointly
in a non-parametric and non-myopic manner. This opens up new possibilities for guided
rapid prototyping on cheaply observed partial algorithm learning curves in data science.
We empirically demonstrate that masif combats myopia effectively and can jointly
interpret partial learning curves of the candidate algorithm


## Preparation

If you want to use LCBench, you have to download the dataset first:

```bash
bash scripts/download_lcbench.sh
```

## Installation
```bash
cd masif
conda create -n masif python=3.9.7
conda activate masif

# Install for usage
pip install -e .

# Install for development
make install-dev
```


## Start experiments

An example command is fiven below
```bash
python main.py '+experiment=masif_h'+model.model_opts=['reduce','pe_g','d_meta_guided']
```

The project extensively uses [hydra](https://hydra.cc/docs/intro/) for configurations and [Weights and Biases](https://wandb.ai/site) for tracking experiments. Please set-up the project and account on this and then update ```configs/base.yaml``` with the ```entity``` and ```project_name``` fields for running full tests. 
Notice, that all the experiments can be found in bashfiles/masif


## Generate synthetic data 
To generate synthetic data, you can first call the scipt
```bash
python masif/models/baselines/lcdb_parametric_best_lc.py
```

to generate an optimized parameteric curve sets for one particular dataset. Then you need to call 
```bash
python masif/models/baselines/synthetic_parameteric_curves.py
```

The synthetic dataset used in our experiment can be found under `data/preprocessed/Synthetic/synthetic_func.npy`
