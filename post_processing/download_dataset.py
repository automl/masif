import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import wandb
from pathlib import Path
api = wandb.Api()
import os

# Project is specified by <entity/project-name>
runs = api.runs("tnt/imfas-iclr", filters={
                    "state": "finished"
                })
allowed_models = {
    #'RandomBaselineFIXED': 'Random_Baseline',
    #'masif_H_transformerNew_Reduce_GPe_DGuide': 'MASIF',
    #'masif_H_transformerNew_Reduce_GPe_DGuideSHScheduler': 'MASIF_SH',
    #'masif_H_transformer_Reduce_GPe_DGuide_NoMD': 'MASIF_NoMD',
    #'masif_H_transformer_Reduce_GPe_DGuide_NoMA': 'MASIF_NoMA',
    #'masif_H_transformer_Reduce_GPe_DGuide_NoMDA': 'MASIF_NoMDA',
    #'Baseline: SH': 'SH',
    #'augmented satzilla_11': 'SatZilla11',
    #'satzilla_11': 'Augmented_SatZilla11',
    #'Baseline: Best Parametric': 'Parametric',
    #'augmented satzilla_11OpenML': 'Augmented_SatZilla11',
    #"masif_WP_NoReduce": 'MASIF'
    #'masif_H_transformerNew_Reduce_GPe_DGuide_Ablation_lr0.001_dp0.4_ntflayers4_adamw': 'MASIF_adamw_lr0.001_dp0.4_ntfkayers4'
    'Baseline LCNET2': "LCNet"
}

allowed_datasets = {
    'LCBench': 'LCBench',
    #'Synthetic': 'Synthetic',
    #'openml': 'OpenML',
    #'task_set_new': 'Task_Set'
    #'task_set_cnn': 'Task_Set NLP'
}

metrics = {
#'Test, Slice Evaluation: spearman': 'Spearman',
'Test, Slice Evaluation: top1_regret': 'Top1',
'Test, Slice Evaluation: top3_regret': 'Top3',
'Test, Slice Evaluation: top5_regret': 'Top5'
}

summary_list = {}
config_list = {}
name_list = {}
rh_list = {}

valid_datasets = []
for i in range(10):
    for key in allowed_datasets.keys():
        valid_datasets.append(f'{key}_{i}')
valid_datasets = set(valid_datasets)
for dataset in set(allowed_datasets.values()):
    summary_list[dataset] = {}
    config_list[dataset] = {}
    name_list[dataset] = {}
    rh_list[dataset] = {}

for model in set(allowed_models.values()):
    for dataset in set(allowed_datasets.values()):
        summary_list[dataset][model] = []
        config_list[dataset][model] = []
        name_list[dataset][model] = []
        rh_list[dataset][model] = []


for wandb_run in runs:
    group = wandb_run.group
    job_type = wandb_run.job_type
    if group in allowed_models.keys():
        for dataset in allowed_datasets:
            if job_type.startswith(dataset) and job_type in valid_datasets:
                if dataset == 'openml' and group == 'augmented satzilla_11':
                    continue
                dataset_name = allowed_datasets[dataset]
                model = allowed_models[group]
                summary_list[dataset_name][model].append(wandb_run.summary._json_dict)

                config = {k: v for k, v in wandb_run.config.items() if not k.startswith('_')}
                config_list[dataset_name][model].append(config)
                try:
                    if model != 'LCNet':
                        rh = wandb_run.history(x_axis='fidelity', keys=list(metrics.keys()))[['fidelity', *list(metrics.keys())]].melt(id_vars='fidelity').dropna(0)
                        rh = rh.pivot(index='fidelity', columns='variable', values='value')
                        rh_list[dataset_name][model].append(rh)

                    else:
                        rh = [wandb_run.history(x_axis='fidelity', keys=[key]) for key in metrics.keys()]
                        rh = pd.concat(rh).melt(id_vars='fidelity').dropna(0)
                        rh = rh.pivot(index='fidelity', columns='variable', values='value')
                        rh_list[dataset_name][model].append(rh)

                except Exception as e:
                    print(e)
                    print('Failed!!!!!')
                    print(group)
                    print(job_type)

                # run.name is the name of the run.
                name_list[dataset_name][model].append(wandb_run.name)

path = Path.cwd()
for dataset in set(allowed_datasets.values()):
    for model in set(allowed_models.values()):
        if len(summary_list[dataset][model]) > 0:
            summary_df = pd.DataFrame.from_records(summary_list[dataset][model])
            config_df = pd.DataFrame.from_records(config_list[dataset][model])
            name_df = pd.DataFrame({'name': name_list[dataset][model]})
            rh = rh_list[dataset][model]
            try:
                all_df = pd.concat([name_df, config_df, summary_df], axis=1)
                all_rh = pd.concat(rh, axis=1)
            except:
                continue
            if not (path / dataset).exists():
                os.makedirs(path / dataset, exist_ok=True)
            for key, value in metrics.items():
                all_rh[key].to_csv(str(path / dataset / f'{model}_{value}.csv' ))



runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv("project.csv")
