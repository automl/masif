import pathlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import wandb
from pathlib import Path
api = wandb.Api()
import os
max_fildelity = 51
# Project is specified by <entity/project-name>
runs = api.runs("tnt/imfas-iclr", filters={
                    "group": "Baseline: Best Parametric",
                })
allowed_models = {
    'Baseline: Best Parametric': 'Parametric',
}

allowed_datasets = {
    'task_set_cnn': 'Task_Set NLP'
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


for wandb_run in runs:
    group = wandb_run.group
    job_type = wandb_run.job_type
    if group in allowed_models.keys():
        for dataset in allowed_datasets:
            if job_type.startswith(dataset) and job_type in valid_datasets:
                dataset_name = allowed_datasets[dataset]
                model = allowed_models[group]

                try:
                    rh = wandb_run.history()[['fidelity', *list(metrics.keys())]].melt(id_vars='fidelity').dropna(0)
                    rh = rh.pivot(index='fidelity', columns='variable', values='value')

                    res = pathlib.Path('/home/deng/Project/imfas/PLres') / f"fold_{wandb_run.config['train_test_split']['fold_idx']}" / f"seed_{wandb_run.config['seed']}"
                    if rh.shape[0] < 51:
                        continue
                        if not res.exists():
                            os.makedirs(res)
                        rh.to_csv(res / 'res.csv')
                    else:
                        if not res.exists():
                            os.makedirs(res)
                        rh.to_csv(res / 'res.csv')


                except Exception as e:
                    print(e)

                    print('!' * 50)
                    print('Failed!!!!!')
                    print(group)
                    print(job_type)

                # run.name is the name of the run.
import pdb
pdb.set_trace()

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
