import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import scipy

DATASET = 'Task_Set'
METRIC = 'Top1'

random = {
    'LCBench': {
        'Top1': 0.1469679747,
        'Top3': 0.06599725285,
    },
    'Task_Set': {
        'Top1': 64.56159885,
        'Top3': 49.80802285,
    },
    'Synthetic': {
        'Top1': 37.67961838,
        'Top3': 49.80802285
    },
    'OpenML': {
        'Top1': 0.06559239471,
        'Top3': 0.02349796821
    },
    'Task_Set_NLP':{
        'Top1': 81.948,
        'Top3': 60.792
    }

}

satzilla = {
    'LCBench': {
        'Top1': 0.09554,
        'Top3': 0.02372,
    },
    'OpenML': {
        'Top1': 0.04035,
        'Top3': 0.01416,
    }

}

plot_keys = {
    'random': 'Random Baseline',

    # 'MASIF': 'MASIF',
    # 'MASIF_SH': 'MASIF + SH',
    'SH': 'SH',
    # 'MASIF_NoMD': 'MASIF NoMD',
    # 'MASIF_NoMA': 'MASIF NoMA',
    # 'MASIF_NoMDA':'MASIF NoMDA',
    # 'MASIF_M' : 'MASIF with MLP',
    # 'Imfas': 'Imfas',
    # 'Augmented_SatZilla11': 'Augmented SatZilla11',
    'Parametric': 'Parametric',
    # 'Satzilla': 'Satzilla11',


    'MASIF': 'MASIF',


    'MASIF_dp_0.0': 'MASIF_dp_0.0',
    'MASIF_dp_0.1': 'MASIF_dp_0.1',
    'MASIF_dp_0.3': 'MASIF_dp_0.3',
    'MASIF_dp_0.4': 'MASIF_dp_0.4',
    'MASIF_dp_0.5': 'MASIF_dp_0.5',
    'MASIF_lr_0.0001': 'MASIF_lr_0.0001',
    'MASIF_lr_0.001': 'MASIF_lr_0.001',
    'MASIF_lr_0.01': 'MASIF_lr_0.01',
    'MASIF_lr_0.1': 'MASIF_lr_0.1',
    'MASIF_lr_0.0001_adamw': 'MASIF_lr_0.0001_adamw',
    'MASIF_lr_0.001_adamw': 'MASIF_lr_0.001_adamw',
    'MASIF_lr_0.01_adamw': 'MASIF_lr_0.01_adamw',
    'MASIF_lr_0.1_adamw': 'MASIF_lr_0.1_adamw',
    'MASIF_ntflayers_1': 'MASIF_ntflayers_1',
    'MASIF_ntflayers_3': 'MASIF_ntflayers_3',
    'MASIF_ntflayers_4': 'MASIF_ntflayers_4',
    'MASIF_ntflayers_64_nhead_4': 'MASIF_dmodel_64_nhead_4',
    'MASIF_ntflayers_128_nhead_4': 'MASIF_dmodel_128_nhead_4',
    'MASIF_ntflayers_256_nhead_4': 'MASIF_dmodel_256_nhead_4',
    'MASIF_ntflayers_64_nhead_8': 'MASIF_dmodel_64_nhead_8',
    'MASIF_ntflayers_128_nhead_8': 'MASIF_dmodel_128_nhead_8',
    'MASIF_ntflayers_256_nhead_8': 'MASIF_dmodel_256_nhead_8',
    'MASIF_lr_0.001_adamw_dp_0.4_ntflayers_4': 'MASIF_lr_0.001_adamw_dp_0.4_ntflayers_4'

}

regret_keys = [
    'Test, Slice Evaluation: top1_regret',
    'Test, Slice Evaluation: top3_regret',
]

regret_types = ['_Top1', '_Top3']

regrets = {}

selection = ['dp', 'lr', 'adamw', 'ntflayers', 'architecture', 'mixed', 'nhead4', 'nhead8'][5]
selection_full_names = {'dp': "DropOut",
                        'lr': "Learning Rate",
                        'adamw': "AdamW Optimizer",
                        "ntflayers": "Number of Transformer Layers",
                        'architecture': "Transformer Architecture",
                        'mixed': 'Marginal Best'}
colors = {
    'dp': {
        'random': 'black',
        'MASIF': 'tab:blue',
        'MASIF_dp_0.0': 'tab:orange',
        'MASIF_dp_0.1': 'tab:green',
        'MASIF_dp_0.3': 'tab:red',
        'MASIF_dp_0.4': 'tab:purple',
        'MASIF_dp_0.5': 'tab:brown',
    },

    'lr': {
        'random': 'black',
        'MASIF': 'tab:blue',
        'MASIF_lr_0.0001': 'tab:orange',
        # 'MASIF_lr_0.001': 'tab:green',
        'MASIF_lr_0.01': 'tab:purple',
        'MASIF_lr_0.1': 'tab:red',
    },

    'adamw': {
        'random': 'black',
        'MASIF': 'tab:blue',
        'MASIF_lr_0.0001_adamw': 'tab:orange',
        'MASIF_lr_0.001_adamw': 'tab:green',
        'MASIF_lr_0.01_adamw': 'tab:purple',
        'MASIF_lr_0.1_adamw': 'tab:red',
    },

    'ntflayers': {
        'random': 'black',
        'MASIF': 'tab:blue',
        'MASIF_ntflayers_1': 'tab:orange',
        'MASIF_ntflayers_3': 'tab:green',
        'MASIF_ntflayers_4': 'tab:brown',
    },

    'architecture': {
'random': 'black',
        'MASIF': 'tab:blue',
        'MASIF_ntflayers_64_nhead_4': 'tab:orange',
        # 'MASIF_ntflayers_128_nhead_4': 'tab:green',
        'MASIF_ntflayers_256_nhead_4': 'tab:purple',
        'MASIF_ntflayers_64_nhead_8': 'tab:red',
        'MASIF_ntflayers_128_nhead_8': 'tab:brown',
        'MASIF_ntflayers_256_nhead_8': 'tab:pink',
    },

    'nhead4': {
        'random': 'black',
        'MASIF': 'tab:blue',
        'MASIF_ntflayers_64_nhead_4': 'tab:orange',
        'MASIF_ntflayers_128_nhead_4': 'tab:green',
        'MASIF_ntflayers_256_nhead_4': 'tab:purple',
    },

    'nhead8': {
        'random': 'black',
        'MASIF': 'tab:blue',
        'MASIF_ntflayers_64_nhead_8': 'tab:orange',
        'MASIF_ntflayers_128_nhead_8': 'tab:green',
        'MASIF_ntflayers_256_nhead_8': 'tab:purple',
    },

    'mixed':{
        'random': 'black',
        'MASIF': 'tab:blue',
        'MASIF_dp_0.4': 'tab:purple',
        'MASIF_lr_0.001_adamw': 'tab:green',
        'MASIF_ntflayers_4': 'tab:brown',
        'MASIF_lr_0.001_adamw_dp_0.4_ntflayers_4': 'tab:cyan'
}

    # 'MASIF': 'tab:blue',
    # 'MASIF_SH': 'tab:orange',
    # 'SH': 'tab:green',
    # 'Parametric': 'tab:red',
    # 'Augmented_SatZilla11': 'tab:purple',
    # 'Imfas': 'tab:brown',
    # 'random': 'black',
    # 'Satzilla': 'tab:pink',
    # 'MASIF_NoMD': 'tab:olive',
    # 'MASIF_NoMA': 'tab:cyan',
    # 'MASIF_NoMDA': 'tab:gray',
    # 'MASIF_M': 'm'

}[selection]

working_path = Path.cwd() / 'plots' / DATASET

plots = {}
max_fidelity = -1
plt.rcParams['font.size'] = '55'
linewidth = 5.0

fig, ax = plt.subplots(1, 1)
for model in colors.keys():
    data_path = working_path / f'{model}_{METRIC}.csv'

    if not data_path.exists():
        continue
    data = pd.read_csv(data_path).drop('fidelity', 1)
    data = data.fillna(method='ffill').values
    mean = np.nanmean(data, 1)
    bar = scipy.stats.sem(data, 1, nan_policy='omit')

    # bar = np.nanstd(data, 1)
    x_axis = np.arange(len(mean)) / (len(mean) - 1)
    if max_fidelity < 0:
        max_fidelity = len(mean)

    ax.plot(x_axis, mean,
        label=plot_keys[model],
        color=colors[model], linewidth=linewidth)

    ax.fill_between(x_axis,
                    mean - bar,
                    mean + bar,
                    alpha=0.2,
                    facecolor=colors[model])

ax.hlines(random[DATASET][METRIC], xmin=0, xmax=1, linestyles='dashed', label=plot_keys['random'],
          colors=colors['random'], linewidth=linewidth)

if DATASET in satzilla:
    ax.hlines(satzilla[DATASET][METRIC], xmin=0, xmax=1, linestyles='dashed',
              label=plot_keys['Satzilla'],
              colors=colors['Satzilla'], linewidth=linewidth)
ax.grid()

handles = []
labels = []

handles, labels = ax.get_legend_handles_labels()
# fig_legend = plt.figure(figsize=(35, 1.5)) # NOMDA
# fig_legend = plt.figure(figsize=(22, 1.5)) # MLP
# fig_legend.legend(handles, labels, loc='lower center', ncol=3, labelspacing=0. ) # ablation
"""
fig_legend = plt.figure(figsize=(33, 4.5))
fig_legend.legend(handles, labels, loc='lower center', ncol=2, labelspacing=0.)
fig_legend.canvas.draw()
fig_legend.savefig(f'legend_{selection}.png')
exit()
"""
# plt.xticks(np.arange(0, max_fidelity + 1, 2), np.arange(10+1) / 10)
plt.xlabel('Available Fidelity')
plt.ylabel(f"Top-{METRIC[-1]} Regret")
# plt.legend()
if DATASET == 'OpenML':
    plt.title('Scikit-CC18')
else:
    plt.title(' '.join(DATASET.split('_')) + f' Ablation on {selection_full_names[selection]}')

# plt.legend()
# plt.subplots_adjust(0.067, 0.105, 0.983, 0.943)
plt.subplots_adjust(0.08, 0.12, 0.99, 0.92)
plt.show()
