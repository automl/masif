import os
import random
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
import wandb

# from networkx import Graph, minimum_spanning_edges
from omegaconf import DictConfig, OmegaConf


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def print_cfg(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


def freeze(listoflayers, frosty=True):
    """freeze the parameters of the list of layers"""
    for l in listoflayers:
        for p in l.parameters():
            p.requires_grad = not frosty


def freeze_tensors(listoftensors, frosty=True):
    for t in listoftensors:
        t.requires_grad = not frosty


def calc_min_eucl_spanning_tree(d_test: torch.tensor):
    """
    calculates the minimum spanning tree of the euclidean distance matrix

    :param d_test: torch.tensor: the euclidean distance matrix

    :return: torch.tensor: the minimum spanning tree

    """
    dist_mat = torch.cdist(d_test, d_test)
    dist_mat = dist_mat.cpu().detach().numpy()

    nodes = list(range(len(dist_mat)))
    d = [(src, dst, dist_mat[src, dst]) for src in nodes for dst in nodes if src != dst]

    df = pd.DataFrame(data=d, columns=["src", "dst", "eucl"])

    g = Graph()
    for index, row in df.iterrows():
        g.add_edge(row["src"], row["dst"], weight=row["eucl"])

    return list(minimum_spanning_edges(g))


def check_diversity(representation, title, epsilon=0.01):
    """
    :param representation: ndarray.
    :param title: name of the matrix
    :param epsilon: float: the value needed to exceed (should be close to zero)
    :raises: Warning if representation is not diverse
    """
    # Check for (naive) representation collapse by checking sparsity after
    # translation by 90% quantile
    translated = representation - np.quantile(representation, 0.9, axis=0)
    sparsity = (translated < epsilon).sum() / np.product(representation.shape)
    if sparsity >= 0.95:
        warnings.warn(f"The {title} representation is not diverse.")

        # Warning(f'The {title} representation is not diverse.')
        # print(representation)


def measure_embedding_diversity(model, data):
    """
    Calculate the diversity based on euclidiean minimal spanning tree
    :return:  diversity for datasets, diversity for algos
    """

    data_fwd = model.encode(data)
    z_algo = model.Z_algo

    data_tree = calc_min_eucl_spanning_tree(data_fwd)
    z_algo_tree = calc_min_eucl_spanning_tree(z_algo)

    d_diversity = sum([tup[2]["weight"] for tup in data_tree])
    z_diversity = sum([tup[2]["weight"] for tup in z_algo_tree])

    # sum of weighted edges
    return d_diversity, z_diversity


def check_wandb_exists(cfg, unique_fields: List[str]):
    flat_cfg = list(pd.json_normalize(cfg).T.to_dict().values())[0]
    query_config = {}
    for key, value in flat_cfg.items():
        if key not in unique_fields:
            continue
        query_config[key] = value

    query_config_wandb = {"config.{}".format(key): value for key, value in query_config.items()}

    query_wandb = {"state": "finished", **query_config_wandb}
    # print(query_wandb)

    api = wandb.Api()
    runs = api.runs("example/carl", query_wandb)

    found_run = False
    for run in runs:
        if cfg["env"] == "CARLPendulumEnv":
            episode = run.summary["train/episode"] if "train/episode" in run.summary else -1
            if episode != 2488:
                # run not completed
                continue
        elif cfg["env"] == "CARLAnt":
            episode = run.summary["train/episode"] if "train/episode" in run.summary else -1
            if episode < 500:
                # run not completed
                continue
        found_run = True

    return found_run
