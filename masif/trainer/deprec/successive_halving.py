import pandas as pd
import torch
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

enable_halving_search_cv


def train_evaluate(model: HalvingGridSearchCV, **kwargs):
    model.fit(X=list(range(1000)), y=list(range(1000)))  # X & y must be larger than n_slices
    # fixme: make sure CV' s n_split don't fuck with us - since we want to lookup all of the
    #  datasets at once. i.e. be aware of _BaseKFold.n_splits_ !

    # print(model.best_estimator_)
    pd.DataFrame.from_dict(model.cv_results_)

    # get runhistory
    runhistory = pd.DataFrame.from_dict(model.cv_results_)

    relevant_runhistory = runhistory[["iter", "param_algo_id", "mean_test_score"]]

    # compute at which level the algorithm was terminated ----------------------
    # compute which algos where evaluated at that level
    survivors = relevant_runhistory.groupby("iter").agg({"param_algo_id": set})["param_algo_id"].tolist()

    # compute the number of algos that survived at each level
    surv = list(reversed(survivors))
    for i, s in enumerate(surv):
        for j, s2 in enumerate(surv[i + 1 :]):
            surv[j + i + 1] = s2 - s

    survivors = list(reversed(surv))

    # sanity check:
    if not set.union(*survivors) == set(relevant_runhistory["param_algo_id"].unique()):
        raise ValueError("survivors are not consistent with the available algorithms")

    # get ordinal ranking based on termination
    # keep only survivors at respective level
    level_dfs = [
        df.loc[df["param_algo_id"].isin(s)] for (_, df), s in zip(relevant_runhistory.groupby("iter"), survivors)
    ]
    terminated = pd.concat(level_dfs)

    # sanity check:
    if not set(terminated["param_algo_id"]) == set(relevant_runhistory["param_algo_id"].unique()):
        raise ValueError("survivors are not consistent with the available algorithms")

    # break the ties based on the observed performances
    terminated.sort_values(["iter", "mean_test_score"], ascending=False, inplace=True)
    terminated.reset_index(drop=True, inplace=True)

    rank = pd.DataFrame.from_dict({"rank": terminated.index, "algo_id": terminated["param_algo_id"]})

    rank.sort_values(["algo_id"], inplace=True)
    return torch.tensor(rank["rank"])  # 0 entry has the highest ranking

    # compute kendal's tau to deal with ties?
