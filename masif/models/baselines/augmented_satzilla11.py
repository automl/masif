from typing import Tuple, Optional
import bisect
from collections import Counter

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from masif.models.baselines.satzilla11 import SATzilla11


class AugmentedSATzilla11(SATzilla11):
    def __init__(self, max_fidelity, device="cpu",  n_estimators: int = 2, max_features: str = 'log2', n_jobs: int = 5, random_state: int = 0, aug_fidelity: Optional[Tuple[int]] = (0, 10)):
        """
        a SATzilla11 algorithm selector that also consider the fidelity values as dataset meta features.
        Therefore, this model neds to work with masif.data.dataset_join_Dmajor.Dataset_Join_Dmajor
        aug_fidelity: which fidelity values from the learing curves will be attached to the features
        """
        super(AugmentedSATzilla11, self).__init__(max_fidelity, device, n_estimators, max_features, n_jobs, random_state)
        self.aug_fidelity = aug_fidelity
        self._fitted = False

    def forward(self, dataset_meta_features, learning_curves, mask, *args, **kwargs):
        if self.training:
            return self.fit(dataset_meta_features, learning_curves)
        else:
            return self.predict(dataset_meta_features, learning_curves)

    def fit(self, dataset_meta_features, learning_curves):
        if self._fitted:
            return
        fidelity = learning_curves[:, :, self.max_fidelity]
        self._num_algorithms = np.shape(fidelity)[-1]
        self._algo_ids = np.arange(0, self._num_algorithms, 1)

        # lc_argsort = torch.argsort(learning_curves, 1).numpy() / self._num_algorithms
        lc_normalized = (learning_curves / torch.std(learning_curves, 1, keepdim=True)).numpy()

        lc_features = lc_normalized

        features = dataset_meta_features.numpy()
        performances = fidelity.numpy()
        learning_curves = learning_curves.numpy()

        # create and fit rfcs' for all pairwise comparisons between two algorithms
        self._pairwise_indices = [
            (i, j) for i in range(self._num_algorithms) for j in range(i + 1, self._num_algorithms)
        ]

        for (i, j) in tqdm(self._pairwise_indices):
            # determine pairwise target, initialize models and fit each RFC wrt. instance weights
            pair_target = self._get_pairwise_target((i, j), performances)
            sample_weights = self._compute_sample_weights((i, j), performances)

            # account for nan values (i.e. ignore pairwise comparisons that involve an algorithm run violating
            # the cutoff if the config specifies 'ignore_censored'), hence set all respective weights to 0
            sample_weights = np.nan_to_num(sample_weights)

            # There are different ways of attaching the lc values to the feautre, here I simply take the most simple
            # approach: take the difference between these values
            if self.aug_fidelity is not None:
                lc_value_i = lc_features[:, i, self.aug_fidelity]
                lc_value_j = lc_features[:, j, self.aug_fidelity]

                features_augmented = np.concatenate([features, lc_value_i - lc_value_j], axis=-1)
            else:
                features_augmented = features

            # TODO: how to set the remaining hyperparameters?
            self._models[(i, j)] = RandomForestClassifier(**self.rfc_kwargs)  # TODO set random state to the torch seed
            self._models[(i, j)].fit(features_augmented, pair_target, sample_weight=sample_weights)
        self._fitted = True

    def predict(self, dataset_meta_features, learning_curves):

        batch_features = dataset_meta_features
        lc_argsort = torch.argsort(learning_curves, 1).numpy() / self._num_algorithms
        lc_normalized = (learning_curves / torch.std(learning_curves, 1, keepdim=True)).numpy()
        lc_feature = lc_normalized

        selections = []
        # Selection an algorithm per task in the test-set
        for features, lc in zip(batch_features, lc_feature):

            assert features.ndim == 1, "`features` must be one dimensional"
            features = np.expand_dims(features, axis=0)
            predictions = {}
            for (i, j), rfc in self._models.items():
                if self.aug_fidelity is not None:
                    lc_value_i = np.expand_dims(lc[i, self.aug_fidelity], axis=0)
                    lc_value_j = np.expand_dims(lc[j, self.aug_fidelity], axis=0)
                    features_augmented = np.concatenate([features, lc_value_i - lc_value_j], axis=-1)
                else:
                    features_augmented = features

                predictions[(i, j)] = rfc.predict(features_augmented).item()

            counter = Counter(predictions.values())

            for id in self._algo_ids:
                if id not in counter.keys():
                    counter[id] = 0

            # TODO revisit the tie breaking procedure
            # ranking = []

            # for i in range(len(counter.most_common())):

            #     key, val  = counter.most_common()[i]

            #     if counter.most_common()[i+1][1] == val:

            #         ties  = [key]

            #         k = i+1
            #         tie  = True
            #         while tie:
            #             ties.append(counter.most_common()[k][0])

            #             k += 1
            #             if counter.most_common()[k][1] != val:
            #                 tie = False

            #         print(ties)
            #         [ranking.append(a) for a,_ in self._break_ties(ties, predictions)]
            #     else:
            ranking = [key for key, _ in counter.most_common()]

            selections.append(ranking)

        return torch.Tensor(selections)


class MultiAugmentedSATzilla11(AugmentedSATzilla11):
    def __init__(self, max_fidelity, device="cpu",  n_estimators: int = 2, max_features: str = 'log2', n_jobs: int = 5, random_state: int = 0,aug_fidelity: Tuple[int] = (0, 10, 20, 30, 40, 50)):
        """
        a SATzilla11 algorithm selector that also consider the fidelity values as dataset meta features.
        Therefore, this model neds to work with masif.data.dataset_join_Dmajor.Dataset_Join_Dmajor
        aug_fidelity: which fidelity values from the learing curves will be attached to the features
        """
        super(MultiAugmentedSATzilla11, self).__init__(max_fidelity, device, n_estimators, max_features, n_jobs, random_state, aug_fidelity)
        aug_fidelity = list(aug_fidelity)
        aug_fidelity.sort()
        
        model_init_kwargs = {'max_fidelity':max_fidelity,
                             'device': device,
                             'n_estimators':n_estimators,
                             'max_features': max_features,
                             'n_jobs': n_jobs,
                             'random_state': random_state}
        self.models = [AugmentedSATzilla11(aug_fidelity=None, **model_init_kwargs)]
        for i, fid in enumerate(aug_fidelity):
            self.models.append(AugmentedSATzilla11(aug_fidelity=tuple(aug_fidelity[: i + 1]), **model_init_kwargs))

    def forward(self, dataset_meta_features, learning_curves, mask, *args, **kwargs):
        if self.training:
            for model in self.models:
                res = model.forward(dataset_meta_features, learning_curves, mask)
            return res
        else:
            n_observed_values = torch.sum(mask.bool(), dim=-1)
            length_min = torch.min(n_observed_values)

            # we assume that the more fidelity values a model receive, the better it will be
            if length_min == 0:
                model = self.models[0]
            else:
                model_idx = bisect.bisect_right(self.aug_fidelity, length_min - 1)
                old_model_idx = bisect.bisect_right(self.aug_fidelity, length_min - 2)

                if model_idx == old_model_idx and hasattr(self, "last_res"):
                    return self.last_res
                model = self.models[model_idx]

            res = model.forward(dataset_meta_features, learning_curves, mask)
            self.last_res = res
            return res

    def train(self):
        self.training = True
        for model in self.models:
            model.train()

    def eval(self):
        self.training = False
        if hasattr(self, "no_opt"):
            delattr(self, "no_opt")  # FIXME: do we also need that for the base class?
        for model in self.models:
            model.eval()
