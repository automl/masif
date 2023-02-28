# TODO code source adaptation mention


import pickle

from aslib_scenario.aslib_scenario import ASlibScenario
import pandas as pd
import numpy as np
from matplotlib.pyplot import sci
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import logging
from sklearn.base import clone

import torch
import torch.nn as nn
import wandb


class MultiClassAlgorithmSelector(nn.Module):
    def __init__(self):

        super(MultiClassAlgorithmSelector, self).__init__()

        self.scikit_classifier = RandomForestClassifier(n_jobs=1, n_estimators=100)

        # attributes
        self.trained_model = None
        self.num_algorithms = 0

    def forward(self, dataset_meta_features, fidelity):
        self.fit(dataset_meta_features.numpy()[0], fidelity.numpy()[0])

        self.predict(dataset_meta_features.numpy()[0])

        pass

    def fit(self, dataset_meta_features, fidelity):
        self._num_algorithms = np.shape(fidelity)[-1]
        self._algo_ids = np.arange(0, self._num_algorithms, 1)

        features = dataset_meta_features.numpy()[0]
        performances = fidelity.numpy()[0]

        X_train, y_train = self.construct_dataset(features, performances)

        self.trained_model = clone(self.scikit_classifier)
        self.trained_model.set_params(random_state=0)  # TODO change to torch seed for MCTS folds

        self.trained_model.fit(X_train, y_train)

    def predict(self, dataset_meta_features, fidelity):

        batch_features = dataset_meta_features.numpy()[0]

        predicted_performances = []

        for features in batch_features:
            X_test = np.reshape(dataset_meta_features, (1, len(dataset_meta_features)))

            prediction = self.trained_model.predict(X_test)[0]

            predicted_performances.append(prediction)

        return np.asarray(predicted_performances)

    def construct_dataset(self, instance_features, performances):

        num_instances = len(performances)
        best_algorithm_ids = list()

        for i in range(0, num_instances):
            # Get the maximum  utilit from the performance
            # NOTE if the performances representat runtime, then the array
            # should be modified to 1/runtime for this logic to work

            max_utility = np.nanmax(performances[i])
            best_algorithm_id = np.nonzero(performances[i] == max_utility)[0][0]

            best_algorithm_id = np.argmax(performances[i])
            best_algorithm_ids.append(best_algorithm_id)

        return instance_features, np.asarray(best_algorithm_ids)


if __name__ == "__main__":

    x = MultiClassAlgorithmSelector()

    print("Hallelujah")
