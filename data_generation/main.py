import os
import logging

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import openml
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from py_experimenter.result_processor import ResultProcessor

from py_experimenter.experimenter import PyExperimenter

import time

from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.base import clone


def get_cc18_datasets():
    cc18_benchmark_suite = openml.study.get_suite(99)
    return openml.datasets.get_datasets(cc18_benchmark_suite.data)


def evaluate_algorithm_on_dataset(openml_dataset, algorithm, budget, number_of_splits, seed):
    base_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("classifier", algorithm),
        ]
    )

    X, y, categorical_indicator, attribute_names = openml_dataset.get_data(
        dataset_format="dataframe", target=openml_dataset.default_target_attribute
    )
    X = X.to_numpy()
    y = y.to_numpy()

    stratiefiedKFold = StratifiedKFold(n_splits=number_of_splits, random_state=seed, shuffle=True)
    stratified_X_trains = list()
    stratified_y_trains = list()
    stratified_X_vals = list()
    stratified_y_vals = list()
    stratified_X_tests = list()
    stratified_y_tests = list()

    for train_index, test_index in stratiefiedKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        stratified_X_trains.append(X_train)
        stratified_X_vals.append(X_val)
        stratified_X_tests.append(X_test)
        stratified_y_trains.append(y_train)
        stratified_y_vals.append(y_val)
        stratified_y_tests.append(y_test)

    train_accuracies = list()
    val_accuracies = list()
    test_accuracies = list()
    times = list()

    for fold in range(number_of_splits):
        # sample down to the correct size
        X_subsampled, y_subsampled = resample(
            stratified_X_trains[fold],
            stratified_y_trains[fold],
            random_state=fold,
            n_samples=int(budget * stratified_X_trains[fold].shape[0]),
        )

        try:
            pipeline = clone(base_pipeline)
            # train pipeline
            start_time = time.time()
            pipeline.fit(X_subsampled, y_subsampled)
            elapsed_time_in_seconds = time.time() - start_time
            times.append(elapsed_time_in_seconds)

            # predict and compute accuracy on train set
            predictions = pipeline.predict(X_subsampled)
            accuracy = accuracy_score(y_subsampled, predictions)
            train_accuracies.append(accuracy)

            # predict and compute accuracy on validation set
            predictions = pipeline.predict(stratified_X_vals[fold])
            accuracy = accuracy_score(stratified_y_vals[fold], predictions)
            val_accuracies.append(accuracy)

            # predict and compute accuracy on test set
            predictions = pipeline.predict(stratified_X_tests[fold])
            accuracy = accuracy_score(stratified_y_tests[fold], predictions)
            test_accuracies.append(accuracy)
        except:
            train_accuracies.append(np.nan)
            val_accuracies.append(np.nan)
            test_accuracies.append(np.nan)

    average_train_accuracy_over_folds = np.asarray(train_accuracies).mean()
    average_val_accuracy_over_folds = np.asarray(val_accuracies).mean()
    average_test_accuracy_over_folds = np.asarray(test_accuracies).mean()
    average_training_time = np.asarray(times).mean()

    print(
        f"{openml_dataset.name}:: budget: {budget}, train_accuracy: {average_train_accuracy_over_folds}, val_accuracy: {average_val_accuracy_over_folds}, test_accuracy: {average_test_accuracy_over_folds}, time: {average_training_time}s"
    )

    evaluation_result = dict()
    evaluation_result["pipeline"] = str(pipeline)
    evaluation_result["average_train_accuracy"] = average_train_accuracy_over_folds
    evaluation_result["train_accuracy_per_fold"] = train_accuracies
    evaluation_result["average_val_accuracy"] = average_val_accuracy_over_folds
    evaluation_result["val_accuracy_per_fold"] = val_accuracies
    evaluation_result["average_test_accuracy"] = average_test_accuracy_over_folds
    evaluation_result["test_accuracy_per_fold"] = test_accuracies
    evaluation_result["average_train_time_s"] = average_training_time
    evaluation_result["train_time_s_per_fold"] = times

    return evaluation_result


def run_experiment(keyfields: dict, result_processor: ResultProcessor, custom_fields: dict):
    # Extracting given parameters
    number_of_splits = int(custom_fields["number_of_splits"])

    seed = int(keyfields["seed"])
    np.random.seed(seed)

    dataset_id = keyfields["dataset_id"]
    budget = keyfields["budget"]

    classifier = eval(keyfields["classifier"])

    dataset = openml.datasets.get_dataset(dataset_id)

    evaluation_result = evaluate_algorithm_on_dataset(dataset, classifier, budget, number_of_splits, seed)

    # Write intermediate results to database
    result_processor.process_results(evaluation_result)


if __name__ == "__main__":
    experimenter = PyExperimenter()
    experimenter.fill_table_from_config()
    experimenter.execute(run_experiment, max_experiments=-1, random_order=True)
