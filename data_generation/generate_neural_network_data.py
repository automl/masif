import torch

import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*MPS available.*")
warnings.filterwarnings("ignore", ".*Global seed.*")


import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

from py_experimenter.result_processor import ResultProcessor
from py_experimenter.experimenter import PyExperimenter
from neural_network import get_sequential_model, MLP

import openml
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import time
import os


def evaluate_algorithm_on_dataset(
    openml_dataset, budget, number_of_splits, num_hidden_layers, num_hidden_units, learning_rate, max_num_epochs, seed
):

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
    label_encoders = list()

    for train_index, test_index in stratiefiedKFold.split(X, y):
        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("standard_scaler", StandardScaler(with_mean=False)),
            ]
        )
        label_encoder = LabelEncoder()
        label_encoders.append(label_encoder)

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        X_train = pipeline.fit_transform(X=X_train).todense()
        y_train = label_encoder.fit_transform(y=y_train)
        X_val = pipeline.transform(X=X_val).todense()
        y_val = label_encoder.transform(y=y_val)
        X_test = pipeline.transform(X=X_test).todense()
        y_test = label_encoder.transform(y=y_test)

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

        X_train = stratified_X_trains[fold]
        X_val = stratified_X_vals[fold]

        y_train = stratified_y_trains[fold]
        y_val = stratified_y_vals[fold]

        y_test = stratified_y_tests[fold]

        input_units = X_train.shape[1]
        output_units = len(np.unique(list(y_train) + list(y_val) + list(y_test)))
        network_architecture = get_sequential_model(num_hidden_layers, num_hidden_units, input_units, output_units)
        network_model = MLP(network_architecture, verbose=False)

        try:
            # train pipeline
            start_time = time.time()
            network_model.fit(
                X_train, y_train, X_val, y_val, learning_rate=learning_rate, num_epochs=int(budget * max_num_epochs)
            )
            elapsed_time_in_seconds = time.time() - start_time
            times.append(elapsed_time_in_seconds)

            label_encoder = label_encoders[fold]

            # predict and compute accuracy on train set
            predictions = network_model.predict(X_train).detach().numpy()
            accuracy = accuracy_score(y_train, predictions)
            train_accuracies.append(accuracy)

            # predict and compute accuracy on validation set
            predictions = network_model.predict(stratified_X_vals[fold]).detach().numpy()
            accuracy = accuracy_score(stratified_y_vals[fold], predictions)
            val_accuracies.append(accuracy)

            # predict and compute accuracy on test set
            predictions = network_model.predict(stratified_X_tests[fold]).detach().numpy()
            accuracy = accuracy_score(stratified_y_tests[fold], predictions)
            test_accuracies.append(accuracy)
        except:
            train_accuracies.append(np.nan)
            val_accuracies.append(np.nan)
            test_accuracies.append(np.nan)
            raise

    average_train_accuracy_over_folds = np.asarray(train_accuracies).mean()
    average_val_accuracy_over_folds = np.asarray(val_accuracies).mean()
    average_test_accuracy_over_folds = np.asarray(test_accuracies).mean()
    average_training_time = np.asarray(times).mean()

    print(
        f"{openml_dataset.name}:: budget: {budget}, train_accuracy: {average_train_accuracy_over_folds}, val_accuracy: {average_val_accuracy_over_folds}, test_accuracy: {average_test_accuracy_over_folds}, time: {average_training_time}s"
    )

    evaluation_result = dict()
    evaluation_result["pipeline"] = str(learning_rate)
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
    num_hidden_layers = int(custom_fields["number_of_hidden_layers"])
    num_hidden_units = int(custom_fields["number_of_hidden_units"])
    max_num_epochs = int(custom_fields["maximum_number_of_epochs"])

    seed = int(keyfields["seed"])
    np.random.seed(seed)

    dataset_id = int(keyfields["dataset_id"])
    budget = keyfields["budget"]

    learning_rate = keyfields["learning_rate"]

    dataset = openml.datasets.get_dataset(dataset_id)

    evaluation_result = evaluate_algorithm_on_dataset(
        dataset, budget, number_of_splits, num_hidden_layers, num_hidden_units, learning_rate, max_num_epochs, seed
    )

    # Write intermediate results to database
    result_processor.process_results(evaluation_result)


if __name__ == "__main__":
    torch.set_num_threads(1)
    experimenter = PyExperimenter(config_file=os.path.join("config", "neural_network_configuration.cfg"))
    experimenter.fill_table_from_config()
    experimenter.execute(run_experiment, max_experiments=-1, random_order=True)
