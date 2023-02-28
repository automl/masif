import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from pytorch_lightning.callbacks import TQDMProgressBar


def get_sequential_model(num_hidden_layers, num_hidden_units, input_units, output_units):
    """
    Returns a sequential model with 2+num_hidden_layers linear layers.
    All linear layers (except the last one) are followed by a ReLU function.

    Parameters:
        num_hidden_layers (int): The number of hidden layers.
        num_hidden_units (int): The number of features from the hidden
            linear layers.
        input_units (int): The number of input units.
            Should be number of features.
        output_units (int): The number of output units. In case of regression task,
            it should be one.

    Returns:
        model (nn.Sequential): Neural network as sequential model.
    """

    layers = [
        nn.Linear(input_units, num_hidden_units),
        nn.ReLU(),
    ]

    for _ in range(num_hidden_layers):
        layers += [
            nn.Linear(num_hidden_units, num_hidden_units),
            nn.ReLU(),
        ]

    layers += [nn.Linear(num_hidden_units, output_units)]

    return nn.Sequential(*layers)


class PyTorchDataset(Dataset):
    """
    Since we have numpy data, it is required to convert
    them into PyTorch tensors first.
    """

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)

        self.y = torch.zeros((self.X.shape[0], 1), dtype=torch.float32)
        if y is not None:
            y = y.astype(np.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(pl.LightningModule):
    """
    Multi Layer Perceptron wrapper for pytorch lightning.
    """

    def __init__(self, sequential_model, verbose=False):
        """
        Parameters:
            sequential_model: Underlying Neural Network.
            verbose (bool): If true, the highest val accuracy is plotted after every epoch.
        """

        super().__init__()
        self.verbose = verbose
        self.highest_val_accuracy = 0

        # Important to init the weights the same way
        pl.seed_everything(0)
        self.model = sequential_model

        # (Multi-)Classification problem
        self.loss_fn = nn.CrossEntropyLoss()

    def calculate_accuracy(self, y_pred, y_test):
        """
        Calculates the accuracy for `y_pred` and `y_test`.
        """

        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)

        return acc

    def validation_step(self, batch, _):
        """
        Receives the validation data and calculates the accuracy on them.

        Parameters:
            batch: Tuple of validation data.

        Returns:
            metrics (dict): Dict with val accuracy.
        """

        X, y = batch
        y_hat = self.model(X)

        val_accuracy = self.calculate_accuracy(y_hat, y)
        return {"val_accuracy": val_accuracy}

    def validation_epoch_end(self, outputs):
        """
        Collects the outputs from `validation_step` and sets
        the average for the accuracy.

        Parameters:
            outputs: List of dicts from `validation_step`.
        """

        val_accuracy = torch.stack([o["val_accuracy"] for o in outputs]).numpy().flatten()

        val_accuracy = float(np.mean(val_accuracy))
        if val_accuracy > self.highest_val_accuracy:
            self.highest_val_accuracy = val_accuracy

        if self.verbose:
            print(f"{self.current_epoch}: {self.highest_val_accuracy}")

    def training_step(self, batch, _):
        """
        Receives the training data and calculates
        cross entropy as loss, which is used to train
        the classifier.

        Parameters:
            batch: Tuple of training data.

        Returns:
            loss (Tensor): Loss of current step.

        """

        X, y = batch
        y_hat = self.model(X)

        return self.loss_fn(y_hat, y.long())

    def configure_optimizers(self):
        """
        Configures Adam as optimizer.

        Returns:
            optimizer (torch.optim): Optimizer used internally by
                pytorch lightning.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def fit(self, X_train, y_train, X_val, y_val, learning_rate=1e-1, num_epochs=10, batch_size=8):
        """
        Fits the model with training data. Model is validated after every epoch on the validation
        data.

        Parameters:
            X_train, y_train: Training data.
            X_val, y_val: Validation data.
            learning_rate (float): Learning rate used in the optimizer.
            num_epochs (int): Number of epochs.
            batch_size (int): How many instances are used to update the weights.
        """

        pl.seed_everything(0)
        self.learning_rate = learning_rate

        self.trainer = pl.Trainer(
            num_sanity_val_steps=0,  # No validation sanity
            max_epochs=num_epochs,  # We only train one epoch
            callbacks=[TQDMProgressBar(refresh_rate=0)],  # No progress bar
            enable_checkpointing=False,
            logger=False,
            devices=1,
        )

        # Define training loader
        # `train_loader` is a lambda function, which takes batch_size as input
        train_loader = DataLoader(PyTorchDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)

        # Define validation loader
        val_loader = DataLoader(PyTorchDataset(X_val, y_val), batch_size=1, num_workers=0)

        # Train model
        self.trainer.fit(self, train_loader, val_loader)

    def predict(self, X_test):
        pytorch_dataset = PyTorchDataset(X_test)

        prediction = self.model(pytorch_dataset.X)
        y_pred_softmax = torch.log_softmax(prediction, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        return y_pred_tags
