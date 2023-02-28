import logging
from functools import partial
from typing import Callable, Dict, Optional, Union

import torch.nn as nn
import torch.optim
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from masif.utils.masking import mask_lcs_to_max_fidelity

log = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """

        :param model: torch.nn.Module
        :param optimizer: A partially instantiated object accepting model.parameters()

        """
        self._step = 0
        self.model = model
        self.device = self.model.device

        self.optimizer = None

        if isinstance(self.model, nn.Module) and not hasattr(self.model, "no_opt"):
            # partial instantiation from hydra!
            self.optimizer = optimizer(self.model.parameters())

    @property
    def step(self):
        return self._step

    def to_device(
        self,
        input,
    ):
        for k, v in input.items():
            input[k] = v.to(self.device).float()
        return input

    def train(self, train_loader, epochs, loss_fn, log_freq=1) -> None:
        """define one epoch of training"""
        # TODO: incorporate callbacks to the training loop (before and after each epoch, and )
        #  at every (k-th) step?
        for _, (X, y) in enumerate(train_loader):
            X = self.to_device(X)
            y = self.to_device(y)

            if self.optimizer:
                self.optimizer.zero_grad()

            y_hat = self.model(**X)

            if y_hat is None:
                loss = 0
                continue

            # if torch.isnan(y_hat).any():
            #     print(
            #         f"y_hat: {y_hat}",
            #         f'y: {y["final_fidelity"]}',
            #         f'lcs: {X["learning_curves"]}',
            #         f'mask: {X["mask"]}',
            #         f'Dataset metafeatures {X["dataset_meta_features"]}',
            #     )
            #     print()

            if not hasattr(self.model, "no_opt"):
                loss = loss_fn(y_hat, y["final_fidelity"])
                # print(X, y, y_hat, loss)
                loss.backward()

                # FIXME: y needs to be explicit or have a strong conventioN
                if self.optimizer:
                    self.optimizer.step()
            else:
                loss = 0

        # Log the training loss at the end of the epoch
        if self.step % log_freq == 0:
            wandb.log({"trainingloss": loss}, step=self.step)

        self._step += 1

    def validate(self, valid_loader, valid_loss_fn, function_name):
        """evaluate the model on the validationset after epoch ends for a single validation
        function"""
        if valid_loader is not None:
            with torch.no_grad():
                losses = torch.zeros(len(valid_loader))
                for i, data in enumerate(valid_loader):
                    X, y = data
                    self.to_device(X)
                    self.to_device(y)

                    y_hat = self.model.forward(**X)
                    losses[i] = valid_loss_fn(y_hat, y["final_fidelity"])

                wandb.log({f"Validation: {function_name}": losses.mean()}, step=self.step)

    def test(
        self,
        test_loader,
        test_loss_fns,
    ):
        """
        Slice Evaluation Protocol;
        Evaluate the model on the testset after training is done.
        1. Mask the testset to the (currently) maximum fidelity (minimum is 1,
           giving at least on e)
        2. Evaluate the model on the masked testset & record the loss
        3. Repeat 1 & 2 for all available fidelities in the test set

        """
        prediction = []
        ground_truth = []
        with torch.no_grad():
            max_fidelity = test_loader.dataset.learning_curves.shape[-1]
            for fidelity in tqdm(range(max_fidelity), desc="Fidelity"):

                # fixme: figure out how wandb can recieve the actual fidelity label.

                test_loader.dataset.masking_fn = partial(mask_lcs_to_max_fidelity, max_fidelity=fidelity)

                # fixme: make one fwd pass and compute all validation losses on the fwd pass
                #  to drastically reduce the number of fwd passes!
                losses = {k: torch.zeros(len(test_loader)) for k in test_loss_fns.keys()}

                pred_i = []
                ground_truth_i = []

                for i, (X, y) in enumerate(test_loader):
                    self.to_device(X)
                    self.to_device(y)

                    for j, (fn_name, fn) in enumerate(test_loss_fns.items()):
                        if isinstance(fn, DictConfig):  # fixme: can we remove this?
                            fn = instantiate(fn)

                        if hasattr(self.model, "no_opt"):
                            self.model.training = False

                        if j == 0:  # for all those baselines that don't have a online training
                            # phase (parametric LC & Satzilla)
                            self.model.train()

                        y_hat = self.model.forward(**X)

                        if j == 0:
                            self.model.eval()

                        # Successive halving will inf for zero available fidelities.
                        # Some test loss functions may take an issue with that.
                        # This is a save-guard to prevent this (intended) behaviour.
                        try:
                            if hasattr(self.model, "no_opt"):
                                losses[fn_name][i] = fn(y_hat, y["final_fidelity"][0])
                            else:
                                losses[fn_name][i] = fn(y_hat, y["final_fidelity"])
                        except Exception as e:
                            log.error(f"Error in test loss fn {fn_name}:\n{e}")
                            losses[fn_name][i] = float("nan")

                    pred_i.append(y_hat)
                    if hasattr(self.model, 'no_opt'):
                        ground_truth_i.append(y["final_fidelity"][0])
                    else:
                        ground_truth_i.append(y["final_fidelity"])

                prediction.append(torch.cat(pred_i, 0).unsqueeze(-1))
                ground_truth.append(torch.cat(ground_truth_i, 0).unsqueeze(-1))

                for fn_name, fn in test_loss_fns.items():
                    wandb.log({f"Test, Slice Evaluation: {fn_name}": losses[fn_name].mean(), "fidelity": fidelity})
        prediction = torch.cat(prediction, dim=-1) # prediction has shape of [num_datasets, n_algos, n_fidelities]
        ground_truth = torch.cat(ground_truth, dim=-1)

        return prediction, ground_truth

    def run(
        self,
        train_loader: torch.utils.data.DataLoader = None,
        valid_loader: torch.utils.data.DataLoader = None,
        test_loader: torch.utils.data.DataLoader = None,
        epochs: int = 0,
        log_freq: int = 5,
        train_loss_fn: Union[Callable, DictConfig] = None,
        valid_loss_fns: Dict[str, Callable] = None,
        test_loss_fns: Dict[str, Callable] = None,
    ):
        """
        Main loop including training & test evaluation, all of which report to wandb

        :returns None, but tracks using wandb both the trainloss all the validation losses
        """
        # Class & functional interface
        if isinstance(train_loss_fn, DictConfig):
            train_loss_fn = instantiate(train_loss_fn)

        for epoch in tqdm(range(epochs), desc="Training epochs"):
            if isinstance(self.model, nn.Module):
                self.train(train_loader, epoch, train_loss_fn)

            # move this  parameter hist tracker to a callback?
            # for k, t in self.model.state_dict().items():
            #     wandb.log({k: wandb.Histogram(torch.flatten(t))}, step=self.step)

            # validation loss during training
            if valid_loss_fns is not None and self.step % log_freq == 0:
                self.model.eval()
                for fn_name, fn in valid_loss_fns.items():
                    if isinstance(fn, DictConfig):  # fixme: can we remove this?
                        fn = instantiate(fn)

                    if hasattr(self.model, "no_opt"):
                        self.model.training = False

                    self.validate(valid_loader, fn, fn_name)

                self.model.train()

        # Test loss for comparison of selectors
        self.model.eval()

        prediction, ground_truth = self.test(test_loader, test_loss_fns)
        return prediction, ground_truth
