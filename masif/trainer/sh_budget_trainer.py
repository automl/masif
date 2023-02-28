import logging
from functools import partial
from typing import Optional

import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from masif.models.baselines.successive_halving import SuccessiveHalving
from masif.trainer import BaseTrainer
from masif.utils.masking import mask_lcs_to_max_fidelity

log = logging.getLogger(__name__)


class SHBudgetTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        sh_model: SuccessiveHalving,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        super().__init__(model, optimizer)
        self.sh_model = sh_model

    def test(self, test_loader, test_loss_fns):
        """
        Using Slice evaluation, let SH determine the schedule,
        evaluate the same schedule (mask) for the transformer model and
        compare their regrets (tracking both individually and the transformer's
        improvement over sh's expectation.
        """
        prediction = []
        ground_truth = []
        with torch.no_grad():
            max_fidelity = test_loader.dataset.learning_curves.shape[-1]
            for fidelity in tqdm(range(max_fidelity), desc="Fidelity"):
                losses_sh = {k: torch.zeros(len(test_loader)) for k in test_loss_fns.keys()}
                losses_model = {k: torch.zeros(len(test_loader)) for k in test_loss_fns.keys()}

                test_loader.dataset.masking_fn = partial(mask_lcs_to_max_fidelity, max_fidelity=fidelity)
                pred_i = []
                ground_truth_i = []

                for i, (X, y) in enumerate(test_loader):
                    mask = X.pop("mask")
                    print(mask)
                    self.to_device({"mask": mask})
                    self.to_device(X)
                    self.to_device(y)

                    for j, (fn_name, fn) in enumerate(test_loss_fns.items()):
                        if isinstance(fn, DictConfig):  # fixme: can we remove this?
                            fn = instantiate(fn)

                        y_hat_sh = self.sh_model.forward(**X, mask=mask)

                        # collect sh's budget mask (already conditioned on max fidelity)
                        mask_sh = self.sh_model.observed_mask
                        self.to_device({"mask": mask_sh})

                        # predict model based on that mask
                        y_hat_model = self.model.forward(**X, mask=mask_sh)
                        print(y_hat_model, y_hat_sh)

                        # Successive halving will inf for zero available fidelities.
                        # Some test loss functions may take an issue with that.
                        # This is a save-guard to prevent this (intended) behaviour.
                        try:
                            losses_sh[fn_name][i] = fn(y_hat_sh, y["final_fidelity"])
                            # print(losses_sh[fn_name][i])
                        except Exception as e:
                            log.error(f"Error in test loss fn {fn_name}:\n{e}")
                            losses_sh[fn_name][i] = float("nan")

                        try:
                            losses_model[fn_name][i] = fn(y_hat_model, y["final_fidelity"])
                            # print(losses_model[fn_name][i])
                        except Exception as e:
                            log.error(f"Error in test loss fn {fn_name}:\n{e}")
                            losses_model[fn_name][i] = float("nan")

                    pred_i.append(y_hat_sh)
                    ground_truth_i.append(y["final_fidelity"])

                prediction.append(torch.cat(pred_i, 0).unsqueeze(-1))
                ground_truth.append(torch.cat(ground_truth_i, 0).unsqueeze(-1))

                # write out the average losses for each loss function out to wandb
                for fn_name, fn in test_loss_fns.items():
                    wandb.log(
                        {f"Test, Slice Evaluation sh: {fn_name}": losses_sh[fn_name].mean(), "fidelity": fidelity}
                    )

                    wandb.log(
                        {
                            f"Test, Slice Evaluation model on sh's mask: {fn_name}": losses_model[fn_name].mean(),
                            "fidelity": fidelity,
                        }
                    )

                    wandb.log(
                        {
                            f"Test, Slice Evaluation improvement of model over sh both on sh's mask:"
                            f" {fn_name}": losses_sh[fn_name].mean() - losses_model[fn_name].mean(),
                            "fidelity": fidelity,
                        }
                    )

        prediction = torch.cat(prediction, dim=-1) # prediction has shape of [num_datasets, n_algos, n_fidelities]
        ground_truth = torch.cat(ground_truth, dim=-1)

        return prediction, ground_truth
