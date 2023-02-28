import logging

import torch
import torchsort
import wandb
from tqdm import tqdm

from masif.losses.spearman import SpearmanLoss
from masif.trainer.deprec.hierarchical_transformer_trainer import (
    Trainer_Hierarchical_Transformer,
    Trainer_Hierarchical_TransformerRankingLoss,
)
from masif.trainer.deprec.lstm_trainer import Trainer_Ensemble_lstm

# A logger for this file
log = logging.getLogger(__name__)


def train_lstm(
    model,
    train_dataloader,
    test_dataloader,
    epochs,
    lr,
    trainer_type="rank_lstm",
    loss_type="spearman",
    ranking_fn=torchsort.soft_rank,
    optimizer_cls=torch.optim.Adam,
    test_lim=5,
    log_freq=10,
):
    """ """
    if loss_type == "spearman":
        loss_fn = SpearmanLoss(ranking_fn=ranking_fn)
    elif loss_type == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss_type == "l1":
        loss_fn = torch.nn.L1Loss()
    else:
        raise NotImplementedError(f"Unknown loss type {loss_type}")

    optimizer = optimizer_cls(model.parameters(), lr)

    trainer_kwargs = {
        "model": model,
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "test_lim": test_lim,
    }

    # Initialize the trainer
    if trainer_type == "rank_lstm":
        trainer = Trainer_Ensemble_lstm(**trainer_kwargs)
    elif trainer_type == "hierarchical_transformer":
        if loss_type == "spearman":
            trainer = Trainer_Hierarchical_TransformerRankingLoss(**trainer_kwargs)
        else:
            trainer = Trainer_Hierarchical_Transformer(**trainer_kwargs)
    else:
        raise NotImplementedError(f"Unknown trainer {trainer_type}")

    losses = {k: [] for k in trainer.losses}

    for e in tqdm(range(epochs)):
        # Train the model

        trainer.train(train_dataloader)

        # Evaluate the model
        trainer.evaluate(test_dataloader)

        # Take the next step
        trainer.step_next()

        for key in trainer.losses:
            losses[key].append(trainer.losses[key])

        if e % log_freq == 0:

            for key in trainer.losses:
                losses[key] = torch.stack(losses[key]).mean()

            wandb.log(losses, commit=False, step=e)

            for key in trainer.losses:
                losses[key] = []

    # return score
