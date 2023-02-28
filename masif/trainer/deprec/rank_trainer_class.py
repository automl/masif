import numpy as np
import torch
import wandb
from tqdm import tqdm


class Trainer_Rank:
    def __init__(self):
        self.step = 0
        self.losses = {
            # 'ranking_loss': 0
        }

    def train(
        self,
        model,
        loss_fn,
        train_dataloader,
        test_dataloader,
        epochs,
        lr=0.001,
        log_wandb=True,
        slice_index=-1,
    ):

        optimizer = torch.optim.Adam(model.parameters(), lr)
        for e in tqdm(range(int(epochs))):

            losses = []
            for i, data in enumerate(train_dataloader):
                # Dataset meta features and final  slice labels
                D0 = data[0].to(model.device)
                labels = data[1][0, slice_index].reshape(1, -1).to(model.device)

                # calculate embedding
                D0_fwd = model.forward(D0)

                # Calculate the loss
                loss = loss_fn(
                    pred=D0_fwd,
                    target=labels,
                )

                # gradient step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().item())

            test_losses = []
            for i, data in enumerate(test_dataloader):
                # Dataset meta features and final  slice labels
                D0 = data[0].to(model.device)
                labels = data[1][0, slice_index].reshape(1, -1).to(model.device)

                # calculate embedding
                D0_fwd = model.forward(D0)

                # Calculate the loss
                loss = loss_fn(
                    pred=D0_fwd,
                    target=labels,
                )

                test_losses.append(loss.detach().item())

            # log losses
            self.losses[f"{slice_index}/ranking_loss"] = np.mean(test_losses)

            if log_wandb:
                wandb.log(self.losses, commit=False, step=self.step)

            self.step += 1

        return {"loss": self.losses[f"{slice_index}/ranking_loss"], "step": self.step - 1}
