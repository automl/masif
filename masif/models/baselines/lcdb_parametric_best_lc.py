import json
import pathlib
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt

from masif.models.baselines.lcdb_parametric_lc import ParametricLC
from masif.utils.modelinterface import ModelInterface


class BestParametricLC(ModelInterface):
    def __init__(self, budgets: List[int], restarts: int = 10):
        super().__init__()
        self.budgets = np.array(budgets, dtype=np.float64)
        self.parametric_lcs = {
            name: ParametricLC(name, budgets, restarts=restarts) for name in ParametricLC.functionals.keys()
        }

    def fit(self, x: np.ndarray, Y: np.ndarray) -> None:
        for parametric_lc in self.parametric_lcs.values():

            # saveguard for the case that the learning curve is not observed sufficiently (n<p)
            if self.max_fidelity + 1 < parametric_lc.n_parameters:
                return torch.ones(x.shape[:-1]) * float("nan")
            else:
                parametric_lc.fit(x, Y)

        # find the best parametric learning curve for each learning curve by cost
        self.costs = np.array([parametric_lc.cost for parametric_lc in self.parametric_lcs.values()])

        # find out which parametric lc is the best for which algorithm
        self.curve_name = np.array(list(self.parametric_lcs.keys()))[np.argmin(self.costs, axis=0)]
        self.curve = np.nanargmin(
            self.costs,
            axis=0,
        )  # is the curve with the lowest cost for
        # each
        # algorithm. we want this curve as predictor for the extrapolation

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.array(
            [
                parametric_lc.predict(x) if hasattr(parametric_lc, "parameters_lc") else np.ones(self.n_algos) * np.nan
                for parametric_lc in self.parametric_lcs.values()
            ]
        )
        # fancy indexing to get the final prediction of the best (the lowest cost during training) curve
        # for each algorithm.
        final_performance = predictions[self.curve, list(range(predictions.shape[1]))]

        return final_performance

    def forward(self, learning_curves: torch.Tensor, mask: torch.Tensor, **kwargs):  # ->
        # torch.Tensor:
        self.max_fidelity = int(mask.sum(dim=-1).max().item())
        self.n_algos = learning_curves.shape[1]

        # fit the parametric learning curve to the data.
        if self.training:
            self.fit(self.budgets, learning_curves[:, :, : self.max_fidelity].cpu().numpy())

        # SAFEGUARD Against untrained models (such as n < p) at the beginning of the training
        if hasattr(self, "curve"):  # safeguard for untrained models
            return torch.tensor(self.predict(self.budgets[-1])).view(1, -1)
        else:
            return torch.ones(self.n_algos).view(1, -1) * float("nan")

        # return torch.tensor(self.predict(self.budgets[-1])).view(1, -1)

    def plot_curves(self, x, y, ax):
        y_hats = self.predict(x)
        for y_, y_hat in zip(y[0], y_hats):
            ax.plot(
                x,
                y_hat,
                color="red",
                alpha=0.5,
                linewidth=1.0,
            )
            ax.plot(
                x,
                y_,
                color="grey",
                alpha=0.5,
                linewidth=0.5,
            )
        ax.set_title("Best Parametric LC for each Algorithm")
        ax.set_xlabel("Budget")
        ax.set_ylabel("Performance")
        # plt.legend()
        plt.ylim(*(y.min().item(), y.max().item()))


if __name__ == "__main__":
    from masif.data.lcbench.example_data import train_dataset
    from masif.evaluation.topkregret import TopkRegret

    # Batched learning curves (batch_size, n_algorithms, n_budgets)
    X, y = train_dataset[1]
    lc_tensor = X["learning_curves"][20:30, :]
    shape = lc_tensor.shape
    lc_tensor = lc_tensor.view(1, *shape)
    mask = torch.ones_like(lc_tensor, dtype=torch.long)
    threshold = 20
    mask[:, :, threshold:] = 0  # only partially available learning curves
    ranking = y["final_fidelity"][20:30].view(1, shape[0])

    lc_predictor = BestParametricLC(list(range(1, 52)), restarts=10)
    final_performance = lc_predictor.forward(lc_tensor, mask)

    y_hat = lc_predictor.forward(lc_tensor, mask)

    topkregret = TopkRegret(k=1)
    topkregret(y_hat, ranking)

    lc_predictor.plot_curves(x=lc_predictor.budgets, y=lc_tensor, ax=plt.gca())
    plt.show()

    topkregret = TopkRegret(k=1)
    topkregret(final_performance, ranking)
    print()

    lcs_parameters = {}
    for key, values in lc_predictor.parametric_lcs.items():
        lcs_parameters[key] = values.parameters_lc.tolist()

    with open(str(pathlib.Path(__file__).resolve().parent / "lcs_parameters.json"), "w") as f:
        json.dump(lcs_parameters, f)
