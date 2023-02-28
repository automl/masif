import logging
import operator
from functools import reduce
from typing import List, Optional

import torch

from masif.models.baselines.successive_halving import SuccessiveHalving

logger = logging.getLogger(__name__)


# logger.setLevel('DEBUG')


class SH_masif(SuccessiveHalving):
    def __init__(self, model, budgets: List, eta: int = 2, device: str = "cpu", budget_type="additive"):
        """
        This class assumes a single batch (i.e. a single dataset!)
        :param budgets: List[int] of budget 'labels' to be used in the successive halving run
        (actual budget values available in the learning curve tensor provided during forward).
        :param eta: int, the reduction factor of the budget in each successive halving run.

        model: the predictor for the learning curves, based of which the termination of
        low performers will be done.
        """
        super().__init__(budgets, eta, device, budget_type)
        self.model = model

    def __repr__(self):
        return (
            f"SH_masif(model={self.model}," f"budgets={self.budgets}, eta={self.eta}, budget_type={self.budget_type})"
        )

    def forward(
        self,
        learning_curves: torch.Tensor,
        mask: torch.Tensor,
        cost_curves: Optional[torch.Tensor] = None,
        **kwargs,  # to capture all the other data, competitors might have available
    ):
        """
        SH execution.
        :return: torch.Tensor (float), ranking of the algorithms.
        """

        # batch = learning_curves.shape[0]
        n_algos = learning_curves.shape[1]
        algos_idx = torch.arange(n_algos)

        # running variables
        n_dead = 0
        rankings_idx = torch.zeros((n_algos,))
        rankings_values = torch.zeros((n_algos,))
        survivors = torch.ones((n_algos), dtype=torch.int64)
        self.elapsed_time = torch.zeros(1)

        # required by sh_budget_trainer
        self.observed_mask = torch.zeros_like(mask, dtype=torch.float32)

        # is attribute for debugging (assertion)
        self.schedule_index = self.plan_budgets(self.budgets, mask)

        if self.schedule_index.shape[0] == 0:
            # no fidelity information available (safeguard)
            return torch.tensor([torch.inf for _ in range(n_algos)]).view(1, -1)

        for level, budget in enumerate(self.schedule_index.tolist(), start=1):
            # number of survivors in this round
            k = max(1, int(n_algos / self.eta**level))

            # available fidelity for each algorithm
            self.observed_mask[:, survivors == 1, : (budget + 1)] = 1  # fixme: is this correct

            # visited fidelities for each algorithm respectively
            # self.observed_mask[:, survivors == 1, budget] = 1

            # final round to break all ties?
            if level == len(self.schedule_index):
                k = n_algos - n_dead

            logger.debug(f"Budget {budget}: {k} will survivors out of {sum(survivors)}")
            logger.debug(f"Survivors: {survivors}")

            if sum(survivors) == 0:
                break

            self.update_costs(cost_curves, budget, survivors)

            # dead in relative selection (index requires adjustment)
            dead = self.kill_low_performers(
                learning_curves=learning_curves, k=k, survivors=survivors, budget=budget, **kwargs
            )
            # translate to the algorithm index
            new_dead_algo = algos_idx[survivors == 1][dead.indices]
            logger.debug(f"Dead: {new_dead_algo}")

            # change the survivor flag
            survivors[new_dead_algo] = 0
            logger.debug(f"Deceased' performances: " f"{learning_curves[:, :, budget][0, new_dead_algo]}")
            logger.debug(f"Survivors' performances: {learning_curves[:, :, budget] * survivors}")

            # BOOKKEEPING: add the (sorted by performance) dead to the ranking
            rankings_idx[n_dead : n_dead + dead.indices.shape[-1]] = new_dead_algo
            rankings_values[n_dead : n_dead + dead.indices.shape[-1]] = dead.values
            n_dead += dead.indices.shape[-1]

            logger.debug(f"rankings: {rankings_idx}")

        ranking = torch.tensor([rankings_idx.long().tolist().index(i) for i in range(n_algos)])
        return ranking.float().view(1, -1)
        # fixme: do we need to sort this ranking? based on the algorithms
        #  original positions.

    # def kill_low_performers(self, learning_curves, algos_idx, k, budget, survivors, **kwargs):
    #     the classical sh algorithm
    #     slice = learning_curves[:, :, budget]
    #     alive_idx = algos_idx[survivors == 1]
    #     dead = torch.topk(slice[0, alive_idx], k, dim=0, largest=False)
    #
    #     return dead

    def kill_low_performers(self, learning_curves, k, survivors, **kwargs):
        # assuming that the model is masif_tmlr transformer
        mask = self.observed_mask
        dataset_meta_features = kwargs.get("dataset_meta_features", None)
        algo_meta_features = kwargs.get("algo_meta_features", None)
        expectation = self.model(
            learning_curves=learning_curves, mask=mask, dataset_meta_features=dataset_meta_features,
            algo_meta_features=algo_meta_features
        )

        # we can use our (relative) expectation only on those that are still alive!
        dead = torch.topk(expectation[0, survivors == 1], k, dim=0, largest=False)

        return dead


if __name__ == "__main__":
    import torch

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
    # wandb.init()

    from masif.utils.mlp import MLP

    class MLP_LC:
        def __init__(self, n_algos, fidelities):
            # expects that the learning curve tensor is being stacked
            self.model = MLP(hidden_dims=[n_algos * fidelities, 100, n_algos])

        def __call__(self, learning_curves, mask):
            shape = 1, reduce(operator.mul, learning_curves.shape, 1)
            return self.model(learning_curves.view(shape) * mask.view(shape))

    # Check Eta schedules and rankings
    # fixme: move to test!
    batch = 1
    n_algos = 10

    budgets = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 51]
    fidelities = len(budgets)
    lcs = torch.arange(batch * n_algos * fidelities, dtype=torch.float32).view(
        batch,
        n_algos,
        fidelities,
    )
    cost = torch.arange(batch * n_algos * fidelities).view(batch, n_algos, fidelities)
    mask = torch.cat(
        [
            torch.ones((batch, n_algos, fidelities - 4), dtype=torch.bool),
            torch.zeros((batch, n_algos, 4), dtype=torch.bool),
        ],
        axis=2,
    )

    # SH with eta=3

    model = MLP_LC(n_algos, fidelities)
    model(lcs, mask)
    sh = SHScheduler(model, budgets, 3)

    sh.forward(learning_curves=lcs, mask=mask, cost_curves=cost)
