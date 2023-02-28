import logging
from typing import List, Optional

import math
import torch

from masif.utils.modelinterface import ModelInterface

logger = logging.getLogger(__name__)

import pdb

# logger.setLevel('DEBUG')


class SuccessiveHalving(ModelInterface):
    def __init__(self, budgets: List, eta: int = 2, device: str = "cpu", budget_type="additive"):
        """
        This class assumes a single batch (i.e. a single dataset!)
        :param budgets: List[int] of budget 'labels' to be used in the successive halving run
        (actual budget values available in the learning curve tensor provided during forward).
        :param eta: int, the reduction factor of the budget in each successive halving run.
        """
        self.eta = eta
        self.budgets = budgets
        self.device = device
        self.budget_type = budget_type

        self.update_costs = self.update_costs_additive if budget_type == "additive" else self.update_costs_cumulative

        # FIXME: with the observed map, the tracing of the budget can be "trivially" refactored
        #  and the update cost need not be executed at every step

    def plan_budgets(self, budgets: List, mask: torch.Tensor):
        """
        Plan the successive halving run. This method is called once per run.
        Idea is to calculate the schedule dynamically depending on the currently available budget.
        :param budgets: List[int] of budget 'labels' to be used in the successive halving run,
        amounting to the budget; e.g. epochs or dataset subset size.
        :param mask: torch.Tensor (long), the mask of the learning curve tensor.
        indicating which budgets are accessible. Assumes that max budget is available to all algos.

        :return: torch.Tensor (long), the schedule of indices on the learning curve tensor for the
        successive halving run.
        """
        assert len(budgets) == mask.shape[-1], "Budgets and mask do not match!"

        # determine min & max budget
        max_fidelity_idx = mask.sum(dim=-1).max().long() - 1
        if isinstance(max_fidelity_idx, torch.Tensor):  # depending on the mask dtype!
            max_fidelity_idx = max_fidelity_idx.item()  # convert to int

        if max_fidelity_idx == -1:
            logger.debug("No budget available, returning empty schedule!")
            return torch.tensor([], dtype=torch.long, device=self.device)

        max_budget = budgets[max_fidelity_idx]
        min_budget = budgets[0]

        # Actual budget schedule (based on min_budget and eta)
        schedule = torch.tensor(
            [min_budget * self.eta**i for i in range(int(math.log(max_budget / min_budget, self.eta)) + 1)]
        )

        # translate the budget schedule to the index of the learning curve
        # i.e. find first surpassing index
        schedule_index = torch.tensor([next(x for x, val in enumerate(budgets) if val >= b) for b in schedule])

        # If more budget has been observed, make decision for final fidelity on the basis of
        # the last observed budget. This is used to make sh comparable to other algorithms
        # from the available max fidelity perspective (which likely is most informative).
        if schedule[-1] < max_budget:
            schedule_index = torch.cat((schedule_index, torch.tensor([max_fidelity_idx])))

        return schedule_index

    def update_costs_additive(
        self,
        cost_curves,
        budget,
        survivors,
    ):
        """
        Additive costs (i.e. every level evaluates from scratch)
        """

        # inquired cost  for the next evaluation round
        if cost_curves is not None:
            self.elapsed_time += cost_curves[:, :, budget][0, survivors == 1].sum(dim=0)
        else:
            self.elapsed_time += self.budgets[budget] * torch.sum(survivors)

    def update_costs_cumulative(
        self,
        cost_curves,
        budget,
        survivors,
    ):
        old_budget = self.schedule_index[self.schedule_index.tolist().index(budget) - 1]
        if cost_curves is not None:
            self.elapsed_time += (cost_curves[:, :, budget] - cost_curves[:, :, old_budget])[0, survivors == 1].sum(
                dim=0
            )
        else:
            self.elapsed_time += (self.budgets[budget] - self.budgets[old_budget]) * torch.sum(survivors)  # n_survivors

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

        learning_curves = learning_curves

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
            self.observed_mask[:, survivors == 1, : (budget + 1)] = 1

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
            slice = learning_curves[:, :, budget]
            alive_idx = algos_idx[survivors == 1]

            dead = torch.topk(slice[0, alive_idx], k, dim=0, largest=False)

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


if __name__ == "__main__":

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
    # wandb.init()

    # Check Eta schedules and rankings
    # fixme: move to test!
    batch = 1
    n_algos = 10

    budgets = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 51]
    fidelities = len(budgets)
    lcs = torch.arange(batch * n_algos * fidelities, dtype=torch.float32).view(batch, n_algos, fidelities)
    cost = torch.arange(batch * n_algos * fidelities).view(batch, n_algos, fidelities)
    mask = torch.cat(
        [
            torch.ones((batch, n_algos, fidelities - 4), dtype=torch.bool),
            torch.zeros((batch, n_algos, 4), dtype=torch.bool),
        ],
        axis=2,
    )

    # SH with eta=3
    sh = SuccessiveHalving(budgets, 2)

    assert torch.equal(
        sh.forward(learning_curves=lcs, mask=mask, cost_curves=cost),
        torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]),
    )
    assert torch.equal(
        sh.elapsed_time[0].long(),
        torch.tensor([0, 11, 22, 33, 44, 55, 66, 77, 88, 99]).sum()
        + torch.tensor([35, 46, 57, 68, 79, 90, 101]).sum()
        + torch.tensor([50, 61, 72, 83, 94, 105]).sum(),
    )
    assert torch.equal(sh.schedule_index, torch.tensor([0, 2, 6]))  # capped at 6 (last budget)
    # because of the mask, which stops at 6 although eta=3 would expect 8 as last budget!
    assert torch.equal(torch.tensor(budgets)[sh.schedule_index], torch.tensor([5, 15, 35]))  # capped at 35

    assert torch.equal(
        sh.forward(learning_curves=lcs, mask=mask), torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
    )

    # Same SH, but with eta=2
    sh = SuccessiveHalving(budgets, 2)

    assert torch.equal(
        sh.forward(learning_curves=lcs, mask=mask), torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
    )
    assert torch.equal(sh.schedule_index, torch.tensor([0, 1, 3, 6]))  # due to mask: capping at 6

    # Check the foward call for different budgets
    budgets = [1, 2, 3, 3.5, 4, 5, 6, 7, 8, 9]
    sh = SuccessiveHalving(budgets, 3)

    batch = 1
    n_algos = 10
    fidelities = len(budgets)
    lcs = torch.arange(batch * n_algos * fidelities).reshape(batch, n_algos, fidelities)
    mask = torch.cat(
        [
            torch.ones((batch, n_algos, fidelities - 4), dtype=torch.bool),
            torch.zeros((batch, n_algos, 4), dtype=torch.bool),
        ],
        axis=2,
    )

    rankings = sh.forward(lcs, mask)
    assert torch.equal(sh.schedule_index, torch.tensor([0, 2, 5]))
    sh.elapsed_time

    # No longer supports batch capabilities
    # assert torch.equal(
    #     rankings, torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
    #                            dtype=torch.float)
    # )
