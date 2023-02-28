from typing import Callable, Dict

import torch
import torchsort
from torch.nn import Softmax
from torch.nn.modules.loss import _Loss as Loss

import pdb


class SpearmanLoss(Loss):
    def __init__(
        self, reduction: str = "mean", ranking_fn: Callable = torchsort.soft_rank, ts_kwargs: Dict = {}
    ) -> None:
        super(SpearmanLoss, self).__init__(reduction=reduction)
        self.ranking_fn = ranking_fn
        self.ts_kwargs = ts_kwargs

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # generate soft ranks

        y_hat = self.ranking_fn(y_hat, **self.ts_kwargs)
        y_true = self.ranking_fn(y_true, **self.ts_kwargs)

        # normalize the soft ranks
        y_hat = y_hat - y_hat.mean()
        y_hat = y_hat / y_hat.norm()

        # target = (target - target.min()) / (target.max() - target.min())
        y_true = y_true - y_true.mean()
        y_true = y_true / y_true.norm()

        # compute the correlation
        speark_rank = (y_hat * y_true).sum()

        # loss is the complement, which needs to be minimized
        return 1 - speark_rank


class WeightedSpearman(Loss):
    def __init__(
        self, reduction: str = "mean", ranking_fn: Callable = torchsort.soft_rank, ts_kwargs: Dict = {}
    ) -> None:
        super(WeightedSpearman, self).__init__(reduction=reduction)
        self.spearman_loss = SpearmanLoss(reduction, ranking_fn=ranking_fn, ts_kwargs=ts_kwargs)
        self.weight_func = Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.weight_func(target)
        input = input * weights
        return self.spearman_loss(input, target)
