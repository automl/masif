import torch
import torch.nn as nn

import pdb


class TopkRegret(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, y_pred, y_true):
        """
        Compute the top-k regret. Given the scores in y_pred, find the top-k performing models.
        Then look up their performances in the ground true y_true and compute the difference
        between the best performing candidate and the best available model.

        :param: y_pred: torch.Tensor, shape (n_samples, n_models, n_outputs). Model scores.
        :param: y_true: torch.Tensor, shape (n_samples, n_outputs). Ground truth scores,
        which imply a ranking of the algorithms.

        NOTE: Careful, TopkMaxRegret is assuming that the maximum of y_scores is the best.
        """

        # print(y_pred, y_true)
        # pdb.set_trace()

        contenders = torch.topk(y_pred, k=self.k, dim=1)
        contenders_true_performance = y_true.gather(1, contenders.indices)
        # y_true[:, contenders.indices]
        best_contender = torch.max(contenders_true_performance, dim=1).values

        if self.k == 1:
            best_contender = best_contender.unsqueeze(1)
        best_available = torch.max(y_true, dim=1).values

        return torch.mean(best_available - best_contender)  # batch mean


if __name__ == "__main__":
    # check top-1 regret
    torch.manual_seed(1)

    y_pred = torch.randperm(10)[:10].view(1, -1).double()
    y_true = torch.randperm(10)[:10].view(1, -1).double()
    regret = TopkRegret(k=1)
    assert regret(y_pred, y_true) == torch.tensor(3.0)

    # batch test
    torch.manual_seed(1)

    y_pred = torch.randperm(10)[:10].view(2, -1).double()
    y_true = torch.randperm(10)[:10].view(2, -1).double()
    regret = TopkRegret(k=1)
    assert torch.equal(regret(y_pred, y_true), torch.tensor(3.5).double())

    # batch test: k=2 the next candidates are both worse.
    regret = TopkRegret(k=2)
    assert torch.equal(regret(y_pred, y_true), torch.tensor(3.5).double())
