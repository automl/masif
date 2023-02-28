import torch

from masif.utils.modelinterface import ModelInterface


class RandomBaseline(ModelInterface):
    def __init__(self):
        pass

    def forward(self, learning_curves: torch.Tensor, **kwargs):
        # find the number of algorithms from the shape of the learning curves
        n_algos = learning_curves.shape[1]
        n_dataset = learning_curves.shape[0]

        # return random performance for each algorithm
        return torch.rand([n_dataset, n_algos]).view(n_dataset, -1)
