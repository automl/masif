import torch.nn as nn


class MLPD(nn.Module):
    def __init__(self, encoder: nn.Module, device: str = "cpu"):
        """Interface to MLP, that accepts any input from the Dataloader, but makes
        use of the dataset_meta_features only 'MLP(D)'"""
        super(MLPD, self).__init__()
        self.encoder = encoder  # to allow for arbitrary parametrization of MLP
        self.device = device

    def forward(self, dataset_meta_features, *args, **kwargs):
        return self.encoder(dataset_meta_features)
