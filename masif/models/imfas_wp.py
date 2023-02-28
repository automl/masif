import torch
import torch.nn as nn

from masif.utils.mlp import MLP


class masif_WP(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        input_dim: int,
        n_layers: int,
        device: str = "cpu",
    ):
        """
        Workshop paper version of the masif model: https://arxiv.org/pdf/2206.03130.pdf
        MLP1(D) = h_0 -> for_{i=0,..k}  LSTM(h_i, f_i)=h_{i+1}  -> MLP2(h_k) = output

        D: dataset meta-features
        f_i: performances of all n (=algo_dim) algorithms on the i-th fidelity

        The dimensions of the model are the following:
        MLP1: input_dim -> mlp1_hidden_dims -> h_dim
        LSTM: h_dim -> h_dim
        MLP2: h_dim -> mlp2_hidden_dims -> algo_dim

        Args:
            encoder: MLP to encode the datasets meta-features
            deocder: MLP to decode the hidden state of the LSTM to values that can be ranked
            input_dim: dimension of the learning curves that is fed as input to the LSTM
            n_layers: number of layers of the LSTM
            device: device to run the model on

        """

        super(masif_WP, self).__init__()

        self.encoder = encoder

        self.input_dim = input_dim
        self.n_layers = n_layers

        # The hidden dims of the LSTM are the output features of the encoder
        self.hidden_dim = [l for l in self.encoder.layers if isinstance(l, nn.Linear)][-1].out_features

        # LSTM layer of the network
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=n_layers,
            batch_first=True,  # We work with tensors, not tuples
        )

        self.double()

        self.device = torch.device(device)

        self.decoder = decoder

    def forward(self, dataset_meta_features, learning_curves, *args, **kwargs):
        """
        Forward path through the meta-feature ranker

        Args:
            dataset_meta_features: tensor of shape (batch_dim, meta_features_dim)
            learning_curves: tensor of shape (batch_dim, n_learning_curves)
            mask: if the learning curve values are observed

        Returns:
            tensor of shape (batch_dim, algo_dim)
        """
        dataset_meta_features = dataset_meta_features.double()
        learning_curves = learning_curves.double()

        # Initialize the hidden state with the output of the encoder
        h0 = torch.stack([self.encoder(dataset_meta_features) for _ in range(self.n_layers)]).requires_grad_().double()

        # Initialize cell state with 0s
        c0 = (
            torch.zeros(
                self.n_layers,  # Number of layers
                learning_curves.shape[0],  # batch_dim
                self.hidden_dim,  # hidden_dim
            )
            .requires_grad_()
            .double()
        )

        # Feed the learning curves as a batched sequence so that at every rollout step, a fidelity
        # is fed as an input to the LSTM

        out, (hn, cn) = self.lstm(learning_curves.permute(0, 2, 1).double(), (h0, c0))

        out = self.decoder(out[:, -1, :].float())

        # Return the decoded output and the list of hidden states and cell states
        return out


if __name__ == "__main__":
    network = masif_WP(
        encoder=MLP([2, 3, 4]),
        decoder=MLP([4, 3, 2]),
        input_dim=200,
        n_layers=2,
    )
    # print the network

    print("shared network", network)
