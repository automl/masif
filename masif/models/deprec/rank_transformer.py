from typing import List

import torch
from torch import nn

from masif.models.masif_wp import AlgoRankMLP


class RankTransfromer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        layer_dim: int,
        output_dim: int,
        dropout: float,
        norm_first=False,
        use_src_mask: bool = False,
        use_src_key_padding_mask: bool = False,
        readout=None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Basic implementation of a Transformer Network

        Args:
            input_dim   : Dimension of the input
            d_model     : Model dimensions (embedding size)
            n_head      : Number of heads
            layer_dim   : Number of layers
            output_dim  : Dimension of the output
            readout     : Optional readout layer for decoding the hidden state
        """
        super(RankTransfromer, self).__init__()
        self.embedding_layer = nn.Identity() if input_dim == d_model else nn.Linear(input_dim, d_model)
        self.positional_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )

        self.encoder = nn.TransformerEncoder(encoder_layers, layer_dim)

        if readout is None:
            self.readout = nn.Linear(d_model, output_dim)
        else:
            self.readout = readout

        self.use_src_mask = use_src_mask
        self.use_src_key_padding_mask = use_src_key_padding_mask

        self.device = device
        self.double()

    def forward(self, init_features: torch.Tensor, context: torch.Tensor):
        """
        transformer forward pass. we first embedds context to a sequence and then attach init_features in front of it

        Args:
            init_features: initialized features generated by MLP with size [B, D_model]
            context: contextual information with size [B, L, N_input]

        Returns:

        """
        # We only add positional information to the fidelity values
        context = self.positional_encoder(self.embedding_layer(context))
        # net input is located as the first element in the sequence
        net_input = torch.concat([torch.unsqueeze(init_features, 1), context], dim=1)

        src_mask = None
        src_key_padding_mask = None
        if self.training:
            if self.use_src_mask:
                src_mask = nn.Transformer.generate_square_subsequent_mask(net_input.shape[1]).double().to(self.device)
            if self.use_src_key_padding_mask:
                # masked out part of the learning curves. This will force the network to do the prediction without
                # observing the full learning curve.
                input_shape = net_input.shape
                batch_size = input_shape[0]
                seq_length = input_shape[1]
                n_reserved_data = torch.randint(1, seq_length, (batch_size, 1))
                all_steps = torch.arange(0, seq_length)
                src_key_padding_mask = all_steps >= n_reserved_data
                src_key_padding_mask = src_key_padding_mask.to(self.device)
        out = self.encoder(net_input, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        # Convert the last element of the lstm into values that
        # can be ranked
        # Similar to ViT, readout tout the meta-feature related data
        out = self.readout(out[:, 0, :])

        return out


class RankTransformer_Ensemble(nn.Module):
    def __init__(
        self,
        input_dim: int = 107,
        algo_dim: int = 58,
        # lstm_hidden_dims: List[int] = 100,
        transformer_layers: int = 2,
        shared_hidden_dims: List[int] = [300, 200],
        n_head: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        norm_first: bool = False,
        use_src_mask: bool = False,
        use_src_key_padding_mask: bool = True,
        device: str = "cpu",
    ):
        """

        Sequential Ensemble of Transformers to rank based on multiple fidelities

        Args:
            input_dim: input dimension
            algo_dim: number of algorithms
            shared_hidden_dims: list of hidden dimensions for the shared MLP
            n_fidelities: number of fidelities
            multi_head_dims: list of hidden dimensions for each multi-head
            fc_dim: list of hidden dimensions for the FC layers
            joint: options: 'avg', plain average of rank outputs
            'wavg' learnable weighted combination of model mlp outputs
            device: device to run the model on

        """
        super(RankTransformer_Ensemble, self).__init__()
        self.meta_features_dim = input_dim
        self.algo_dim = algo_dim
        self.transformer_layers = transformer_layers
        self.shared_hidden_dims = shared_hidden_dims
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.device = torch.device(device)

        self.dropout = dropout
        self.norm_first = norm_first

        self.use_src_mask = use_src_mask
        self.use_src_key_padding_mask = use_src_key_padding_mask

        self._build_network()

    def _build_network(self):
        """
        Build the network based on the initialized hyperparameters

        """

        # Build the shared network
        self.shared_network = AlgoRankMLP(
            input_dim=self.meta_features_dim,
            algo_dim=self.shared_hidden_dims[-1],
            hidden_dims=self.shared_hidden_dims[:-1],
        )

        # Build the transformer
        self.seq_network = RankTransfromer(
            input_dim=self.algo_dim,
            d_model=self.shared_hidden_dims[-1],
            n_head=self.n_head,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            norm_first=self.norm_first,
            layer_dim=self.transformer_layers,
            output_dim=self.algo_dim,
            readout=None,
            use_src_mask=self.use_src_mask,
            use_src_key_padding_mask=self.use_src_key_padding_mask,
            device=self.device,
        )

    def forward(self, dataset_meta_features, fidelities):
        """
        Forward path through the meta-feature ranker

        Args:
            D: input tensor

        Returns:
            algorithm values tensor
        """

        # Forward through the shared network
        shared_D = self.shared_network(dataset_meta_features)

        # Forward through the lstm networks to get the readouts
        transformer_D = self.seq_network(init_features=shared_D, context=fidelities)

        return shared_D, transformer_D
