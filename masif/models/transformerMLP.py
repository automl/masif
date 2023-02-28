from typing import Optional

import torch
from torch import nn

from masif.models.masif_transformer_guided_attention import masifGuidedAttentionTransformerEncoder
from masif.utils import MLP


# marius proposal: prepend the dataset meta features to the sequence.

class masifTransformerMLP(nn.Module):
    def __init__(
            self,

            positional_encoder,
            transformer_lc,
            # decoder,
            device,
            decoder_hidden_dims: Optional[list],
            n_algos: int,
            n_fidelities: int,
            dataset_metaf_encoder: Optional[nn.Module] = None,

            **kwargs  # placeholder to make configuration more convenient
    ):
        """This model class defines the computational macro graph of the masif model.
        This variant passes the entire learning curve tensor to the transformer encoder jointly! """
        super(masifTransformerMLP, self).__init__()

        self.dataset_metaf_encoder = dataset_metaf_encoder

        dmetaf_encoder_outdim = self.dataset_metaf_encoder.layers[-1].weight.shape[-2]
        self.positional_encoder = positional_encoder

        self.transformer_lc = transformer_lc

        # n_fidelities + 1 for nan safeguard
        decoder_input_dim = dmetaf_encoder_outdim + (n_fidelities + 1) * n_algos
        self.decoder = MLP(hidden_dims=[decoder_input_dim, *decoder_hidden_dims, n_algos])

        self.device = device

        self.check_conformity()

    def check_conformity(self):
        pass

    def forward(self, learning_curves: torch.Tensor, mask: torch.Tensor,
                dataset_meta_features: Optional[torch.Tensor] = None, **kwargs):

        # n_datasets is batch dimension
        self.n_datasets, self.n_algos, self.n_fidelities = learning_curves.shape


        if dataset_meta_features is None:
            # if dataset meta features are not available, we use a zero vector.
            dataset_metaf_encoding = torch.zeros(
                self.dataset_metaf_encoder.layers[-1].weight.shape[-1]
            ).view(1, -1).to(self.device)

        else:
            # default case: dataset meta features are available
            dataset_metaf_encoding = self.dataset_metaf_encoder(dataset_meta_features)

        pos_learning_curves, mask = self.convert_lc(learning_curves, mask)
        if isinstance(self.transformer_lc, masifGuidedAttentionTransformerEncoder):
            lc_encoding = self.transformer_lc(
                pos_learning_curves,
                src_key_padding_mask=~mask.bool().view(self.n_algos, self.n_fidelities + 1),
                guided_attention=dataset_metaf_encoding
            )
        else:
            lc_encoding = self.transformer_lc(
                # reshape the learning curve tensor to trick the transformer in believing
                # the algo dim is the batch dim
                # alter mask encoding (since pytorch expects mask that indicates missing values)
                pos_learning_curves,

                # same as above regarding permutation. But we need to invert & convert the mask,
                # since pytorch expects mask that indicates missing values & type bool,
                # such that mask.masked_fill(~mask.bool(), float('-inf')) is performed directly before
                # softmax in the attention head # +1 due to the prepended zero token!
                src_key_padding_mask=~mask.bool().view(self.n_algos, self.n_fidelities + 1)

            )

        lc_encoding = lc_encoding.permute(0, 1, 2)  # undo the batch trick
        # consider: do we want to have a separate mlp for the lc_encoding, before joining?
        return self.decoder(
            # stack flattened lc_encoding with dataset_metaf_encoding
            torch.cat([lc_encoding.view(1, -1), dataset_metaf_encoding], 1)
        )

    def convert_lc(self, learning_curves: torch.Tensor, mask: torch.Tensor, **kwargs):

        # prepend a zero timestep to the learning curve tensor if n_fidelities is uneven
        # reasoning: transformer n_heads must be a devisor of d_model
        # if learning_curves.shape[-2] % 2 == 1:
        #     # fixme: untested code
        #     learning_curves = torch.cat(
        #         (torch.zeros_like(learning_curves[:, :, :1]), learning_curves), dim=-2)
        #
        #     mask = torch.cat((torch.zeros_like(mask[:, :, :1]), mask), dim=-2)

        pos_learning_curves = self.positional_encoder(learning_curves)

        # safeguard against nan values resulting from unobserved learning curves,
        # by prepending a zero token at the beginning of the fidelity sequence
        mask = torch.cat([mask.new_ones([1, self.n_algos, 1]), mask], dim=-1)
        pos_learning_curves = torch.cat(
            [pos_learning_curves.new_zeros([1, self.n_algos, 1, ]), pos_learning_curves],
            dim=-1
        )

        # batch trick: make the algorithms batch dim to allow for parallelization
        # of independently attending to the learning curves of different algorithms
        pos_learning_curves = pos_learning_curves.permute(1, 2, 0)
        mask = mask.permute(1, 2, 0)

        return pos_learning_curves, mask
