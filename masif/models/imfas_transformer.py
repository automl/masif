from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
from masif.utils.positionalencoder import PositionalEncoder
from masif.models.masif_transformer_guided_attention import masifGuidedAttentionTransformerEncoder


class AbstractmasifTransformer(nn.Module):
    def __init__(
            self,
            n_algos: int,
            n_fidelities: int,
            dataset_metaf_encoder: nn.Module,
            algo_metaf_encoder: nn.Module,
            decoder: nn.Module,
            transformer_lc: masifGuidedAttentionTransformerEncoder,
            device: str = "cpu",
            model_opts: list[str] = []
    ):
        """
        Abstract masif Transformer models. Here I propose two types of Transformer models. The first one is to consider
        all the learning curve values of different sequences as features and apply a single forward pass to the tensors.
        The second one is to

        Args:
            encoder:
            decoder:
            transformer_layer:
            n_layers:
            device:
            model_opts: model options: 
                if any element exist in model_opts, the corresponding function will be updated:
                    reduce: if the reduce layer is applied to the transformer
                    pe_g (hierarchical transformer only): if positional encoding is applied to global transformer layer
                    eos_tail (hierarchical transforemr only): if the EOS embedding is attached in the end of the padded 
                        sequence instead of the raw sequence 
                    
        """
        super(AbstractmasifTransformer, self).__init__()
        self.dataset_metaf_encoder = dataset_metaf_encoder
        self.algo_metaf_encoder = algo_metaf_encoder
        self.decoder = decoder

        self.transformer_lc = transformer_lc

        self.d_model = self.transformer_lc.layers[0].linear1.in_features
        self.n_layers = self.transformer_lc.num_layers

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.lc_proj_layer = self.build_lc_embedding_layer(n_algos, self.d_model)
        self.positional_encoder = PositionalEncoder(d_model=self.d_model)

        model_opts = set(model_opts)

        self.has_reduce_layer = 'reduce' in model_opts
        if self.has_reduce_layer:
            self.reduce_layer = torch.nn.Linear(n_fidelities + 1, 1)

        self.norm_mid = nn.LayerNorm(self.d_model)
        self.norm_before_decoder = nn.LayerNorm(self.decoder.layers[0].in_features)

        self.to(device)

        # for k, i in self.named_parameters():
        #     print(k, i.device)

    def build_lc_embedding_layer(self, n_algos: int, d_model_transformer: int):
        raise NotImplementedError

    def preprocessing_lcs(self, learning_curves, lc_values_observed):
        (batch_size, n_algos, lc_length) = learning_curves.shape

        learning_curves = learning_curves.transpose(1, 2)

        lc_values_observed = lc_values_observed.transpose(1, 2)

        return learning_curves, lc_values_observed, (batch_size, n_algos, lc_length)

    def embeds_lcs(self, learning_curves, lc_values_observed):

        # learning_curves_embedding = self.positional_encoder(learning_curves)

        learning_curves_embedding = self.positional_encoder(self.lc_proj_layer(learning_curves))

        return learning_curves_embedding, lc_values_observed

    def forward(self, learning_curves, mask, dataset_meta_features=None, algo_meta_features=None):
        # FIXME: Consider How to encode Missing values here

        learning_curves, lc_values_observed, lc_shape_info = self.preprocessing_lcs(learning_curves, mask)

        if dataset_meta_features is not None:
            D_embedding = self.dataset_metaf_encoder(dataset_meta_features)
        else:
            D_embedding = None

        A_embedding = self.algo_metaf_encoder(algo_meta_features[0]) if algo_meta_features is not None else None

        learning_curves_embedding, lc_values_observed = self.embeds_lcs(learning_curves, lc_values_observed)

        encoded_lcs = self.encode_lc_embeddings(learning_curves_embedding, D_embedding, A_embedding, lc_values_observed,
                                                lc_shape_info)

        if dataset_meta_features is None:
            dataset_meta_features = torch.full([0], fill_value=0.0, dtype=encoded_lcs.dtype, device=encoded_lcs.device)
            D_embedding = self.dataset_metaf_encoder(dataset_meta_features).repeat([*encoded_lcs.shape[:-1], 1])
        else:
            if len(encoded_lcs.shape) == 3:
                D_embedding = D_embedding.unsqueeze(1).repeat(1, encoded_lcs.shape[1], 1)

        decoder_input = self.norm_before_decoder(torch.cat((encoded_lcs, D_embedding), -1))

        return self.decoder(decoder_input).squeeze(-1)

    def encode_lc_embeddings(
            self,
            learning_curves_embedding: torch.Tensor,
            D_embedding: Optional[torch.Tensor],
            A_embedding: Optional[torch.Tensor],
            lc_values_observed: torch.Tensor,
            lc_shape_info: Tuple[int, int, int],
    ) -> torch.Tensor:
        raise NotImplementedError


class masifBaseTransformer(AbstractmasifTransformer):
    """
    This implementation follow the masif's LSTM's implementation, switching the
    LSTM layer with a Transformer layer and being ignorant to HPs, as they are constant.
    """

    def __init__(
            self,
            n_algos: int,
            n_fidelities: int,
            dataset_metaf_encoder: nn.Module,
            algo_metaf_encoder: nn.Module,
            decoder: nn.Module,
            transformer_lc: torch.nn.TransformerEncoderLayer,
            device: str = "cpu",
            model_opts: list[str] = []
    ):
        super(masifBaseTransformer, self).__init__(n_algos, n_fidelities, dataset_metaf_encoder, algo_metaf_encoder,
                                                   decoder, transformer_lc, device, model_opts)
        # This is attached at the end of each LCs to indicate that the LC ends here and we could extrac their
        # corresponding feature values. Here I simply compute the number of observed values as an input
        # TODO: Alternative: different embeddings w.r.t. position or algos
        self.lc_length_embedding = nn.Linear(n_algos, transformer_lc.linear1.in_features)

        self.to(torch.device(device))

    def build_lc_embedding_layer(self, n_algos: int, d_model_transformer: int):
        return nn.Linear(n_algos, d_model_transformer)

    def encode_lc_embeddings(self,
                             learning_curves_embedding,
                             D_embedding: Optional[torch.Tensor],
                             A_embedding: Optional[torch.Tensor],
                             mask, lc_shape_info):
        lc_length_embedding = self.lc_length_embedding(mask.sum(1)).unsqueeze(1)

        # lc_values_observed = lc_values_observed.transpose(1, 2)

        # NOTE Transformer encoder should not have the necessity for this
        # encoded_lcs = self.transformer_lc(learning_curves_embedding)

        encoded_lcs = self.transformer_lc(
            torch.cat([learning_curves_embedding, lc_length_embedding], dim=1),
        )
        if self.has_reduce_layer:
            encoded_lcs = torch.transpose(encoded_lcs, 1, 2)
            encoded_lcs = self.reduce_layer(encoded_lcs)
            encoded_lcs = torch.transpose(encoded_lcs, 1, 2).squeeze(1)
        else:
            encoded_lcs = encoded_lcs[:, -1, :]

        return encoded_lcs


class masifHierarchicalTransformer(AbstractmasifTransformer):
    """
    An masif Trasnforemr with hierarchical architecture. For the input, we first flatten them to perform local attention
    operation with self.transformer_lc. Then in the second stage, we perform a global attention operation with
    self.transformer_algo_encoder
    """

    def __init__(
            self,
            n_algos: int,
            n_fidelities: int,
            dataset_metaf_encoder: nn.Module,
            algo_metaf_encoder: nn.Module,
            decoder: nn.Module,
            transformer_lc: masifGuidedAttentionTransformerEncoder,
            transformer_algo: masifGuidedAttentionTransformerEncoder,
            device: str = "cpu",
            model_opts: list[str] = []
    ):
        self.EOS = torch.tensor(0, device=torch.device(device))  # End of Sequence

        super(masifHierarchicalTransformer, self).__init__(
            n_algos, n_fidelities, dataset_metaf_encoder, algo_metaf_encoder, decoder, transformer_lc, device,
            model_opts
        )

        self.transformer_algo = transformer_algo

        self.eos_embedding_layer = torch.nn.Embedding(2, self.d_model)

        if self.has_reduce_layer:
            self.reduce_layer = torch.nn.Linear(self.d_model, n_fidelities + 1)
            self.reduce_weights = torch.nn.Linear(n_fidelities + 1, 1)

        model_opts = set(model_opts)

        self.pe_on_global_level = 'pe_g' in model_opts

        self.eos_tail = True

        self.d_meta_guided = 'd_meta_guided' in model_opts

        self.to(torch.device(device))

    def to(self, device):
        self.EOS = self.EOS.to(device)
        super(masifHierarchicalTransformer, self).to(device)

    def build_lc_embedding_layer(self, n_algos: int, d_model_transformer: int):
        return nn.Linear(1, d_model_transformer)

    def preprocessing_lcs(self, learning_curves, lc_values_observed):
        lc_shape_info = learning_curves.shape
        batch_size, n_algos, lc_length = learning_curves.shape

        learning_curves = learning_curves.view(batch_size * n_algos, lc_length, 1)
        lc_values_observed = lc_values_observed.view(batch_size * n_algos, lc_length)

        return learning_curves, lc_values_observed, lc_shape_info

    def embeds_lcs(self, learning_curves, lc_values_observed):
        lc_embeddings = self.positional_encoder(self.lc_proj_layer(learning_curves)) * lc_values_observed.unsqueeze(-1)

        n_lcs, lc_length, d_model = lc_embeddings.shape

        # We attach the ending Embedding to the end of each sequences
        eos_embedding = self.eos_embedding_layer(self.EOS)

        n_observed_lcs = lc_values_observed.sum(1).long()

        lc_embeddings = torch.cat([lc_embeddings, eos_embedding.repeat(n_lcs, 1, 1)], dim=1, )
        # we have an additional item
        lc_values_observed = torch.cat(
            [
                lc_values_observed,
                torch.ones(
                    (len(lc_values_observed), 1), dtype=lc_values_observed.dtype, device=lc_values_observed.device
                ),
            ],
            dim=1,
        )

        return lc_embeddings, lc_values_observed

    def encode_lc_embeddings(self, learning_curves_embedding,
                             D_embedding: Optional[torch.Tensor],
                             A_embedding: Optional[torch.Tensor], lc_values_observed, lc_shape_info):
        batch_size, n_algos, lc_length = lc_shape_info
        n_observed_lcs = lc_values_observed.sum(1).long() - 1

        if D_embedding is not None and self.d_meta_guided:
            encoded_lcs_local = self.transformer_lc(learning_curves_embedding,
                                                    src_key_padding_mask=~lc_values_observed.bool(),
                                                    guided_attention=D_embedding)
        else:
            encoded_lcs_local = self.transformer_lc(
                learning_curves_embedding,
                src_key_padding_mask=~lc_values_observed.bool(),
            )
        encoded_lcs_local = encoded_lcs_local.unflatten(0, [batch_size, -1])

        if self.has_reduce_layer:
            if D_embedding is not None:
                encoded_lcs_local = encoded_lcs_local * self.reduce_layer(D_embedding).view([batch_size, 1, -1, 1])
                encoded_lcs_local = self.norm_mid(torch.sum(encoded_lcs_local, -2))
            else:
                encoded_lcs_local = self.norm_mid(self.reduce_weights(encoded_lcs_local.transpose(-1, -2)).squeeze(-1))
        else:
            encoded_lcs_local = encoded_lcs_local[:, :, -1]
        if self.pe_on_global_level:
            encoded_lcs_local = self.norm_mid(encoded_lcs_local + A_embedding.unsqueeze(0).repeat(batch_size, 1, 1))
        # TODO adjust the Meta features with this type of transformation

        # guided attention on algo_transformer is somehow probmatic, I will drop it at the current stage
        encoded_lcs = self.transformer_algo(encoded_lcs_local)

        return encoded_lcs


class masifMLPTransformer(masifHierarchicalTransformer):
    def __init__(
            self,
            n_algos: int,
            n_fidelities: int,
            dataset_metaf_encoder: nn.Module,
            algo_metaf_encoder: nn.Module,
            decoder: nn.Module,
            transformer_lc: masifGuidedAttentionTransformerEncoder,
            device: str = "cpu",
            model_opts: list[str] = []
    ):
        self.EOS = torch.tensor(0, device=torch.device(device))  # End of Sequence
        super(masifMLPTransformer, self).__init__(
            n_algos, n_fidelities, dataset_metaf_encoder, algo_metaf_encoder, decoder, transformer_lc, None, device,
            model_opts
        )

        self.has_flatten_layer = False
        if not self.has_reduce_layer:
            if 'flatten_t_out' in model_opts:
                self.has_flatten_layer = False
                self.flatten_layer = torch.nn.Linear((n_fidelities + 1) * self.d_model, self.d_model)
        self.to(torch.device(device))

    def encode_lc_embeddings(self, learning_curves_embedding,
                             D_embedding: Optional[torch.Tensor],
                             A_embedding: Optional[torch.Tensor], lc_values_observed, lc_shape_info):
        batch_size, n_algos, lc_length = lc_shape_info

        if D_embedding is not None and self.d_meta_guided:
            encoded_lcs_local = self.transformer_lc(learning_curves_embedding,
                                                    src_key_padding_mask=~lc_values_observed.bool(),
                                                    guided_attention=D_embedding)
        else:
            encoded_lcs_local = self.transformer_lc(
                learning_curves_embedding,
                src_key_padding_mask=~lc_values_observed.bool(),
            )
        encoded_lcs_local = encoded_lcs_local.unflatten(0, [batch_size, -1])

        if self.has_reduce_layer:
            if D_embedding is not None:
                encoded_lcs_local = encoded_lcs_local * self.reduce_layer(D_embedding).view([batch_size, 1, -1, 1])
                encoded_lcs_local = self.norm_mid(torch.sum(encoded_lcs_local, -2))
            else:
                encoded_lcs_local = self.norm_mid(self.reduce_weights(encoded_lcs_local.transpose(-1, -2)).squeeze(-1))
        elif self.has_flatten_layer:
            encoded_lcs_local = self.flatten_layer(encoded_lcs_local.view([batch_size, n_algos, -1]))
        else:
            encoded_lcs_local = encoded_lcs_local[:, :, -1]
        # TODO adjust the Meta features with this type of transformation
        return encoded_lcs_local.flatten(1)




class masifCrossTransformer(masifHierarchicalTransformer):
    """
    An masif Transformer with hierarchical architecture. For the input, we first flatten them to perform local attention
    operation with self.transformer_lc. Then in the second stage, we perform a global attention operation with
    self.transformer_algo_encoder
    """

    def __init__(
            self,
            n_algos: int,
            n_fidelities: int,
            dataset_metaf_encoder: nn.Module,
            algo_metaf_encoder: nn.Module,
            decoder: nn.Module,
            transformer_lc: masifGuidedAttentionTransformerEncoder,
            transformer_algo: masifGuidedAttentionTransformerEncoder,
            n_layers=2,
            device: str = "cpu",
            model_opts: list[str] = []
    ):
        super(masifCrossTransformer, self).__init__(n_algos, n_fidelities, dataset_metaf_encoder, algo_metaf_encoder,
                                                    decoder, transformer_lc, transformer_algo, device, model_opts)
        self.full_lc2global_former = 'full_lc2global' in model_opts
        if not self.eos_tail and self.full_lc2global_former:
            raise ValueError("Unsupported combiantion of eos_tail and full_lc2global_former")
        if self.full_lc2global_former:
            self.transformer_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(n_layers)])
        else:
            self.transformer_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(n_layers - 1)])
        self.to(torch.device(device))

    def encode_lc_embeddings(self, learning_curves_embedding,
                             D_embedding: Optional[torch.Tensor],
                             A_embedding: Optional[torch.Tensor], lc_values_observed, lc_shape_info):
        batch_size, n_algos, lc_length = lc_shape_info
        n_observed_lcs = lc_values_observed.sum(1).long() - 1

        lc_encoder_input = learning_curves_embedding

        src_key_padding_mask = ~lc_values_observed.bool()

        if self.has_reduce_layer and D_embedding is not None:
            fidelity_weights = self.reduce_layer(D_embedding).view([batch_size, 1, -1, 1])
        if not self.full_lc2global_former:

            for i, (lc_encoder, algo_encoder) in enumerate(
                    zip(self.transformer_lc.layers, self.transformer_algo.layers)):
                if D_embedding is not None and self.d_meta_guided:
                    lc_encoder_output = lc_encoder(learning_curves_embedding,
                                                   src_key_padding_mask=~lc_values_observed.bool(),
                                                   guided_attention=D_embedding)
                else:
                    lc_encoder_output = lc_encoder(
                        lc_encoder_input, src_key_padding_mask=src_key_padding_mask
                    )

                if self.has_reduce_layer:
                    if D_embedding is not None:
                        algo_encoder_input = lc_encoder_output.unflatten(0, [batch_size, -1]) * fidelity_weights
                        algo_encoder_input = self.norm_mid(torch.sum(algo_encoder_input, -2))
                    else:
                        algo_encoder_input = self.norm_mid(
                            self.reduce_weights(lc_encoder_output.transpose(-1, -2)).squeeze(-1)).unflatten(0, [batch_size, -1])
                else:
                    algo_encoder_input = lc_encoder_output[:, -1].unflatten(0, [batch_size, -1])

                if self.pe_on_global_level:
                    algo_encoder_input = self.norm_mid(
                        algo_encoder_input + A_embedding.unsqueeze(0).repeat(batch_size, 1, 1))
                # TODO adjust the Meta features with this type of transformation

                algo_encoder_output = algo_encoder(algo_encoder_input)

                if i < self.n_layers - 1:
                    lc_encoder_input = lc_encoder_output + algo_encoder_output.view(batch_size * n_algos, 1, -1).repeat(
                        1, lc_encoder_output.shape[1], 1)
                    lc_encoder_input = self.transformer_norms[i](lc_encoder_input)

        else:
            lc_seq_length = lc_length + 1
            for i, (lc_encoder, algo_encoder) in enumerate(
                    zip(self.transformer_lc.layers, self.transformer_algo.layers)):
                if D_embedding is not None and self.d_meta_guided:
                    lc_encoder_output = lc_encoder(learning_curves_embedding,
                                                   src_key_padding_mask=~lc_values_observed.bool(),
                                                   guided_attention=D_embedding)
                else:
                    lc_encoder_output = lc_encoder(
                        lc_encoder_input, src_key_padding_mask=src_key_padding_mask
                    )
                lc_encoder_output = lc_encoder_output.view(batch_size, n_algos, lc_seq_length, -1).transpose(1, 2)
                lc_encoder_output = lc_encoder_output.reshape(batch_size * lc_seq_length, n_algos, -1)

                src_key_padding_mask_algo = src_key_padding_mask.reshape(batch_size, n_algos, lc_seq_length).transpose(
                    1, 2)
                src_key_padding_mask_algo = src_key_padding_mask_algo.reshape(batch_size * lc_seq_length, n_algos)

                valid_seq = ~(src_key_padding_mask_algo.all(1))

                if self.pe_on_global_level:
                    algo_encoder_input = self.norm_mid(
                        (lc_encoder_output + A_embedding.unsqueeze(0).repeat(batch_size * lc_seq_length, 1, 1))[
                            valid_seq])
                else:
                    algo_encoder_input = lc_encoder_output[valid_seq]

                algo_encoder_out = algo_encoder(algo_encoder_input,
                                                src_key_padding_mask=src_key_padding_mask_algo[valid_seq])
                if sum(valid_seq) < len(lc_encoder_output):
                    algo_encoder_output = torch.zeros_like(lc_encoder_output)
                    algo_encoder_output[valid_seq] = algo_encoder_out
                else:
                    algo_encoder_output = lc_encoder_output

                lc_encoder_input = self.transformer_norms[i]((lc_encoder_output + algo_encoder_output))
                lc_encoder_input = lc_encoder_input.view(batch_size, lc_seq_length, n_algos, -1).transpose(1,
                                                                                                           2)  # [B,A,F, E]
                lc_encoder_input = lc_encoder_input.reshape([batch_size * n_algos, lc_seq_length, -1])

                if i == self.n_layers - 1:
                    algo_encoder_output = lc_encoder_input.unflatten(0, [batch_size, -1])
                    if self.has_reduce_layer:
                        if D_embedding is not None:
                            algo_encoder_output = algo_encoder_output * fidelity_weights
                            algo_encoder_output = self.norm_mid(torch.sum(algo_encoder_output, -2))
                        else:
                            algo_encoder_output = self.norm_mid(
                                self.reduce_weights(algo_encoder_output.transpose(-1, -2)).squeeze(-1))
                    else:
                        algo_encoder_output = algo_encoder_output[:, :, -1]

        encoded_lcs = algo_encoder_output

        return encoded_lcs
