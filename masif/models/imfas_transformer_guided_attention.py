"""
Defines an alternative to masifTransformerMLP.transformer_lc that uses guided attention.
Main target in the subclassing is TransformerEncoderLayer._sa_block to accept the dataset meta
feature encoding as input and (using an mlp) reshape it to the same shape as the query and value
"""
from typing import Optional
from collections.abc import Iterable
import torch
from torch import nn

from masif.utils import MLP


#
class masifGuidedAttentionTransformerEncoder(nn.TransformerEncoder):
    def __init__(self, metaf_embed_dim, l_seq, hidden_dims=tuple(), *args, **kwargs):
        """
        :param dataset_metaf_embed_dim: dimension of the dataset meta feature embedding
        :param n_fidelities: number of fidelities
        :param hidden_dims: intermediate hidden dimensions of the MLP that maps the dataset meta
        feature embedding. input and output are predetermined & automatically computed
        """
        # to meet the transformer-nan-safeguard we need to add 1 to the output dim
        if isinstance(l_seq, Iterable):
            l_seq = sum(l_seq)
        kwargs['encoder_layer'] = kwargs['encoder_layer'](
            guided_attention_encoder=MLP(hidden_dims=[metaf_embed_dim, *hidden_dims, l_seq])
        )

        super(masifGuidedAttentionTransformerEncoder, self).__init__(*args, **kwargs)

    def forward(
            self,
            src: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            guided_attention: Optional[torch.Tensor] = None,
            **kwargs
    ):
        output = src
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                guided_attention=guided_attention
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class GuidedAttentionTransformerEncoderLayer(nn.TransformerEncoderLayer):

    def __init__(self, guided_attention_encoder: Optional[nn.Module] = None, *args, **kwargs):
        print('transformer layer init')
        print(args, kwargs)
        super(GuidedAttentionTransformerEncoderLayer, self).__init__(*args, **kwargs)
        # if guided_attention_encoder is None, we return to vanilla transformer layer
        self.guided_attention_encoder = guided_attention_encoder

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                guided_attention: Optional[torch.Tensor] = None) \
            -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            guided_attention: tensors for guiding the attention (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        if (src.dim() == 3 and not self.norm_first and not self.training and
                self.self_attn.batch_first and
                self.self_attn._qkv_same_embed_dim and self.activation_relu_or_gelu and
                self.norm1.eps == self.norm2.eps and
                src_mask is None and
                not (src.is_nested and src_key_padding_mask is not None)):
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )
            if (not torch.overrides.has_torch_function(tensor_args) and
                    # We have to use a list comprehension here because TorchScript
                    # doesn't support generator expressions.
                    all([(x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]) and
                    (not torch.is_grad_enabled() or all(
                        [not x.requires_grad for x in tensor_args]))):
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    False,  # norm_first, currently not supported
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    src_mask if src_mask is not None else src_key_padding_mask,
                    # TODO: split into two args
                )
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                guided_attention
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, guided_attention))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor],
            key_padding_mask: Optional[torch.Tensor],
            guided_attention: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # print(x.shape, self.guided_attention_encoder(guided_attention).T.shape)
        # fixme: encoder needs output dim + 1!
        if guided_attention is not None:
            batch_size = len(guided_attention)
            if batch_size > 1:
                guided_attention_weights = self.guided_attention_encoder(guided_attention).view([batch_size, 1, -1, 1])
                q = (x.unflatten(0, [batch_size, -1]) * guided_attention_weights).flatten(0, 1)
            else:
                q = x * self.guided_attention_encoder(guided_attention).T
        else:
            q = x
        k = x
        x = self.self_attn(q, k, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)
