from typing import Optional, Tuple

import math
import torch
from torch import nn


class PositionalEncoder(nn.Module):
    r"""This function is direcly copied from Auto-PyTorch Forecasting and is a modified version of
        https://github.com/pytorch/examples/blob/master/word_language_model/model.py

        NOTE: different from the raw implementation, this model is designed for the batch_first inputs!
        Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model (int):
            the embed dim (required).
        dropout(float):
            the dropout value (default=0.1).
        max_len(int):
            the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:  # even vs uneven number of dimensions
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, pos_idx: Optional[Tuple[int]] = None) -> torch.Tensor:
        r"""Inputs of forward function
        Args:
            x (torch.Tensor(B, L, N)):
                the sequence fed to the positional encoder model (required).
            pos_idx (Tuple[int]):
                position idx indicating the start (first) and end (last) time index of x in a sequence

        Examples:
            >>> output = pos_encoder(x)
        """
        if pos_idx is None:
            x = x + self.pe[:, : x.size(1), :]
        else:
            x = x + self.pe[:, pos_idx[0]: pos_idx[1], :]  # type: ignore[misc]
        return self.dropout(x)
