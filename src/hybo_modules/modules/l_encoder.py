# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: l_encoder.py
@time: 2023/5/15 15:10
@e-mail: xansar@ruc.edu.cn
"""
import torch.nn as nn

from .l_multihead_att import LorentzMultiHeadedAttention
from .hyper_nets import LorentzLinear, LorentzPositionwiseFeedForward

class LorentzTransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, manifold, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0):
        super(LorentzTransformerEncoderLayer, self).__init__()

        self.manifold = manifold
        self.self_attn = LorentzMultiHeadedAttention(
            heads, d_model, self.manifold, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = LorentzPositionwiseFeedForward(d_model, d_ff, self.manifold, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.residual = LorentzLinear(manifold, d_model, d_model, dropout=dropout, head_num=heads, merge=True, bias=False)


    def forward(self, inputs, src_mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        context, _ = self.self_attn(inputs, inputs, inputs,
                                    mask=src_mask, attn_type="self")
        context = self.residual(context, inputs)
        output = self.feed_forward(context)
        return output

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout