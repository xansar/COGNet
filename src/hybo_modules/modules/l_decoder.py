# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: l_decoder.py
@time: 2023/5/15 15:09
@e-mail: xansar@ruc.edu.cn
"""
import torch
import torch.nn as nn

from .l_multihead_att import LorentzMultiHeadedAttention
from .hyper_nets import LorentzLinear, LorentzPositionwiseFeedForward

class LorentzTransformerDecoderLayer(nn.Module):
    """Transformer Decoder layer block in Pre-Norm style.
    Pre-Norm style is an improvement w.r.t. Original paper's Post-Norm style,
    providing better converge speed and performance. This is also the actual
    implementation in tensor2tensor and also avalable in fairseq.
    See https://tunz.kr/post/4 and :cite:`DeeperTransformer`.

    .. mermaid::

        graph LR
        %% "*SubLayer" can be self-attn, src-attn or feed forward block
            A(input) --> B[Norm]
            B --> C["*SubLayer"]
            C --> D[Drop]
            D --> E((+))
            A --> E
            E --> F(out)


    Args:
        d_model (int): the dimension of keys/values/queries in
            :class:`MultiHeadedAttention`, also the input size of
            the first-layer of the :class:`PositionwiseFeedForward`.
        heads (int): the number of heads for MultiHeadedAttention.
        d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        self_attn_type (string): type of self-attention scaled-dot, average
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
    """
    def __init__(self,
                 manifold,
                 d_model,
                 heads,
                 d_ff,
                 dropout,
                 attention_dropout,
                 self_attn_type="scaled-dot",
                 max_relative_positions=0,
                 aan_useffn=False,
                 full_context_alignment=False,
                 alignment_heads=0):
        super(LorentzTransformerDecoderLayer, self).__init__()

        self.manifold = manifold

        self.self_attn = LorentzMultiHeadedAttention(
            heads,
            d_model,
            dropout=attention_dropout,
            manifold=self.manifold,
            max_relative_positions=max_relative_positions)
        self.context_attn = LorentzMultiHeadedAttention(
            heads, d_model, dropout=attention_dropout, manifold=self.manifold)

        self.feed_forward = LorentzPositionwiseFeedForward(
            d_model, d_ff, self.manifold, dropout)
        self.full_context_alignment = full_context_alignment
        self.alignment_heads = alignment_heads
        self.residual_1 = LorentzLinear(self.manifold, d_model, d_model, head_num=heads, merge=True, dropout=dropout, bias=False)
        self.residual_2 = LorentzLinear(self.manifold, d_model, d_model, head_num=heads, merge=True, dropout=dropout, bias=False)

    def _forward(self,
                 inputs,
                 memory_bank,
                 src_pad_mask,
                 tgt_pad_mask,
                 layer_cache=None,
                 step=None,
                 future=False):
        """ A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            inputs (FloatTensor): ``(batch_size, T, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (bool): ``(batch_size, 1, src_len)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, src_len)``

        """
        dec_mask = None

        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            if not future:  # apply future_mask, result mask in (B, T, T)
                future_mask = torch.ones([tgt_len, tgt_len],
                                         device=tgt_pad_mask.device,
                                         dtype=torch.uint8)
                future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                # BoolTensor was introduced in pytorch 1.2
                try:
                    future_mask = future_mask.bool()
                except AttributeError:
                    pass
                dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
            else:  # only mask padding, result mask in (B, 1, T)
                dec_mask = tgt_pad_mask

        query, _ = self.self_attn(inputs,
                                  inputs,
                                  inputs,
                                  mask=dec_mask,
                                  layer_cache=layer_cache,
                                  attn_type="self")
        query = self.residual_1(query, inputs)
        mid, attns = self.context_attn(memory_bank,
                                       memory_bank,
                                       query,
                                       mask=src_pad_mask,
                                       layer_cache=layer_cache,
                                       attn_type="context")
        mid = self.residual_2(mid, query)
        output = self.feed_forward(mid)

        return output, attns

    def forward(self, *args, **kwargs):
        """ Extend `_forward` for (possibly) multiple decoder pass:
        Always a default (future masked) decoder forward pass,
        Possibly a second future aware decoder pass for joint learn
        full context alignement, :cite:`garg2019jointly`.

        Args:
            * All arguments of _forward.
            with_align (bool): whether return alignment attention.

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * output ``(batch_size, T, model_dim)``
            * top_attn ``(batch_size, T, src_len)``
            * attn_align ``(batch_size, T, src_len)`` or None
        """
        with_align = kwargs.pop('with_align', False)
        output, attns = self._forward(*args, **kwargs)
        top_attn = attns[:, 0, :, :]  # .contiguous()
        attn_align = None
        if with_align:
            assert False
            if self.full_context_alignment:
                # return _, (B, Q_len, K_len)
                _, attns = self._forward(*args, **kwargs, future=True)

            if self.alignment_heads > 0:
                attns = attns[:, :self.alignment_heads, :, :]  # .contiguous()
            # layer average attention across heads, get ``(B, Q, K)``
            # Case 1: no full_context, no align heads -> layer avg baseline
            # Case 2: no full_context, 1 align heads -> guided align
            # Case 3: full_context, 1 align heads -> full cte guided align
            attn_align = attns.mean(dim=1)
        return output, top_attn, attn_align

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.context_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout
        # pass