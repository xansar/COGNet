# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: l_med_transformer_decoder.py
@time: 2023/5/15 18:52
@e-mail: xansar@ruc.edu.cn
"""
import torch
import torch.nn as nn

from . import LorentzMultiHeadedAttention, LorentzLinear, LorentzPositionwiseFeedForward


class LorentzMedTransformerDecoder(nn.Module):
    def __init__(self, manifold, d_model, nhead, dropout=0.1,
                 layer_norm_eps=1e-5) -> None:
        super(LorentzMedTransformerDecoder, self).__init__()
        self.manifold = manifold
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.m2d_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.m2p_multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn = LorentzMultiHeadedAttention(nhead, d_model, dropout=dropout, manifold=self.manifold)
        self.m2d_multihead_attn = LorentzMultiHeadedAttention(nhead, d_model, dropout=dropout, manifold=self.manifold)
        self.m2p_multihead_attn = LorentzMultiHeadedAttention(nhead, d_model, dropout=dropout, manifold=self.manifold)

        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.feed_forward = LorentzPositionwiseFeedForward(
            d_model, d_model, self.manifold, dropout)  # 这个里面包含两层fc

        # 双曲没有layernorm，因为均值和方差不好定义
        # self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.residual_1 = LorentzLinear(self.manifold, d_model, d_model, head_num=nhead, merge=True, dropout=dropout, bias=False)
        self.residual_2 = LorentzLinear(self.manifold, d_model, d_model, head_num=nhead, merge=True, dropout=dropout, bias=False)
        self.residual_3 = LorentzLinear(self.manifold, d_model, d_model, head_num=nhead, merge=True, dropout=dropout, bias=False)
        self.residual_4 = LorentzLinear(self.manifold, d_model, d_model, dropout=dropout, bias=False)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 双曲本身非线性，不要激活
        # self.activation = nn.ReLU()
        self.nhead = nhead

        # self.align = nn.Linear(d_model, d_model)

    def forward(self, input_medication_embedding, input_medication_memory, input_disease_embdding, input_proc_embedding,
                input_medication_self_mask, d_mask, p_mask):
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            input_medication_embedding: [*, max_med_num+1, embedding_size]
        Shape:
            see the docs in Transformer class.
        """
        input_len = input_medication_embedding.size(0)
        tgt_len = input_medication_embedding.size(1)

        # [batch_size*visit_num, max_med_num+1, max_med_num+1]
        subsequent_mask = self.generate_square_subsequent_mask(tgt_len, input_len,
                                                               input_disease_embdding.device)
        self_attn_mask = subsequent_mask + input_medication_self_mask

        x = input_medication_embedding + input_medication_memory

        # x = self.norm1(x + self._sa_block(x, self_attn_mask))
        self_attn_output = self._sa_block(x, self_attn_mask)
        x = self.residual_1(self_attn_output, x)   # 这里因为洛伦兹模型加法没有定义，采用变换近似，加x变成偏置

        # attentioned_disease_embedding = self._m2d_mha_block(x, input_disease_embdding, d_mask)
        # attentioned_proc_embedding = self._m2p_mha_block(x, input_proc_embedding, p_mask)
        # x = self.norm3(x + self._ff_block(torch.cat([attentioned_disease_embedding, self.align(attentioned_proc_embedding)], dim=-1)))
        # 公式12，这里把visit展到第一维上，那么做注意力时不会跨visit计算
        # x = self.norm2(
        #     x + self._m2d_mha_block(x, input_disease_embdding, d_mask) + self._m2p_mha_block(x, input_proc_embedding,
        #                                                                                      p_mask))
        m2d_attn_output = self._m2d_mha_block(x, input_disease_embdding, d_mask)
        x = self.residual_2(m2d_attn_output, x)
        m2p_attn_output = self._m2p_mha_block(x, input_proc_embedding, p_mask)
        x = self.residual_3(m2p_attn_output, x)


        # x = self.norm3(x + self._ff_block(x))
        ff_output = self._ff_block(x)
        x = self.residual_4(ff_output, x)
        return x

    # self-attention block
    def _sa_block(self, x, attn_mask):
        # x = self.self_attn(x, x, x,
        #                    attn_mask=attn_mask,
        #                    need_weights=False)[0]
        x, _ = self.self_attn(x, x, x,
                              mask=attn_mask,
                              attn_type="self")
        return self.dropout1(x)

    # multihead attention block
    def _m2d_mha_block(self, x, mem, attn_mask):
        # x = self.m2d_multihead_attn(x, mem, mem,
        #                             attn_mask=attn_mask,
        #                             need_weights=False)[0]
        x, _ = self.m2d_multihead_attn(query=x, key=mem, value=mem,
                                       mask=attn_mask,
                                       attn_type="context")
        return self.dropout2(x)

    def _m2p_mha_block(self, x, mem, attn_mask):
        # x = self.m2p_multihead_attn(x, mem, mem,
        #                             attn_mask=attn_mask,
        #                             need_weights=False)[0]
        x, _ = self.m2p_multihead_attn(query=x, key=mem, value=mem,
                                       mask=attn_mask,
                                       attn_type="context")
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        # x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.feed_forward(x)
        return self.dropout3(x)

    def generate_square_subsequent_mask(self, sz: int, batch_size: int, device):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
        return mask
