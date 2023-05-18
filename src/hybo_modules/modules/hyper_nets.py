# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: hyper_nets.py
@time: 2023/5/15 15:17
@e-mail: xansar@ruc.edu.cn
"""
import torch
import torch.nn as nn


class LorentzLinear(nn.Module):
    """
        Perform the Lorentz linear transformation.

        args:
            in_features, out_features, bias: Same as nn.Linear
            dropout: Dropout rate in lorentz linear
            manifold: THe manifold that the linear layer operated in.
            nonlin: Non-linear function before the linear operation.
            merge: If set to True, it means that the input has the shape of [..., head_num, head_dim], and the output will has the shape of [..., head_num * head_dim]. The heads are merged.
            head_num: If `merge` is set to True, then head_num specifies the number of heads in input, otherwise it means that the output should be split into `head_num` heads, i.e., [..., head_num, head_dim]. If set to 0, then it is a normal lorentz linear layer.
    """
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 nonlin=None,
                 head_num=0,
                 merge=False):
        super().__init__()
        self.nonlin = nonlin
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.head_num = head_num
        self.merge = merge
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * 2.3)
        self.epsilon = 1e-8

    def forward(self, x, bias=None):
        if self.nonlin is not None:
            x = self.nonlin(x)
        if not self.merge:
            x = self.weight(self.dropout(x))
            if self.head_num > 0:
                x = x.view(x.shape[0], x.shape[1], self.head_num, -1)
        else:
            x = self.weight(
                self.dropout(x.flatten(-2)))
        # The following code has some inconsistency to Eq.7 in the paper. When calculating the time axis,
        # we do not consider the bias in Eq.7, while we add the bias before determining time axis.
        # It is a small bug here. However, both methods give mathematically correct results.
        # For reproducibility, we leave it unchanged here.
        if bias is not None:
            x = x + bias
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1) # 在嵌入维度，取最后一维
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1    # 这是取第一维
        # 这里x过dropout后，x_narrow上会有0，当分母就会有nan
        scale = (time * time - self.manifold.k) / \
            ((x_narrow * x_narrow).sum(dim=-1, keepdim=True) + self.epsilon)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 0.02
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        step = self.in_features // self.head_num if self.head_num > 0 else self.in_features
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)

class LorentzPositionwiseFeedForward(nn.Module):
    def __init__(self,
                 d_model,
                 d_ff,
                 manifold,
                 dropout=0.1):
        super(LorentzPositionwiseFeedForward, self).__init__()
        self.manifold = manifold
        self.w_1 = LorentzLinear(manifold,
                                 d_model,
                                 d_ff,
                                 dropout=dropout)
        self.residual = LorentzLinear(manifold, d_ff, d_model, dropout=dropout, nonlin=nn.ReLU())

    def forward(self, x):
        """Layer definition.

        Args:
            x: ``(batch_size, input_len, model_dim)``

        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """
        x_out = self.w_1(x)
        return self.residual(x_out, x)

    def update_dropout(self, dropout):
        pass

class LorentzFusionLinear(nn.Module):
        def __init__(self,
                     d_model,
                     d_ff,
                     manifold,
                     dropout=0.1):
            super(LorentzFusionLinear, self).__init__()
            self.manifold = manifold
            self.w_1 = LorentzLinear(manifold,
                                     d_model,
                                     d_ff,
                                     dropout=dropout)
            self.w_2 = LorentzLinear(manifold,
                                     d_model,
                                     d_ff,
                                     dropout=dropout)
            self.residual = LorentzLinear(manifold, d_ff, d_model, dropout=dropout, nonlin=nn.ReLU())

        def forward(self, x_left, x_right):
            """Layer definition.

            Args:
                x: ``(batch_size, input_len, model_dim)``

            Returns:
                (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
            """
            x_left = self.w_1(x_left)
            x_right = self.w_2(x_right)
            return self.residual(x_left, x_right)

        def update_dropout(self, dropout):
            pass