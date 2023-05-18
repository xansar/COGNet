# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: l_gnn_layers.py
@time: 2023/5/15 20:25
@e-mail: xansar@ruc.edu.cn
"""
import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module

from ..manifolds import Lorentz

class LorentzHyperGraphEmbed(nn.Module):
    def __init__(self, manifold, embedding_dim, n_layers, H, A, ddi_A, p=0., use_att=False):
        super(LorentzHyperGraphEmbed, self).__init__()
        # 下面的MLP用来作为门控
        self.manifold = manifold
        self.h_gate = nn.Sequential(
            LorentzGNNLinear(self.manifold, embedding_dim, embedding_dim, dropout=p),
            nn.Sigmoid()
        )

        self.l_gate = nn.Sequential(
            LorentzGNNLinear(self.manifold, embedding_dim, embedding_dim, dropout=p),
            nn.Sigmoid()
        )

        self.H = H
        self.A = A
        self.ddi_A = ddi_A

        self.hgcn = nn.Sequential()
        self.gcn = nn.Sequential()
        self.ddi_gcn = nn.Sequential()
        for i in range(n_layers):
            self.hgcn.add_module(f'hgcn_{i}',
                                 LorentzGraphConvolution(embedding_dim, embedding_dim, use_bias=True, dropout=p, use_att=use_att, local_agg=False, is_hyper=True))
            self.gcn.add_module(f'gcn_{i}',
                                 LorentzGraphConvolution(embedding_dim, embedding_dim, use_bias=True, dropout=p, use_att=use_att, local_agg=False))
            self.ddi_gcn.add_module(f'ddi_gcn_{i}',
                                 LorentzGraphConvolution(embedding_dim, embedding_dim, use_bias=True, dropout=p, use_att=use_att, local_agg=False))

        self.hyper_graph_agg = LorentzAgg(self.manifold, embedding_dim, p, False, False)

    def get_hyperedge_representation(self, embed):
        omega = self.hyper_graph_agg(embed, self.H.T)
        # # 获取超边的表示，通过聚合当前超边下所有item的embedding
        # # 实际上就是乘以H(n_edges, n_items)
        # # embed: n_items, dim
        # n_items, n_edges = self.H.shape
        # norm_factor = (tsp.sum(self.H, dim=0) ** -1).to_dense().reshape(n_edges, -1)
        #
        # assert norm_factor.shape == (n_edges, 1)
        # omega = norm_factor * tsp.mm(self.H.T, embed)
        return omega

    def forward(self, diseases_embed, pros_embed, meds_embed):
        diseases_size, pros_size, meds_size = diseases_embed.shape[0], pros_embed.shape[0], meds_embed.shape[0]
        embed = torch.vstack([diseases_embed, pros_embed, meds_embed])

        # ddi卷积
        # ddi_med_embedding = self.ddi_gcn(meds_embed, self.ddi_A.indices())
        ddi_med_embedding, _ = self.ddi_gcn((meds_embed, self.ddi_A))

        # 进行超图卷积
        # 元素乘积这个操作不一定能保证在双曲面上
        h_embed = self.h_gate(embed) * embed
        self.manifold._check_point_on_manifold(h_embed)
        l_embed = self.l_gate(embed) * embed
        self.manifold._check_point_on_manifold(h_embed)

        # 超图卷积
        # hyper_rep = self.get_hyperedge_representation(h_embed)
        # h_embed = self.hgcn(h_embed, self.H.indices(), None, hyper_rep)  # n_items, dim
        h_embed, _ = self.hgcn((h_embed, self.H))

        # 均值聚合
        hyper_rep = self.get_hyperedge_representation(h_embed)  # n_edges, dim
        # 线图
        l_embed = self.get_hyperedge_representation(l_embed)  # n_hyperedge, embed_dim
        # linear_rep = self.gcn(l_embed, self.A.indices(), self.A.values())  # n_edge, dim   # 可以看做药物集合层面的每个超边的表示
        linear_rep, _ = self.gcn((l_embed, self.A))

        return hyper_rep, linear_rep, *h_embed.vsplit([diseases_size, pros_size + diseases_size]), ddi_med_embedding

class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_features, out_features, use_bias, dropout, use_att, local_agg, nonlin=None, is_hyper=False):
        super(LorentzGraphConvolution, self).__init__()
        manifold = Lorentz()
        self.is_hyper = is_hyper
        self.linear = LorentzGNNLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_features, dropout, use_att, local_agg)
        # self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear(x)
        if self.is_hyper:
            # 超图先乘转置再乘原始，计算中点的时候算了归一化，所以不需要再做显式的normalization
            h = self.agg(h, adj.T)
        h = self.agg(h, adj)
        # h = self.hyp_act.forward(h)
        output = h, adj
        return output

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class LorentzGNNLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)

class LorentzAgg(Module):
    """
    Lorentz aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout, use_att, local_agg):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)
            self.key_linear = LorentzGNNLinear(manifold, in_features, in_features)
            self.query_linear = LorentzGNNLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):
        # x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:  # 从注释里看，local_agg的意思应该是在切空间做agg
                # x_local_tangent = []
                # # for i in range(x.size(0)):
                # #     x_local_tangent.append(self.manifold.logmap(x[i], x))
                # # x_local_tangent = torch.stack(x_local_tangent, dim=0)
                # x_local_tangent = self.manifold.clogmap(x, x)
                # # import pdb; pdb.set_trace()
                # adj_att = self.att(x, adj)
                # # att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                # support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                # output = self.manifold.expmap(x, support_t)
                # return output
                query = self.query_linear(x)
                key = self.key_linear(x)
                att_adj = 2 + 2 * self.manifold.cinner(query, key)
                att_adj = att_adj / self.scale + self.bias
                att_adj = torch.sigmoid(att_adj)
                att_adj = torch.mul(adj.to_dense(), att_adj)
                support_t = torch.matmul(att_adj, x)
            else:
                adj_att = self.att(x, adj)
                support_t = torch.matmul(adj_att, x)
        else:
            support_t = torch.spmm(adj, x)
        # output = self.manifold.expmap0(support_t, c=self.c)
        denom = (-self.manifold.inner(None, support_t, keepdim=True))   # 这里跟计算中点完全一致，那么需要对adj进行normalization
        denom = denom.abs().clamp_min(1e-8).sqrt()
        output = support_t / denom
        return output

    def attention(self, x, adj):
        pass

class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.in_features = in_features

    def forward(self, x, adj):
        n = x.size(0)
        # n x 1 x d
        x_left = torch.unsqueeze(x, 1)
        x_left = x_left.expand(-1, n, -1)
        # 1 x n x d
        x_right = torch.unsqueeze(x, 0)
        x_right = x_right.expand(n, -1, -1)

        x_cat = torch.cat((x_left, x_right), dim=2)
        att_adj = self.linear(x_cat).squeeze()
        att_adj = torch.sigmoid(att_adj)
        att_adj = torch.mul(adj.to_dense(), att_adj)
        return att_adj
