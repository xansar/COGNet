# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: hypergraph_layers.py
@time: 2023/5/3 20:36
@e-mail: xansar@ruc.edu.cn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GATv2Conv, GCNConv
import torch.sparse as tsp


import gc

class HGAT(nn.Module):
    def __init__(self, embed_dim, n_layers, p=0.5):
        super(HGAT, self).__init__()
        self.n_layers = n_layers
        self.hgcn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.hgcn_layers.append(
                HypergraphConv(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    use_attention=False,
                    heads=2, dropout=p,
                    concat=False
                )
            )

    def forward(self, X, adj_indices, edge_weight=None, edge_attr=None):
        res_embed = X
        for i in range(self.n_layers):
            X = F.leaky_relu(self.hgcn_layers[i](X, adj_indices, edge_weight, edge_attr)) + X
            res_embed = res_embed + X * (1 / (i + 2))
        # res_embed = torch.mean(torch.cat(res_embed, dim=0).reshape(-1, *res_embed[0].shape), dim=0)
        return res_embed

class GAT(nn.Module):
    def __init__(self, embed_dim, n_layers, p=0.5, edge_dim=None):
        super(GAT, self).__init__()
        self.n_layers = n_layers
        self.gcn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gcn_layers.append(
                GCNConv(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                )
            )
            # self.gcn_layers.append(
            #     GATv2Conv(
            #         in_channels=embed_dim,
            #         out_channels=embed_dim,
            #         heads=2, dropout=p,
            #         edge_dim=edge_dim,
            #         concat=False
            #     )
            # )

    def forward(self, X, adj_indices, edge_attr=None):
        res_embed = X
        for i in range(self.n_layers):
            X = F.leaky_relu(self.gcn_layers[i](X, adj_indices, edge_attr)) + X
            res_embed = res_embed + X * (1 / (i + 2))
        # res_embed = torch.mean(torch.cat(res_embed, dim=0).reshape(-1, *res_embed[0].shape), dim=0)
        return res_embed

class HyperGraphEmbed(nn.Module):
    def __init__(self, embedding_dim, n_layers, H, A, ddi_A, p=0.5):
        super(HyperGraphEmbed, self).__init__()
        # 下面的MLP用来作为门控
        self.h_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )
        self.l_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )

        self.H = H
        self.A = A
        self.ddi_A = ddi_A

        self.hgat = HGAT(embedding_dim, n_layers, p)
        self.gat = GAT(embedding_dim, n_layers, p, edge_dim=1)
        self.ddi_gat = GAT(embedding_dim, n_layers, p)

    def get_hyperedge_representation(self, embed):
        # 获取超边的表示，通过聚合当前超边下所有item的embedding
        # 实际上就是乘以H(n_edges, n_items)
        # embed: n_items, dim
        n_items, n_edges = self.H.shape
        norm_factor = (tsp.sum(self.H, dim=0) ** -1).to_dense().reshape(n_edges, -1)

        assert norm_factor.shape == (n_edges, 1)
        omega = norm_factor * tsp.mm(self.H.T, embed)
        return omega

    def forward(self, diseases_embed, pros_embed, meds_embed):
        diseases_size, pros_size, meds_size = diseases_embed.shape[0], pros_embed.shape[0], meds_embed.shape[0]
        embed = torch.vstack([diseases_embed, pros_embed, meds_embed])
        # ddi卷积
        ddi_med_embedding = self.ddi_gat(meds_embed, self.ddi_A.indices())

        # 进行超图卷积
        h_embed = self.h_gate(embed) * embed
        l_embed = self.l_gate(embed) * embed


        # 超图注意力卷积
        hyper_rep = self.get_hyperedge_representation(h_embed)
        h_embed = self.hgat(h_embed, self.H.indices(), None, hyper_rep)  # n_items, dim

        # 均值聚合
        hyper_rep = self.get_hyperedge_representation(h_embed)  # n_edges, dim
        # 线图
        l_embed = self.get_hyperedge_representation(l_embed)  # n_hyperedge, embed_dim
        linear_rep = self.gat(l_embed, self.A.indices(), self.A.values())  # n_edge, dim   # 可以看做药物集合层面的每个超边的表示

        return hyper_rep, linear_rep, *h_embed.vsplit([diseases_size, pros_size + diseases_size]), ddi_med_embedding