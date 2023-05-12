# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: hypergraph_analyse.py
@time: 2023/5/11 14:40
@e-mail: xansar@ruc.edu.cn
"""
from collections import defaultdict
import dill
import numpy as np

import torch
import torch.sparse as tsp
import matplotlib.pyplot as plt

from hypergraph_construction import construct_graphs


def hypergraph_analyse(H):

    # H_2 = tsp.mm(H, H.T)  两层卷积后，矩阵稠密程度大概有90%
    H = H.to_dense().int()
    # 统计节点度数
    node_degrees = torch.sum(H, dim=1)

    # 统计超边规模
    hyperedge_sizes = torch.sum(H, dim=0)

    # 绘制节点度数分布图
    node_degree_counts = torch.bincount(node_degrees)
    node_degree_distribution = node_degree_counts.float()
    plt.plot(node_degree_distribution.numpy())
    plt.xlabel('Node Degree')
    plt.ylabel('Count')
    plt.title('Node Degree Distribution')
    plt.show()

    # 绘制超边规模分布图
    hyperedge_size_counts = torch.bincount(hyperedge_sizes)
    hyperedge_size_distribution = hyperedge_size_counts.float()
    plt.plot(hyperedge_size_distribution.numpy())
    plt.xlabel('Hyperedge Size')
    plt.ylabel('Count')
    plt.title('Hyperedge Size Distribution')
    plt.show()
    """
    超边分布在20-50之间最多
    """


def main():
    # load data
    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'
    cache_pth = '../data/graphs.pkl'

    # ehr_adj_path = '../data/weighted_ehr_adj_final.pkl'
    ehr_adj_path = '../data/ehr_adj_final.pkl'
    ddi_adj_path = '../data/ddi_A_final.pkl'
    ddi_mask_path = '../data/ddi_mask_H.pkl'
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    # frequency statistic
    med_count = defaultdict(int)
    for patient in data:
        for adm in patient:
            for med in adm[2]:
                med_count[med] += 1

    ## rare first
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur_medications = sorted(data[i][j][2], key=lambda x: med_count[x])
            data[i][j][2] = cur_medications

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    END_TOKEN = voc_size[2] + 1
    DIAG_PAD_TOKEN = voc_size[0] + 2
    PROC_PAD_TOKEN = voc_size[1] + 2
    MED_PAD_TOKEN = voc_size[2] + 2
    SOS_TOKEN = voc_size[2]
    TOKENS = [END_TOKEN, DIAG_PAD_TOKEN, PROC_PAD_TOKEN, MED_PAD_TOKEN, SOS_TOKEN]


    H, A = construct_graphs(
        cache_pth=cache_pth,
        data_train=data_train,
        n_diag=voc_size[0] + 3,  # +3是考虑pad token
        n_pro=voc_size[1] + 3,
        n_drug=voc_size[2] + 3
    )

    hypergraph_analyse(H)



if __name__ == '__main__':
    main()