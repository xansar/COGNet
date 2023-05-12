# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: hypergraph_construction.py
@time: 2023/5/5 18:24
@e-mail: xansar@ruc.edu.cn
"""
import dgl
import torch
import numpy as np
from tqdm import tqdm
import os
def construct_graphs(cache_pth, data_train, n_diag, n_pro, n_drug):
    if os.path.exists(cache_pth):
        graphs = dgl.data.utils.load_info(cache_pth)
        hyper_graph_coo, line_graph_coo, line_graph_weight = graphs['hyper_graph_coo'], graphs['line_graph_coo'], \
                                                             graphs['line_graph_weight']
    else:
        # 这里要注意，因为模型的embedding有pad token end token在，这里要注意n_diag的实际值
        # 构建诊断和手术联合超图
        hyper_edges_lst = []
        hyper_graph_coo = []  # [(item_idx, hyper_edge_idx)]
        train_data_with_h_edge_idx = []  # 原始train_data的每一个visit中有[diag, pro, drug]，这里变成[diag, pro, drug, h_edge_idx]
        total_num = 0
        repeat_num = 0
        for u, visits in enumerate(data_train):
            hyper_edges_in_cur_ehr = []
            visits_with_idx = []
            for t, v in enumerate(visits):
                # 这里v是[诊断，手术，药物]
                # 1961, 1433, 134
                diag, pro, drug = v  # drug暂时用不到
                pro = [p + n_diag for p in pro]  # 将pro的编码与diag的编码接到一起，用来构建异构超图
                drug = [d + n_diag + n_pro for d in drug]
                hyper_edge = sorted(diag + pro + drug)  # 异构超边，包含diag和pro
                total_num += 1
                if hyper_edge not in hyper_edges_lst:
                    hyper_edges_lst.append(hyper_edge)  # 加入超边集合
                    hyper_edge_idx = len(hyper_edges_lst) - 1  # 此时新超边的序号就是超边列表的最后一个位置的序号
                else:
                    hyper_edge_idx = hyper_edges_lst.index(hyper_edge)
                    repeat_num += 1
                for item in hyper_edge:
                    hyper_graph_coo.append([item, hyper_edge_idx])
                v.append(hyper_edge_idx)
                visits_with_idx.append(v)
            # 记录当前病人的visits包含了哪些超边
            train_data_with_h_edge_idx.append(visits_with_idx)
        print(total_num, repeat_num)    # 10489,0   意思就是基本所有的visit表示的超边都是唯一的
        # 构建线图
        # 基于hyper_edges_lst，计算任意一对超对之间的相似度
        line_graph_coo = []  # [(hyper_edge_idx_i, hyper_edge_idx_j, weight)]
        num = (len(hyper_edges_lst) - 1) * len(hyper_edges_lst) / 2
        bar = tqdm(total=num, desc='line graph')
        for i in range(len(hyper_edges_lst)):
            for j in range(i + 1, len(hyper_edges_lst)):
                edge_i = set(hyper_edges_lst[i])
                edge_j = set(hyper_edges_lst[j])
                intersection = len(edge_i & edge_j)
                if intersection != 0:
                    union = len(edge_i | edge_j)
                    sim = intersection / union
                    line_graph_coo.append([i, j, sim])
                bar.update(1)

        # 将稀疏邻接矩阵转换成numpy array
        # 超图
        hyper_graph_coo = np.array(hyper_graph_coo, dtype=np.int64).T
        line_graph_coo = np.array(line_graph_coo, dtype=np.float32).T
        line_graph_weight = line_graph_coo[2, :]
        line_graph_coo = line_graph_coo[:2, :].astype(np.int64)
        dgl.data.utils.save_info(
            cache_pth,
            {
                 'hyper_graph_coo': hyper_graph_coo,
                 'line_graph_coo': line_graph_coo,
                 'line_graph_weight': line_graph_weight
            }
        )
    H_i = torch.from_numpy(hyper_graph_coo)
    H_v = torch.ones(H_i.shape[1])
    H = torch.sparse_coo_tensor(
        indices=H_i,
        values=H_v,
        size=(n_diag + n_pro + n_drug, int(hyper_graph_coo.max(1)[1]) + 1)
    ).coalesce()
    # 注意到这里weight非常稠密，设置阈值做过滤，阈值设置为0.3（启发式）
    mask = line_graph_weight > 0.3
    # A: mean - 0.1626, std - 0.0597, var - 0.0036, max - 0.8125, min - 0.0088, median - 0.16, mode - 0.1667 / 2308830, density - 0.9956
    line_graph_coo = line_graph_coo[:, mask]
    line_graph_weight = line_graph_weight[mask]
    A_i = torch.from_numpy(line_graph_coo)
    A_v = torch.from_numpy(line_graph_weight)
    A = torch.sparse_coo_tensor(
        indices=A_i,
        values=A_v,
        size=(int(hyper_graph_coo.max(1)[1]) + 1, int(hyper_graph_coo.max(1)[1]) + 1)
    )
    A = A + A.T
    A = A.coalesce()

    print('hypergraph construction finished')
    return H, A