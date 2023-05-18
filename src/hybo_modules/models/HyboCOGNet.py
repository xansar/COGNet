# -*- coding: utf-8 -*-

"""
@author: xansar
@software: PyCharm
@file: HyboCOGNet.py
@time: 2023/5/15 18:49
@e-mail: xansar@ruc.edu.cn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from info_nce import InfoNCE

from ..modules import \
    Lorentz, \
    LorentzEmbeddings, \
    LorentzHyperGraphEmbed, \
    LorentzTransformerEncoderLayer, \
    LorentzMedTransformerDecoder, \
    LorentzFusionLinear, \
    LorentzSelfAttend, \
    LorentzLinear


class HyboCOGNet(nn.Module):
    """在CopyDrug_batch基础上将medication的encode部分修改为transformer encoder"""

    def __init__(self, voc_size, ehr_adj, ddi_adj, ddi_mask_H, H, A, emb_dim=64, drop_out=0.3, n_layers=2,
                 device=torch.device('cpu:0')):
        super(HyboCOGNet, self).__init__()
        self.name = 'HyboCOGNet'
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        self.nhead = 2
        self.SOS_TOKEN = voc_size[2]  # start of sentence
        self.END_TOKEN = voc_size[2] + 1  # end   新增的两个编码，两者均是针对于药物的embedding
        self.MED_PAD_TOKEN = voc_size[2] + 2  # 用于embedding矩阵中的padding（全为0）
        self.DIAG_PAD_TOKEN = voc_size[0] + 2
        self.PROC_PAD_TOKEN = voc_size[1] + 2

        self.manifold = Lorentz()

        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)

        # dig_num * emb_dim
        self.diag_embedding = LorentzEmbeddings(self.manifold, emb_dim, voc_size[0] + 3, self.DIAG_PAD_TOKEN, dropout=drop_out)
        # self.diag_embedding = nn.Sequential(
        #     nn.Embedding(voc_size[0] + 3, emb_dim, self.DIAG_PAD_TOKEN),
        #     nn.Dropout(0.3)
        # )

        # proc_num * emb_dim
        self.proc_embedding = LorentzEmbeddings(self.manifold, emb_dim, voc_size[1] + 3, self.PROC_PAD_TOKEN, dropout=drop_out)
        # self.proc_embedding = nn.Sequential(
        #     nn.Embedding(voc_size[1] + 3, emb_dim, self.PROC_PAD_TOKEN),
        #     nn.Dropout(0.3)
        # )

        # med_num * emb_dim
        self.med_embedding = LorentzEmbeddings(self.manifold, emb_dim, voc_size[2] + 3, self.MED_PAD_TOKEN, dropout=drop_out)
        # self.med_embedding = nn.Sequential(
        #     # 添加padding_idx，表示取0向量
        #     nn.Embedding(voc_size[2] + 3, emb_dim, self.MED_PAD_TOKEN),
        #     nn.Dropout(0.3)
        # )

        # 用于对上一个visit的medication进行编码
        self.medication_encoder = LorentzTransformerEncoderLayer(self.manifold, emb_dim, self.nhead, emb_dim, drop_out,
                                                                 drop_out)
        # 用于对当前visit的疾病与症状进行编码
        self.diagnoses_encoder = LorentzTransformerEncoderLayer(self.manifold, emb_dim, self.nhead, emb_dim, drop_out,
                                                                drop_out)
        self.procedure_encoder = LorentzTransformerEncoderLayer(self.manifold, emb_dim, self.nhead, emb_dim, drop_out,
                                                                drop_out)

        # # 用于对上一个visit的medication进行编码
        # self.medication_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)
        # # 用于对当前visit的疾病与症状进行编码
        # self.diagnoses_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)
        # self.procedure_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        ddi_idx = np.where(ddi_adj == 1)
        ddi_i = torch.tensor([ddi_idx[0], ddi_idx[1]], dtype=torch.long)
        ddi_v = torch.ones(ddi_i.shape[1])
        self.tsp_ddi_adj = torch.sparse_coo_tensor(
            indices=ddi_i,
            values=ddi_v,
            size=(self.MED_PAD_TOKEN + 1, self.MED_PAD_TOKEN + 1)
        ).coalesce().to(device)

        H = H.to(device)
        A = A.to(device)

        # self.hgcn = HyperGraphEmbed(embedding_dim=emb_dim, n_layers=n_layers, H=H, A=A, ddi_A=self.tsp_ddi_adj)
        self.hgcn = LorentzHyperGraphEmbed(manifold=self.manifold, embedding_dim=emb_dim, n_layers=n_layers, H=H, A=A,
                                           ddi_A=self.tsp_ddi_adj, use_att=False)

        # 写一个两个view的embed融合的模块
        self.d_global_local_fusion_layer = LorentzFusionLinear(emb_dim, emb_dim, self.manifold, drop_out)
        self.p_global_local_fusion_layer = LorentzFusionLinear(emb_dim, emb_dim, self.manifold, drop_out)
        self.m_1_global_local_fusion_layer = LorentzFusionLinear(emb_dim, emb_dim, self.manifold, drop_out)
        self.m_2_global_local_fusion_layer = LorentzFusionLinear(emb_dim, emb_dim, self.manifold, drop_out)

        # self.d_global_local_fusion_layer = nn.Sequential(
        #     nn.Linear(emb_dim * 2, emb_dim),
        #     nn.LeakyReLU()
        # )
        #
        # self.p_global_local_fusion_layer = nn.Sequential(
        #     nn.Linear(emb_dim * 2, emb_dim),
        #     nn.LeakyReLU()
        # )
        #
        # self.m_global_local_fusion_layer = nn.Sequential(
        #     nn.Linear(emb_dim * 2, emb_dim),
        #     nn.LeakyReLU()
        # )
        #
        # self.m_2_global_local_fusion_layer = nn.Sequential(
        #     nn.Linear(emb_dim * 2, emb_dim),
        #     nn.LeakyReLU()
        # )

        self.inter = nn.Parameter(torch.FloatTensor([1]))

        # 这两个参数是用来计算cross visit attention scores的时候用的
        self.scale_d = nn.Parameter(torch.tensor([math.sqrt(emb_dim)]))
        self.scale_p = nn.Parameter(torch.tensor([math.sqrt(emb_dim)]))
        self.scale_copy = nn.Parameter(torch.tensor([math.sqrt(emb_dim)]))
        self.bias_d = nn.Parameter(torch.zeros(()))
        self.bias_p = nn.Parameter(torch.zeros(()))
        self.bias_copy = nn.Parameter(torch.zeros(()))

        # 聚合单个visit内的diag和proc得到visit-level的表达
        self.diag_self_attend = LorentzSelfAttend(self.manifold, emb_dim)
        self.proc_self_attend = LorentzSelfAttend(self.manifold, emb_dim)

        # self.diag_self_attend = SelfAttend(emb_dim)
        # self.proc_self_attend = SelfAttend(emb_dim)

        self.decoder = LorentzMedTransformerDecoder(self.manifold, emb_dim, self.nhead, dropout=drop_out)
        # self.decoder = MedTransformerDecoder(emb_dim, self.nhead, dim_feedforward=emb_dim * 2, dropout=0.2,
        #                                      layer_norm_eps=1e-5)

        self.info_nce_loss = InfoNCE()

        # 用于对每一个visit的diagnoses进行编码

        # weights
        self.Wo = LorentzLinear(self.manifold,
                                emb_dim,
                                voc_size[2] + 2)  # generate mode
        self.Wc = LorentzLinear(self.manifold, emb_dim, emb_dim)  # copy mode

        # self.Wo = nn.Linear(emb_dim, voc_size[2] + 2)  # generate mode
        # self.Wc = nn.Linear(emb_dim, emb_dim)  # copy mode

        # swtich network to calculate generate probablity
        self.W_z = LorentzLinear(self.manifold, emb_dim, 1)
        # self.W_z = nn.Linear(emb_dim, 1)

    def cache_embedding_for_eval(self):
        diseases_embed = self.diag_embedding(torch.arange(self.DIAG_PAD_TOKEN + 1).to(self.device), use_position_encoding=False)
        pros_embed = self.proc_embedding(torch.arange(self.PROC_PAD_TOKEN + 1).to(self.device), use_position_encoding=False)
        meds_embed = self.med_embedding(torch.arange(self.MED_PAD_TOKEN + 1).to(self.device), use_position_encoding=False)
        # 这里需要区分验证和测试，测试的时候显然不需要每个
        # 这里计算是按照batch来的，一个batch计算一次，但是验证的时候，应该是一个epoch计算一次就可以了
        # 因此需要将这几个embedding，存储下来
        _, _, disease_embed, pro_embed, med_embed, ddi_embedding = self.hgcn(diseases_embed, pros_embed,
                                                                             meds_embed)
        self.disease_embed = disease_embed
        self.pro_embed = pro_embed
        self.med_embed = med_embed
        self.ddi_embedding = ddi_embedding

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = score(sess_emb_hgnn, sess_emb_lgcn)
        neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))  # 感觉shuffle效果一般
        one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1).to(self.device)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        if torch.isnan(con_loss).any():
            con_loss = torch.tensor(0., dtype=con_loss.dtype, device=con_loss.device)

        return con_loss

    def infoNCELoss(self, view_1_embed, view_2_embed):
        assert view_1_embed.shape == view_2_embed.shape
        if len(view_1_embed.shape) > 2:
            embed_dim = view_1_embed.shape[-1]
            view_1_embed = view_1_embed.reshape(-1, embed_dim)
            view_2_embed = view_2_embed.reshape(-1, embed_dim)
        loss = self.info_nce_loss(view_1_embed, view_2_embed)
        return loss

    def encode(self, diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length,
               dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask,
               stay_proc_mask, max_len=20):
        device = self.device
        # batch维度以及seq维度上并行计算（现在不考虑时间序列信息），每一个medication序列仍然按顺序预测
        batch_size, max_visit_num, max_med_num = medications.size()
        max_diag_num = diseases.size()[2]
        max_proc_num = procedures.size()[2]

        ############################ 数据预处理 ########################
        # TODO：考虑超边表示怎么融入，两种方法：
        #       1. 超边表示作为对比学习损失使用
        #             a. 那么嵌入表示需要使用item-level的
        #             b. 需要补充一个对比学习损失函数
        #       2. 超边表示作为和visit-level相同级别的嵌入使用
        #             a. 需要引入一个新的张量(batch_size, max_T)，来记录每条数据包含哪些超边

        ## 初始embed，用于后面local和global的encode
        disease_embed = self.diag_embedding(torch.arange(self.DIAG_PAD_TOKEN + 1).to(device), use_position_encoding=False)
        pro_embed = self.proc_embedding(torch.arange(self.PROC_PAD_TOKEN + 1).to(device), use_position_encoding=False)
        med_embed = self.med_embedding(torch.arange(self.MED_PAD_TOKEN + 1).to(device), use_position_encoding=False)
        hyper_rep, linear_rep, h_disease_embed, h_pro_embed, h_med_embed, ddi_embedding = self.hgcn(disease_embed,
                                                                                                    pro_embed,
                                                                                                    med_embed)
        # if self.training:
        #     # 训练状态下，每个batch都会更新embedding
        #     self.disease_embed = None
        #     self.pro_embed = None
        #     self.med_embed = None
        #     self.ddi_embedding = None
        #
        #     # 这里需要区分验证和测试，测试的时候显然不需要每个
        #     # 这里计算是按照batch来的，一个batch计算一次，但是验证的时候，应该是一个epoch计算一次就可以了
        #     # 因此需要将这几个embedding，存储下来
        #     hyper_rep, linear_rep, h_disease_embed, h_pro_embed, h_med_embed, ddi_embedding = self.hgcn(disease_embed,
        #                                                                                                 pro_embed,
        #                                                                                                 med_embed)
        #
        #     # 计算对比学习损失
        #     # con_loss = self.SSL(hyper_rep, linear_rep)
        #     con_loss = self.infoNCELoss(hyper_rep, linear_rep)
        #     # con_loss = torch.ones(1, device=device)
        # else:
        #     # 测试状态下，不需要重复计算图embedding
        #     h_disease_embed = self.disease_embed
        #     h_pro_embed = self.pro_embed
        #     h_med_embed = self.med_embed
        #     ddi_embedding = self.ddi_embedding
        #     con_loss = torch.nan

        input_disease_embedding = disease_embed[diseases].view(batch_size * max_visit_num, max_diag_num,
                                                               self.emb_dim)  # [batch, seq, max_diag_num, emb]
        input_proc_embedding = pro_embed[procedures].view(batch_size * max_visit_num, max_proc_num, self.emb_dim)

        # 1. 对当前的disease与procedure进行编码
        # 这里相当于把每条数据中多个visit都展开了
        # input_disease_embedding = self.diag_embedding(diseases).view(batch_size * max_visit_num, max_diag_num, self.emb_dim)      # [batch, seq, max_diag_num, emb]
        # input_proc_embedding = self.proc_embedding(procedures).view(batch_size * max_visit_num, max_proc_num, self.emb_dim)      # [batch, seq, max_proc_num, emb]
        # _mask_matrix,用来标记数据中哪些是补的pad
        d_enc_mask_matrix = d_mask_matrix.view(batch_size * max_visit_num, max_diag_num).unsqueeze(
                dim=1).repeat(1, max_diag_num, 1)  # [batch*seq, input_length, output_length]
        d_enc_mask_matrix = d_enc_mask_matrix.view(batch_size * max_visit_num, max_diag_num, max_diag_num)
        p_enc_mask_matrix = p_mask_matrix.view(batch_size * max_visit_num, max_proc_num).unsqueeze(
                dim=1).repeat(1, max_proc_num, 1)  # [batch*seq, input_length, output_length]
        p_enc_mask_matrix = p_enc_mask_matrix.view(batch_size * max_visit_num, max_proc_num, max_proc_num)
        # d_enc_mask_matrix = d_mask_matrix.view(batch_size * max_visit_num, max_diag_num).unsqueeze(dim=1).unsqueeze(
        #     dim=1).repeat(1, self.nhead, max_diag_num, 1)  # [batch*seq, nhead, input_length, output_length]
        # d_enc_mask_matrix = d_enc_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_diag_num, max_diag_num)
        # p_enc_mask_matrix = p_mask_matrix.view(batch_size * max_visit_num, max_proc_num).unsqueeze(dim=1).unsqueeze(
        #     dim=1).repeat(1, self.nhead, max_proc_num, 1)
        # p_enc_mask_matrix = p_enc_mask_matrix.view(batch_size * max_visit_num * self.nhead, max_proc_num, max_proc_num)
        # 经过transformer编码的visit表示
        input_disease_embedding = self.diagnoses_encoder(input_disease_embedding, src_mask=d_enc_mask_matrix).view(
            batch_size, max_visit_num, max_diag_num, self.emb_dim)
        input_proc_embedding = self.procedure_encoder(input_proc_embedding, src_mask=p_enc_mask_matrix).view(batch_size,
                                                                                                             max_visit_num,
                                                                                                             max_proc_num,
                                                                                                             self.emb_dim)

        # hyper_disease_embedding = h_disease_embed[diseases]
        # hyper_proc_embedding = h_pro_embed[procedures]
        #
        # assert hyper_disease_embedding.shape == input_disease_embedding.shape
        # assert hyper_proc_embedding.shape == input_proc_embedding.shape
        #
        # input_disease_embedding = self.d_global_local_fusion_layer(input_disease_embedding, hyper_disease_embedding)
        # input_proc_embedding = self.p_global_local_fusion_layer(input_proc_embedding, hyper_proc_embedding)
        # d_con_loss_btw_transformer_and_gcn = self.infoNCELoss(input_disease_embedding, hyper_disease_embedding)
        # p_con_loss_btw_transformer_and_gcn = self.infoNCELoss(input_proc_embedding, hyper_proc_embedding)

        # input_disease_embedding = self.d_global_local_fusion_layer(torch.cat([input_disease_embedding, hyper_disease_embedding], dim=-1))
        # input_proc_embedding = self.p_global_local_fusion_layer(torch.cat([input_proc_embedding, hyper_proc_embedding], dim=-1))

        # 1.1 encode visit-level diag and proc representations
        # 这里又展开了，线性层算注意力，这里是算一个visit内的所有item的加权和
        visit_diag_embedding = self.diag_self_attend(
            input_disease_embedding.view(batch_size * max_visit_num, max_diag_num, -1),
            d_mask_matrix.view(batch_size * max_visit_num, -1))
        visit_proc_embedding = self.proc_self_attend(
            input_proc_embedding.view(batch_size * max_visit_num, max_proc_num, -1),
            p_mask_matrix.view(batch_size * max_visit_num, -1))
        # 经过下面的变换后，每个visit使用一个embedding表示，这里感觉可以考虑加入超图表示，因为超图表示也可以理解为visit-level的
        visit_diag_embedding = visit_diag_embedding.view(batch_size, max_visit_num, -1)
        visit_proc_embedding = visit_proc_embedding.view(batch_size, max_visit_num, -1)

        # 1.3 计算 visit-level的attention score
        # [batch_size, max_visit_num, max_visit_num]

        cross_visit_scores = self.calc_cross_visit_scores(visit_diag_embedding, visit_proc_embedding)

        # 3. 构造一个last_seq_medication，表示上一次visit的medication，第一次的由于没有上一次medication，用0填补（用啥填补都行，反正不会用到）
        last_seq_medication = torch.full((batch_size, 1, max_med_num), 0).to(device)
        last_seq_medication = torch.cat([last_seq_medication, medications[:, :-1, :]], dim=1)
        # m_mask_matrix矩阵同样也需要后移
        last_m_mask = torch.full((batch_size, 1, max_med_num), -1e9).to(device)  # 这里用较大负值，避免softmax之后分走了概率
        last_m_mask = torch.cat([last_m_mask, m_mask_matrix[:, :-1, :]], dim=1)
        # 对last_seq_medication进行编码
        # last_seq_medication_emb = self.med_embedding(last_seq_medication)   # 这里要使用graph出来的
        last_seq_medication_emb = med_embed[last_seq_medication]

        last_m_enc_mask = last_m_mask.view(batch_size * max_visit_num, max_med_num).unsqueeze(dim=1).repeat(1, max_med_num, 1)
        last_m_enc_mask = last_m_enc_mask.view(batch_size * max_visit_num, max_med_num, max_med_num)
        encoded_medication = self.medication_encoder(
            last_seq_medication_emb.view(batch_size * max_visit_num, max_med_num, self.emb_dim),
            src_mask=last_m_enc_mask)  # (batch*seq, max_med_num, emb_dim)
        encoded_medication = encoded_medication.view(batch_size, max_visit_num, max_med_num, self.emb_dim)
        # encoded_medication = self.m_1_global_local_fusion_layer(encoded_medication, h_med_embed[last_seq_medication])
        # encoded_medication = self.m_global_local_fusion_layer(torch.cat([encoded_medication, h_med_embed[last_seq_medication]], dim=-1))

        # vocab_size, emb_size
        # med_embed, ddi_embedding = self.gcn()   # med_embed: (n_med, embed_dim), ddi_embeding: (n_med, embed_dim)
        # med_embed = self.m_2_global_local_fusion_layer(med_embed, h_med_embed)
        # m_con_loss_btw_transformer_and_gcn = self.infoNCELoss(med_embed, h_med_embed)
        drug_memory = med_embed - ddi_embedding * self.inter
        drug_memory_padding = torch.zeros((3, self.emb_dim), device=self.device).float()
        drug_memory = torch.cat([drug_memory, drug_memory_padding], dim=0)  # 没看懂这里这个padidng是干嘛

        con_loss = torch.zeros(1)
        # con_loss += d_con_loss_btw_transformer_and_gcn + p_con_loss_btw_transformer_and_gcn + m_con_loss_btw_transformer_and_gcn
        return input_disease_embedding, input_proc_embedding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory, con_loss

    def decode(self, input_medications, input_disease_embedding, input_proc_embedding, last_medication_embedding,
               last_medications, cross_visit_scores,
               d_mask_matrix, p_mask_matrix, m_mask_matrix, last_m_mask, drug_memory):
        """
        input_medications: [batch_size, max_visit_num, max_med_num + 1], 开头包含了 SOS_TOKEN
        """
        batch_size = input_medications.size(0)
        max_visit_num = input_medications.size(1)
        max_med_num = input_medications.size(2)
        max_diag_num = input_disease_embedding.size(2)
        max_proc_num = input_proc_embedding.size(2)

        input_medication_embs = self.med_embedding(input_medications, use_position_encoding=False).view(batch_size * max_visit_num, max_med_num, -1)
        # input_medication_embs = self.dropout_emb(input_medication_embs)
        input_medication_memory = drug_memory[input_medications].view(batch_size * max_visit_num, max_med_num, -1)

        # m_sos_mask = torch.zeros((batch_size, max_visit_num, 1), device=self.device).float() # 这里用较大负值，避免softmax之后分走了概率
        m_self_mask = m_mask_matrix
        # 像是公式12的mask
        last_m_enc_mask = m_self_mask.view(batch_size * max_visit_num, max_med_num).unsqueeze(
            dim=1).repeat(1, max_med_num, 1)
        medication_self_mask = last_m_enc_mask.view(batch_size * max_visit_num, max_med_num, max_med_num)
        m2d_mask_matrix = d_mask_matrix.view(batch_size * max_visit_num, max_diag_num).unsqueeze(dim=1).repeat(1, max_med_num, 1)
        m2d_mask_matrix = m2d_mask_matrix.view(batch_size * max_visit_num, max_med_num, max_diag_num)
        m2p_mask_matrix = p_mask_matrix.view(batch_size * max_visit_num, max_proc_num).unsqueeze(dim=1).repeat(1, max_med_num, 1)
        m2p_mask_matrix = m2p_mask_matrix.view(batch_size * max_visit_num, max_med_num, max_proc_num)

        dec_hidden = self.decoder(input_medication_embedding=input_medication_embs,
                                  input_medication_memory=input_medication_memory,
                                  input_disease_embdding=input_disease_embedding.view(batch_size * max_visit_num,
                                                                                      max_diag_num, -1),
                                  input_proc_embedding=input_proc_embedding.view(batch_size * max_visit_num,
                                                                                 max_proc_num, -1),
                                  input_medication_self_mask=medication_self_mask,
                                  d_mask=m2d_mask_matrix,
                                  p_mask=m2p_mask_matrix)

        score_g = self.Wo(dec_hidden)  # (batch * max_visit_num, max_med_num, voc_size[2]+2)
        score_g = score_g.view(batch_size, max_visit_num, max_med_num, -1)
        prob_g = F.softmax(score_g, dim=-1)
        # TODO：改这个
        score_c = self.copy_med(dec_hidden.view(batch_size, max_visit_num, max_med_num, -1), last_medication_embedding,
                                last_m_mask, cross_visit_scores)
        # (batch_size, max_visit_num * input_med_num, max_visit_num * max_med_num)

        ###### case study
        # 这里前提是batch_size等于1
        # 几个取值的说明：
        #   1.取最新生成的药物对于历史药物的attention值，所以第三维度为-1
        #   2.取第最后一个visit的copy值，所以第二维度为-1
        #   3.取最后一个visit对倒数第二个visit的药物的attention值，所以第四维度取最后max_med_num个
        # score_c_buf = score_c.view(batch_size, max_visit_num, max_med_num, -1)
        # score_c_buf = score_c_buf[0, -1, -1, :] # visit_num * (visit_num * max_med_num)
        # max_med_num_in_last = len(score_c_buf) // max_visit_num
        # print(score_c_buf[-max_med_num_in_last:])

        prob_c_to_g = torch.zeros_like(prob_g).to(self.device).view(batch_size, max_visit_num * max_med_num,
                                                                    -1)  # (batch, max_visit_num * input_med_num, voc_size[2]+2)

        # 用scatter操作代替嵌套循环
        # 根据last_seq_medication中的indice，将score_c中的值加到score_c_to_g中去
        copy_source = last_medications.view(batch_size, 1, -1).repeat(1, max_visit_num * max_med_num, 1)
        # todo：这里有加法，但是已经变成概率了，先放着
        prob_c_to_g.scatter_add_(2, copy_source, score_c)
        prob_c_to_g = prob_c_to_g.view(batch_size, max_visit_num, max_med_num, -1)

        generate_prob = F.sigmoid(self.W_z(dec_hidden)).view(batch_size, max_visit_num, max_med_num, 1)
        prob = prob_g * generate_prob + prob_c_to_g * (1. - generate_prob)
        prob[:, 0, :, :] = prob_g[:, 0, :, :]  # 第一个seq由于没有last_medication信息，仅取prob_g的概率

        return torch.log(prob)

    # def forward(self, input, last_input=None, max_len=20):
    def forward(self, diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix, seq_length,
                dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc, dec_proc_mask,
                stay_proc_mask, max_len=20):
        device = self.device
        # diseases: batch_size, max_seq, d_max_num
        # 每一条数据都是一个病人的所有visit中的disease记录，这里首先把每条数据的长度补成一致，然后将每一个visit中的disease数量补成一致
        # batch维度以及seq维度上并行计算（现在不考虑时间序列信息），每一个medication序列仍然按顺序预测
        batch_size, max_seq_length, max_med_num = medications.size()
        max_diag_num = diseases.size()[2]
        max_proc_num = procedures.size()[2]
        input_disease_embdding, input_proc_embedding, encoded_medication, cross_visit_scores, last_seq_medication, last_m_mask, drug_memory, con_loss = self.encode(
            diseases, procedures, medications, d_mask_matrix, p_mask_matrix, m_mask_matrix,
            seq_length, dec_disease, stay_disease, dec_disease_mask, stay_disease_mask, dec_proc, stay_proc,
            dec_proc_mask, stay_proc_mask, max_len=20)

        # 4. 构造给decoder的medications，用于decoding过程中的teacher forcing，注意维度上增加了一维，因为会多生成一个END_TOKEN
        input_medication = torch.full((batch_size, max_seq_length, 1), self.SOS_TOKEN).to(
            device)  # [batch_size, seq, 1]
        input_medication = torch.cat([input_medication, medications], dim=2)  # [batch_size, seq, max_med_num + 1]

        m_sos_mask = torch.zeros((batch_size, max_seq_length, 1),
                                 device=self.device).float()  # 这里用较大负值，避免softmax之后分走了概率
        m_mask_matrix = torch.cat([m_sos_mask, m_mask_matrix], dim=-1)
        #
        output_logits = self.decode(input_medication, input_disease_embdding, input_proc_embedding, encoded_medication,
                                    last_seq_medication, cross_visit_scores,
                                    d_mask_matrix, p_mask_matrix, m_mask_matrix, last_m_mask, drug_memory)
        # 5. 加入ddi loss
        # output_logits_part = torch.exp(output_logits[:, :, :, :-2] + m_mask_matrix.unsqueeze(-1))    # 去掉SOS与EOS
        # output_logits_part = torch.mean(output_logits_part, dim=2)
        # neg_pred_prob1 = output_logits_part.unsqueeze(-1)
        # neg_pred_prob2 = output_logits_part.unsqueeze(-2)
        # neg_pred_prob = neg_pred_prob1 * neg_pred_prob2 # bach * seq * max_med_num * all_med_num * all_med_num
        # batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        # return output_logits, batch_neg
        return output_logits, con_loss

    def calc_cross_visit_scores(self, visit_diag_embedding, visit_proc_embedding):
        """
        visit_diag_embedding: (batch * visit_num * emb)
        visit_proc_embedding: (batch * visit_num * emb)
        """
        max_visit_num = visit_diag_embedding.size(1)
        batch_size = visit_diag_embedding.size(0)

        # mask表示每个visit只能看到自己之前的visit
        mask = (torch.triu(torch.ones((max_visit_num, max_visit_num), device=self.device)) == 1).transpose(0,
                                                                                                           1)  # 下三角矩阵
        mask = mask.float().masked_fill(mask == 0, -1e9).masked_fill(mask == 1, float(0.0))
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)  # batch * max_visit_num * max_visit_num

        # 每个visit后移一位
        padding = torch.zeros((batch_size, 1, self.emb_dim), device=self.device).float()
        diag_keys = torch.cat([padding, visit_diag_embedding[:, :-1, :]],
                              dim=1)  # batch * max_visit_num * emb，这里-1位置取不到
        proc_keys = torch.cat([padding, visit_proc_embedding[:, :-1, :]], dim=1)

        # 得到每个visit跟自己前面所有visit的score，先算任意两个visit的相似度，然后做mask
        # TODO：这里是注意力操作，可以看看多头注意力

        diag_scores = (2 + 2 * self.manifold.cinner(visit_diag_embedding, diag_keys)) \
                      / self.scale_d + self.bias_d
        proc_scores = (2 + 2 * self.manifold.cinner(visit_proc_embedding, proc_keys)) \
                      / self.scale_p + self.bias_p
        # diag_scores = torch.matmul(visit_diag_embedding, diag_keys.transpose(-2, -1)) \
        #               / math.sqrt(visit_diag_embedding.size(-1))  # 公式16
        # proc_scores = torch.matmul(visit_proc_embedding, proc_keys.transpose(-2, -1)) \
        #               / math.sqrt(visit_proc_embedding.size(-1))
        # 1st visit's scores is not zero!
        scores = F.softmax(diag_scores + proc_scores + mask, dim=-1)

        ###### case study
        # 将第0个val置0，然后重新归一化
        # scores_buf = scores
        # scores_buf[:, :, 0] = 0.
        # scores_buf = scores_buf / torch.sum(scores_buf, dim=2, keepdim=True)

        # print(scores_buf)
        return scores

    def copy_med(self, decode_input_hiddens, last_medications, last_m_mask, cross_visit_scores):
        """
        decode_input_hiddens: [batch_size, max_visit_num, input_med_num, emb_size]
        last_medications: [batch_size, max_visit_num, max_med_num, emb_size]
        last_m_mask: [batch_size, max_visit_num, max_med_num]
        cross_visit_scores: [batch_size, max_visit_num, max_visit_num]
        """
        max_visit_num = decode_input_hiddens.size(1)
        input_med_num = decode_input_hiddens.size(2)
        max_med_num = last_medications.size(2)
        # 这里线性已经改了LorentzLinear了
        copy_query = self.Wc(decode_input_hiddens).view(-1, max_visit_num * input_med_num, self.emb_dim)
        # 这里是注意力，需要改成基于距离的
        attn_scores = (2 + 2 * self.manifold.cinner(copy_query, last_medications.view(-1, max_visit_num * max_med_num, self.emb_dim))) \
                      / self.scale_copy + self.bias_copy
        # attn_scores = torch.matmul(copy_query,
        #                            last_medications.view(-1, max_visit_num * max_med_num, self.emb_dim).transpose(-2,
        #                                                                                                           -1)) / math.sqrt(
        #     self.emb_dim)  # 公式19
        med_mask = last_m_mask.view(-1, 1, max_visit_num * max_med_num).repeat(1, max_visit_num * input_med_num, 1)
        # [batch_size, max_vist_num * input_med_num, max_visit_num * max_med_num]
        attn_scores = F.softmax(attn_scores + med_mask, dim=-1)

        # (batch_size, max_visit_num * input_med_num, max_visit_num)
        visit_scores = cross_visit_scores.repeat(1, 1, input_med_num).view(-1, max_visit_num * input_med_num,
                                                                           max_visit_num)

        # (batch_size, max_visit_num * input_med_num, max_visit_num * max_med_num)
        visit_scores = visit_scores.unsqueeze(-1).repeat(1, 1, 1, max_med_num).view(-1, max_visit_num * input_med_num,
                                                                                    max_visit_num * max_med_num)
        # todo:这里存疑
        # scores = 2 + 2 * self.manifold.cinner(attn_scores, visit_scores).clamp(min=1e-9)
        scores = torch.mul(attn_scores, visit_scores).clamp(min=1e-9)  # 公式20
        row_scores = scores.sum(dim=-1, keepdim=True)
        scores = scores / row_scores  # (batch_size, max_visit_num * input_med_num, max_visit_num * max_med_num)

        return scores
