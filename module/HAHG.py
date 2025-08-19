# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .preprocess import *
from .encoder import *
from .loss_fun import *
from .graph_generating import *  
from module.preprocess import remove_self_loop, normalize_adj_from_tensor


class HAHG(nn.Module):
    def __init__(self, feats_dim, sub_num, hidden_dim, embed_dim, tau, dropout, act, drop_feat, nnodes, dataset, alpha, nlayer,graph_k,k_pos,beta,fusion,args):
        super(HAHG, self).__init__()

        self.alpha = alpha
        self.feats_dim = feats_dim
        self.embed_dim = embed_dim
        self.sub_num = sub_num
        self.tau = tau
        self.dataset = dataset
        self.nnodes = nnodes
        self.act = act
        self.nlayer = nlayer
        self.nlayer_c = nlayer
        self.drop = nn.Dropout(drop_feat)
        self.graph_k = graph_k
        self.k_pos = k_pos
        self.beta = beta
        self.fusion = fusion

        self.fc = nn.Linear(feats_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))

        self.Encoder = GCN(hidden_dim, embed_dim, dropout)
        self.att = Attention_NEW(embed_dim, attn_drop=0.1)

        self.decoder = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feats_dim)
        )
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

        self.contrast_l = Contrast(embed_dim, embed_dim, act, self.tau)
        self.contrast_h = Contrast(embed_dim, embed_dim, act, self.tau)
        #self.cluster_loss = ClusterLoss(n_clusters=args.n_clusters, alpha=1.0)

        # 加入可训练的 Node-Level 与 Semantic-Level Attention
        # self.node_attn = SparseNodeLevelAttention(in_channels=feats_dim, out_channels=feats_dim, dropout=dropout, alpha=2)
        self.node_attn = SparseEfficientNodeLevelAttention(
            in_channels=feats_dim,
            out_channels=args.node_out_channels,         # 每个 head 的输出维度
            num_heads=4,             # 总共 4 个头，输出维度为 128
            dropout=args.node_drop,
            alpha=args.node_alpha,
            use_layernorm=True
        )

        self.meta_out_dim = args.node_out_channels * 4 
        # #self.meta_out_dim = feats_dim
        # self.semantic_attn = HybridSemanticAttention(
        #     in_channels=self.meta_out_dim,
        #     hidden_channels=64,
        #     out_channels=feats_dim,
        #     num_meta_paths=sub_num,
        #     beta=beta,
        #     fusion=fusion
        # )
        self.semantic_attn = SemanticAttention(
            in_dim=self.meta_out_dim,
            hidden_dim=args.meta_hidden_channels
        )


    def forward(self, adjs, feat):
        #---------------融合后------------------------#
        adjs = [adj.coalesce() for adj in adjs]
        adjs = remove_self_loop(adjs)

        device = feat.device
        N = feat.size(0)
        adj_I = torch.eye(N).to(device)

        # 对每个 meta-path 构造归一化后的邻接矩阵（添加自环后归一化）
        adjs_o = [normalize_adj_from_tensor(adj_I + adj.to_dense(), mode='sym').to_sparse() for adj in adjs]

        # --------------- 节点级 Attention ----------------
        # 对每个 meta-path，利用节点级 Attention 聚合邻居特征
        # print("对每个 meta-path，利用节点级 Attention 聚合邻居特征")
        meta_path_features = []
        for adj in adjs_o:
            adj = adj.to_sparse()
            meta_feat = self.node_attn(adj, feat)
            # meta_feat = feat
            meta_path_features.append(meta_feat)

        aggregated_feat, att= self.semantic_attn(meta_path_features)

        updated_features = aggregated_feat
        # print("out_dim",aggregated_feat.shape)
        # aggregated_feat = meta_path_features
        # updated_features = torch.stack(aggregated_feat, dim=0).sum(dim=0)

        # 根据更新后的节点特征构造图，并生成正负样本

        adjs_l, adjs_h, pos = graph_construction(updated_features, adjs, self.graph_k, self.graph_k, self.k_pos)

        # 归一化各图：正样本、负样本以及原始图
        adjs_l = [normalize_adj_from_tensor(adj_I + adj.to_dense(), mode='sym') for adj in adjs_l]
        adjs_h = [normalize_adj_from_tensor(adj.to_dense(), mode='sym').to_sparse() for adj in adjs_h]
        adjs_h = [adj_I - adj.to_dense() for adj in adjs_h]
        adjs_o = [normalize_adj_from_tensor(adj_I + adj.to_dense(), mode='sym').to_sparse() for adj in adjs]

        # 调试输出中间结果
        # print("Aggregated node features from Semantic-level Attention:", aggregated_feat)
        # # print("Attention weights for meta-paths:", attn)
        # print("Updated node features from GAT:", updated_features)
        # print("Positive sample adjacencies (adjs_l):", adjs_l)
        # print("Negative sample adjacencies (adjs_h):", adjs_h)

        #return adjs_l, adjs_h, adjs_o, pos, updated_features

        #-------------------------------



        pos_l = pos[0]

        adj_I = torch.eye(self.nnodes, self.nnodes).to(feat.device)

        # 主路径：通过全连接映射和激活函数获得初始节点表示
        h_mp = self.act(self.fc(feat))
        #h_mp = self.act(self.fc(updated_features))
        # 带 Dropout 的特征表示
        h_mask = self.act(self.fc(self.drop(feat)))
        #h_mask = self.act(self.fc(self.drop(updated_features)))
        h_l_ = []  # 存储子图低频编码结果
        h_h_ = []  # 存储子图高频编码结果

        # 使用全图邻接矩阵获得低频和高频全局编码，假定 adjs_l[0] 与 adjs_h[0] 为全图信息
        z_l = self.Encoder(h_mp, adjs_l[0], self.nlayer_c)
        z_h = self.Encoder(h_mp, adjs_h[0], self.nlayer_c)

        # 对每个子图分别计算低频和高频编码
        for i in range(self.sub_num):
            filter_l_o = adjs_o[i].to_dense()  # 低频邻接矩阵（稠密格式）
            filter_h_o = adj_I - adjs_o[i].to_dense()  # 高频邻接矩阵 = 单位矩阵 - 低频矩阵

            h_l_.append(self.Encoder(h_mask, filter_l_o, self.nlayer))
            h_h_.append(self.Encoder(h_mask, filter_h_o, self.nlayer))

        # 使用 Attention 模块对子图低频和高频编码进行融合
        h = self.att(h_l_, h_h_)

        # 计算重构损失
        loss_rec = 0
        for i in range(self.sub_num):
            # 拼接当前子图的低频和高频编码，并经过解码器重构输入特征
            fea_rec = self.decoder(torch.cat((h_l_[i], h_h_[i]), dim=-1))
            loss_rec += sce_loss(fea_rec, feat, self.alpha)
        loss_rec = loss_rec / self.sub_num

        # 计算对比学习损失（低频和高频各一部分）
        loss_l = self.contrast_l(z_l, h, pos_l)
        loss_h = self.contrast_h(z_h, h, pos_l)
        loss = loss_l + loss_h + loss_rec


        return loss

    def get_embeds(self, feat, adjs_o):
        """
        用于获取节点嵌入，不参与反向传播
        feat   : 输入节点特征
        adjs_o : 原始邻接矩阵列表（子图）
        """
        adj_I = torch.eye(self.nnodes, self.nnodes).to(feat.device)
        h_mp_l = self.act(self.fc(feat))
        h_l_ = []
        h_h_ = []
        for i in range(self.sub_num):
            filter_l_o = adjs_o[i].to_dense()
            filter_h_o = adj_I - adjs_o[i].to_dense()
            h_l_.append(self.Encoder(h_mp_l, filter_l_o, self.nlayer))
            h_h_.append(self.Encoder(h_mp_l, filter_h_o, self.nlayer))
        z = self.att(h_l_, h_h_)
        return z.detach()
