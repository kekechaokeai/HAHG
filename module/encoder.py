# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.preprocess import sparse_mx_to_torch_sparse_tensor, normalize_adj_from_tensor, add_self_loop_and_normalize, remove_self_loop

EPS = 1e-15

class GCN(nn.Module):  # 定义图卷积神经网络类，继承自PyTorch的nn.Module
    def __init__(self, input_dim, output_dim, dropout=0.2):  # 构造函数，初始化输入输出维度以及dropout率
        super(GCN, self).__init__()  # 调用父类构造函数
        self.weight_1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim))  # 定义一个可学习的权重矩阵，维度为 (输入维度, 输出维度)
        self.weight_1 = self.reset_parameters(self.weight_1)  # 初始化权重
        if dropout:  # 如果dropout率不为0
            self.gc_drop = nn.Dropout(dropout)  # 定义一个dropout层
        else:
            self.gc_drop = lambda x: x  # 如果dropout率为0，dropout层直接返回输入
        self.bc = nn.BatchNorm1d(output_dim)  # 定义一个批标准化层，用于输出维度进行标准化

    def reset_parameters(self, weight):  # 权重初始化函数
        stdv = 1. / math.sqrt(weight.size(1))  # 根据输入维度计算标准差
        weight.data.uniform_(-stdv, stdv)  # 使用均匀分布初始化权重
        return weight  # 返回初始化后的权重

    def forward(self, feat, adj, nlayer, sampler=False):  # 定义前向传播函数
        if not sampler:  # 如果不使用采样
            z = self.gc_drop(torch.mm(feat, self.weight_1))  # 计算特征和权重矩阵的乘积，然后应用dropout
            z = torch.mm(adj, z)  # 将邻接矩阵与结果相乘，进行图卷积
            for i in range(1, nlayer):  # 对于每一层
                z = torch.mm(adj, z)  # 再次与邻接矩阵相乘，进行多层图卷积

            outputs = F.normalize(z, dim=1)  # 对结果进行行规范化
            return outputs  # 返回输出
        else:  # 如果使用采样
            z_ = self.gc_drop(torch.mm(feat, self.weight_1))  # 计算特征和权重矩阵的乘积，并应用dropout
            z = torch.mm(adj, z_)  # 将邻接矩阵与结果相乘
            for i in range(1, nlayer):  # 对于每一层
                z = torch.mm(adj, z)  # 再次与邻接矩阵相乘

            return z  # 返回结果


class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2, heads=4):
        """
        输入：
        input_dim: 输入特征维度
        output_dim: 输出特征维度
        dropout: dropout概率
        heads: attention头数
        """
        super(GAT, self).__init__()
        self.heads = heads
        
        # 初始化权重矩阵，注意力机制会为每个head生成不同的权重矩阵
        self.weight_1 = nn.Parameter(torch.FloatTensor(input_dim, output_dim * heads))  # 每个head都有一组权重
        self.weight_1 = self.reset_parameters(self.weight_1)
        
        # 为每个head定义一个注意力机制相关的权重
        self.attention_weights = nn.Parameter(torch.FloatTensor(2 * output_dim, heads))  # 用于计算注意力系数
        
        if dropout:
            self.gc_drop = nn.Dropout(dropout)
        else:
            self.gc_drop = lambda x: x
        
        self.bc = nn.BatchNorm1d(output_dim * heads)  # BatchNorm操作

    def reset_parameters(self, weight):
        """初始化权重"""
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        return weight
    
    def forward(self, feat, adj, nlayer, sampler=False):
        """
        前向传播：feat 为节点特征，adj 为邻接矩阵，nlayer 为图卷积层数，sampler 为是否使用采样
        """
        if not sampler:
            # 线性变换计算每个节点的特征
            z = self.gc_drop(torch.mm(feat, self.weight_1))
            z = z.view(-1, self.heads, z.size(1) // self.heads)  # 对每个head进行拆分
            z = torch.sum(z, dim=1)  # 将每个head的输出进行合并
            
            # 使用邻接矩阵计算加权特征
            for i in range(1, nlayer):
                z = torch.mm(adj, z)

            outputs = F.normalize(z, dim=1)
            return outputs
        else:
            # 如果使用采样
            z_ = self.gc_drop(torch.mm(feat, self.weight_1))
            z = torch.mm(adj, z_)
            
            for i in range(1, nlayer):
                z = torch.mm(adj, z)

            return z


class Attention_NEW(nn.Module):
    def __init__(self, hidden_dim, attn_drop=0.1):
        super(Attention_NEW, self).__init__()

        self.act = nn.ELU()  # 使用 ELU 激活函数

        # 定义低频注意力参数向量，形状为 (1, hidden_dim)，可训练
        self.att_l = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att_l.data, gain=1.414)

        # 定义高频注意力参数向量，形状为 (1, hidden_dim)，可训练
        self.att_h = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att_h.data, gain=1.414)

        self.softmax = nn.Softmax(dim=0)  # 在所有子图维度上做 softmax 归一化
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds_l, embeds_h):
        beta_l = []  # 存储所有低频子图嵌入的注意力得分
        beta_h = []  # 存储所有高频子图嵌入的注意力得分

        attn_curr_l = self.attn_drop(self.att_l)  # 低频注意力向量经过 Dropout
        attn_curr_h = self.attn_drop(self.att_h)  # 高频注意力向量经过 Dropout

        # 分别计算每个低频子图嵌入的注意力得分（形状为 1 x N，N为节点数）
        for embed in embeds_l:
            beta_l.append(attn_curr_l.matmul(embed.t()))
        # 分别计算每个高频子图嵌入的注意力得分
        for embed in embeds_h:
            beta_h.append(attn_curr_h.matmul(embed.t()))

        # 合并低频与高频注意力得分以及对应的嵌入向量
        beta = beta_l + beta_h  # 得到长度为 num_subgraph * 2 的列表
        embeds = embeds_l + embeds_h

        # 将所有注意力得分拼接成 (num_subgraph*2, N) 的矩阵
        beta = torch.cat(beta, dim=0)
        beta = self.act(beta)  # 激活处理
        beta = self.softmax(beta)  # 对每个子图维度做归一化

        # 融合编码：对每个子图嵌入加权求和，注意保证最终输出 shape 与原版本一致
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i].reshape(-1, 1)
        return z_mp




class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop=0.1):
        super(Attention, self).__init__()

        self.act = nn.ELU()  # 使用 ELU 作为激活函数

        # 定义低频注意力参数向量，形状为 (1, hidden_dim)，可训练
        self.att_l = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att_l.data, gain=1.414)  # 使用 Xavier 初始化低频参数

        # 定义高频注意力参数向量，形状为 (1, hidden_dim)，可训练
        self.att_h = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att_h.data, gain=1.414)  # 使用 Xavier 初始化高频参数

        self.softmax = nn.Softmax(dim=0)  # 在所有子图维度上做 softmax 归一化
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)  # 如果设置了 drop 概率，就应用 Dropout
        else:
            self.attn_drop = lambda x: x  # 否则，Dropout 是恒等函数（不进行丢弃）

    def forward(self, embeds_l, embeds_h):
        beta_l = []  # 存储所有低频嵌入的注意力权重
        beta_h = []  # 存储所有高频嵌入的注意力权重

        attn_curr_l = self.attn_drop(self.att_l)  # 对低频注意力向量应用 Dropout
        attn_curr_h = self.attn_drop(self.att_h)  # 对高频注意力向量应用 Dropout

        # 计算每个低频子图嵌入的注意力权重
        for embed in embeds_l:
            beta_l.append(attn_curr_l.matmul(embed.t()))  # 1 x N，N是节点数

        # 计算每个高频子图嵌入的注意力权重
        for embed in embeds_h:
            beta_h.append(attn_curr_h.matmul(embed.t()))  # 1 x N

        beta = beta_l + beta_h  # 合并低频和高频注意力得分
        embeds = embeds_l + embeds_h  # 合并低频和高频嵌入向量
        beta = torch.cat(beta, dim=0)  # 拼接成 shape 为 (num_subgraph, N) 的矩阵
        beta = self.act(beta)  # 激活操作，增强非线性表达能力
        beta = self.softmax(beta)  # 对每个子图的注意力得分做归一化处理

        z_mp = 0  # 初始化融合后的主路径嵌入向量
        for i in range(len(embeds)):
            z_mp += embeds[i] * beta[i].reshape(-1, 1)  # 加权求和，每个节点按注意力融合
        return z_mp  # 返回融合后的图嵌入
