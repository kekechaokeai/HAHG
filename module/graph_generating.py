
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.preprocess import remove_self_loop, normalize_adj_from_tensor


EPS = 1e-15  # 防止除零的小常数
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseEfficientNodeLevelAttention(nn.Module):
    """
    改进版稀疏多头节点级注意力，兼顾效率与性能：
    - 稀疏邻接支持；
    - 简化注意力输入结构；
    - 残差连接 + LayerNorm + FFN；
    """

    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.6, alpha=0.2, use_layernorm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.use_layernorm = use_layernorm

        self.W = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        # 使用简化的 attention：只拼接 Wh_i 和 Wh_j
        self.attn_fc = nn.Parameter(torch.empty(size=(num_heads, 2 * out_channels, 1)))
        
        if use_layernorm:
            self.ln = nn.LayerNorm(out_channels * num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels * num_heads, out_channels * num_heads),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_fc)

    def forward(self, adj_sparse, h):
        """
        adj_sparse: torch.sparse_coo_tensor (N, N)
        h: (N, in_channels)
        return: (N, out_dim)
        """
        device = h.device
        N = h.size(0)

        Wh = self.W(h)  # (N, num_heads * out_channels)
        Wh = Wh.view(N, self.num_heads, self.out_channels)  # (N, heads, out)

        row, col = adj_sparse.coalesce().indices()  # (2, E)
        Wh_i = Wh[row]  # (E, heads, out)
        Wh_j = Wh[col]  # (E, heads, out)

        # 注意力计算输入为拼接 (Wh_i, Wh_j)
        edge_input = torch.cat([Wh_i, Wh_j], dim=-1)  # (E, heads, 2*out)

        # 逐 head attention score
        e = torch.einsum('ehf,hfd->ehd', edge_input, self.attn_fc)  # (E, heads, 1)
        e = self.leakyrelu(e).squeeze(-1)  # (E, heads)

        # -------- Segment-wise Softmax --------
        e = torch.exp(e - torch.max(e, dim=0)[0])  # 防止溢出
        e_sum = torch.zeros(N, self.num_heads, device=device)  # (N, heads)
        e_sum.index_add_(0, row, e)  # 聚合每个节点的邻居注意力
        alpha = e / (e_sum[row] + 1e-15)  # (E, heads)
        alpha = self.dropout(alpha)

        # -------- 特征加权聚合 --------
        h_new = torch.zeros(N, self.num_heads, self.out_channels, device=device)  # (N, heads, out)
        h_new.index_add_(0, row, Wh_j * alpha.unsqueeze(-1))  # (N, heads, out)

        h_new = h_new.reshape(N, self.num_heads * self.out_channels)  # (N, heads*out)

        # 残差连接（前提是维度相同）
        if h_new.shape == Wh.reshape(N, -1).shape:
            h_new = h_new + Wh.reshape(N, -1)

        # 可选 LayerNorm
        if self.use_layernorm:
            h_new = self.ln(h_new)

        # FeedForward 强化
        h_out = self.ffn(h_new)

        return h_out


# class HybridSemanticAttention(nn.Module):
#     def __init__(self, in_channels, hidden_channels, num_meta_paths, out_channels=None, beta=0.5, fusion='add'):
#         """
#         in_channels: 每条 meta-path 的特征维度
#         hidden_channels: attention hidden dim
#         num_meta_paths: meta-path 数量
#         out_channels: 融合后输出的维度（若为 None，则不变）
#         beta: 全局 attention 与个性 attention 的权重融合比例
#         fusion: 'add' or 'gate'
#         """
#         super().__init__()
#         self.beta = beta
#         self.fusion = fusion
#         self.num_meta_paths = num_meta_paths
#         self.out_channels = out_channels

#         # Global (meta-path level) attention
#         self.global_fc = nn.Linear(in_channels, hidden_channels, bias=False)
#         self.global_vector = nn.Parameter(torch.Tensor(num_meta_paths, hidden_channels))
#         nn.init.xavier_uniform_(self.global_fc.weight)
#         nn.init.xavier_uniform_(self.global_vector.unsqueeze(0))

#         # Personalized attention
#         self.personal_mlp = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#             nn.Tanh(),
#             nn.Linear(hidden_channels, num_meta_paths)
#         )

#         # Gated fusion
#         if fusion == 'gate':
#             self.gate_fc = nn.Sequential(
#                 nn.Linear(in_channels, hidden_channels),
#                 nn.Tanh(),
#                 nn.Linear(hidden_channels, 1)
#             )

#         # Output projection if needed
#         if out_channels is not None:
#             self.output_fc = nn.Linear(in_channels, out_channels)

#     def forward(self, meta_feats):
#         """
#         meta_feats: list of (N, F), len=M
#         return:
#             - agg: (N, F) or (N, out_channels)
#             - weights: (N, M, 1)
#         """
#         N, feat_dim = meta_feats[0].shape
#         M = self.num_meta_paths
#         meta_stack = torch.stack(meta_feats, dim=1)  # (N, M, F)

#         # --- Global Attention ---
#         h_global = self.global_fc(meta_stack)  # (N, M, hidden)
#         scores_global = (h_global * self.global_vector.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (N, M, 1)
#         weights_global = F.softmax(scores_global, dim=1)  # (N, M, 1)

#         # --- Personalized Attention ---
#         pooled = meta_stack.mean(dim=1)  # (N, F)
#         scores_personal = self.personal_mlp(pooled)  # (N, M)
#         weights_personal = F.softmax(scores_personal, dim=1).unsqueeze(-1)  # (N, M, 1)

#         # --- Combine Attentions ---
#         if self.fusion == 'add':
#             weights = self.beta * weights_global + (1 - self.beta) * weights_personal
#         elif self.fusion == 'gate':
#             gate = torch.sigmoid(self.gate_fc(pooled))  # (N, 1)
#             weights = gate.unsqueeze(1) * weights_global + (1 - gate.unsqueeze(1)) * weights_personal
#         else:
#             raise ValueError("Unsupported fusion mode: choose from ['add', 'gate']")

#         # --- Weighted Aggregation ---
#         agg = (weights * meta_stack).sum(dim=1)  # (N, F)

#         # --- Optional Output Projection ---
#         if self.out_channels is not None:
#             agg = self.output_fc(agg)  # (N, out_channels)

#         return agg, weights



class SemanticAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super(SemanticAttention, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        self.q = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))  # Semantic-level attention vector
        nn.init.xavier_uniform_(self.q.data, gain=1.414)

    def forward(self, Z_list):  # Z_list: list of [N, d] meta-path embeddings
        semantic_embeddings = []
        att_weights = []

        for Z in Z_list:
            h = self.proj(Z)  # [N, hidden_dim]
            score = torch.matmul(h, self.q).mean(dim=0)  # Scalar attention score
            att_weights.append(score)

        att_weights = torch.stack(att_weights, dim=0)  # [P]
        att_weights = F.softmax(att_weights, dim=0)    # [P]

        for i, Z in enumerate(Z_list):
            semantic_embeddings.append(att_weights[i] * Z)

        Z_final = torch.stack(semantic_embeddings, dim=0).sum(dim=0)  # [N, d]
        return Z_final, att_weights  # 返回加权后的融合表示和注意力权重



def get_top_k(sim_l, sim_h, k1, k2):
    _, k_indices_pos = torch.topk(sim_l, k=k1, dim=1)
    _, k_indices_neg = torch.topk(sim_h, k=k2, dim=1)

    source = torch.arange(sim_l.size(0)).view(-1, 1).to(sim_l.device)
    k_source_l = source.repeat(1, k1).flatten()
    k_source_h = source.repeat(1, k2).flatten()

    k_indices_pos = k_indices_pos.flatten()
    k_indices_pos = torch.stack((k_source_l, k_indices_pos), dim=0)

    k_indices_neg = k_indices_neg.flatten()
    k_indices_neg = torch.stack((k_source_h, k_indices_neg), dim=0)

    kg_pos = torch.sparse.FloatTensor(k_indices_pos, torch.ones(k_indices_pos.size(1)).to(sim_l.device),
                                      torch.Size([sim_l.size(0), sim_l.size(0)]))
    kg_neg = torch.sparse.FloatTensor(k_indices_neg, torch.ones(k_indices_neg.size(1)).to(sim_l.device),
                                      torch.Size([sim_l.size(0), sim_l.size(0)]))
    return kg_pos, kg_neg


def graph_construction(x, adjs, k1, k2, k_pos):
    """
    通过节点特征 x 和结构信息（邻接矩阵）计算相似度，
    并利用 top-k 策略构造正负样本图。
    """
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    zero_indices = torch.nonzero(x_norm.flatten() == 0)
    x_norm[zero_indices] += EPS  # 防止除零
    dot_numerator = torch.mm(x, x.t())
    dot_denominator = torch.mm(x_norm, x_norm.t())
    fea_sim = dot_numerator / dot_denominator

    adjs_rw = []
    for adj in adjs:
        adj_ = adj.to_dense() + torch.eye(x.size(0)).to(x.device)  # 添加自环
        RW = adj_ / adj_.sum(dim=1, keepdim=True)
        adjs_rw.append(RW)

    adjs_rw = torch.stack(adjs_rw, dim=0).mean(dim=0)
    adj_norm = torch.norm(adjs_rw, dim=1, keepdim=True)
    zero_indices = torch.nonzero(adj_norm.flatten() == 0)
    adj_norm[zero_indices] += EPS
    adj_sim = torch.mm(adjs_rw, adjs_rw.t()) / torch.mm(adj_norm, adj_norm.t())

    sim_l = adj_sim * fea_sim
    sim_l = sim_l - torch.diag_embed(torch.diag(sim_l))  # 去除自连接
    sim_h = (1 - adj_sim) * (1 - fea_sim)
    kg_pos, kg_neg = get_top_k(sim_l, sim_h, k1, k2)
    # print("正样本是什么",k_pos)
    if k_pos <= 0:
        pos = torch.eye(x.size(0)).to_sparse().to(x.device)
    else:
        pos, _ = get_top_k(sim_l, sim_h, k_pos, k_pos)
        pos = (pos.to_dense() + torch.eye(x.size(0)).to(x.device)).to_sparse()
    return [kg_pos], [kg_neg], [pos]


