import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-15

def diversity_loss(meta_embeddings):
    """
    meta_embeddings: List of tensor, each is [N, d] from different meta-path
    Encourages embeddings from different meta-paths to be diverse
    """
    loss = 0
    num_paths = len(meta_embeddings)
    for i in range(num_paths):
        for j in range(i + 1, num_paths):
            emb_i = F.normalize(meta_embeddings[i], dim=-1)
            emb_j = F.normalize(meta_embeddings[j], dim=-1)
            sim = torch.mean(torch.sum(emb_i * emb_j, dim=-1))  # cosine similarity
            loss += sim
    return loss / (num_paths * (num_paths - 1) / 2)  # 平均对之间相似度


class CrossViewContrast(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau
        self.sim = nn.CosineSimilarity(dim=-1)

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim_matrix = torch.exp(self.sim(z1.unsqueeze(1), z2.unsqueeze(0)) / self.tau)
        pos_sim = torch.exp(self.sim(z1, z2) / self.tau)
        loss = -torch.log(pos_sim / (sim_matrix.sum(dim=1) + 1e-8)).mean()
        return loss
    
    
class Contrast(nn.Module):
    def __init__(self, hidden_dim, project_dim, act, tau):
        """
        Args:
            hidden_dim: 输入隐藏特征维度
            project_dim: 投影后的维度
            act: 激活函数 (如 nn.ELU())
            tau: 温度参数，用于缩放余弦相似度
        """
        super(Contrast, self).__init__()
        self.tau = tau
        # 投影网络 1：将隐藏特征映射到投影空间
        self.proj_1 = nn.Sequential(
            nn.Linear(hidden_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            act,
            nn.Linear(project_dim, project_dim)
        )
        # 投影网络 2：结构与 proj_1 相同
        self.proj_2 = nn.Sequential(
            nn.Linear(hidden_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            act,
            nn.Linear(project_dim, project_dim)
        )
        # 对投影网络中的全连接层进行 kaiming 正态初始化
        for model in self.proj_1:
            if isinstance(model, nn.Linear):
                nn.init.kaiming_normal_(model.weight)
        for model in self.proj_2:
            if isinstance(model, nn.Linear):
                nn.init.kaiming_normal_(model.weight)

    def infoNCE_loss(self, query, key, pos_mask, chunk_size=1024):
        """
        使用 InfoNCE 风格的对比损失，采用分块计算避免一次性占用大量内存。

        Args:
            query: 投影后的 query 表示，形状 [N, d]
            key:   投影后的 key 表示，形状 [N, d]
            pos_mask: 二值正样本对掩码，形状 [N, N]（可能为稀疏张量）
            chunk_size: 分块大小，根据 GPU 内存大小可适当调整
        Returns:
            均值损失标量
        """
        # 如果 pos_mask 为稀疏张量，则先转换为稠密张量
        if pos_mask.is_sparse:
            pos_mask = pos_mask.to_dense()

        # 如果还未归一化，可以归一化 query 和 key（根据实际情况选择是否归一化）
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)

        total_loss = 0.0
        count = 0

        # 分块计算相似度矩阵，避免一次生成整个 [N, N] 的矩阵
        N = query.size(0)
        for i in range(0, N, chunk_size):
            # 取出 query 的一块
            query_chunk = query[i: i + chunk_size]  # [B, d] 其中 B<=chunk_size
            # 计算 query_chunk 与全部 key 的余弦相似度矩阵：[B, N]
            # 使用矩阵乘法代替 F.cosine_similarity 来实现同样效果，因为查询和键均归一化后点乘即可得到余弦相似度
            sim = torch.mm(query_chunk, key.t()) / self.tau
            exp_sim = torch.exp(sim)
            # 计算分子与分母
            pos_mask_chunk = pos_mask[i: i + chunk_size].to(exp_sim.dtype)  # [B, N]
            numerator = (exp_sim * pos_mask_chunk).sum(dim=1)
            denominator = exp_sim.sum(dim=1) + EPS
            loss_chunk = - torch.log(numerator / denominator)
            total_loss += loss_chunk.sum()
            count += loss_chunk.numel()
        return total_loss / count

    def forward(self, z_1, z_2, pos):
        """
        Args:
            z_1, z_2: 输入隐藏特征，形状为 [N, hidden_dim]（例如来自不同视角或频率）
            pos: 二值正样本对掩码，形状 [N, N]，1 表示正样本对（可能为稀疏张量）
        Returns:
            对比损失标量
        """
        # 分别对两个特征进行投影
        z_proj_1 = self.proj_1(z_1)
        z_proj_2 = self.proj_2(z_2)
        # 计算两个方向的 InfoNCE 损失，并取平均
        loss_1 = self.infoNCE_loss(z_proj_1, z_proj_2, pos)
        loss_2 = self.infoNCE_loss(z_proj_2, z_proj_1, pos.t())
        return (loss_1 + loss_2) / 2




def sce_loss(x, y, beta=1):
    """
    简单余弦距离重构损失：
      - 对 x 与 y 进行 L2 归一化，
      - 计算它们的余弦相似度，并计算 (1 - cos_sim)^beta，
      - 取平均后乘上常数因子 10 以平衡损失。
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(beta)
    loss = loss.mean()
    return 10 * loss

