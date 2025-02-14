import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class OTAlignment(nn.Module):
    """基于最优传输的特征对齐模块"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 特征投影层
        self.img_projection = nn.Linear(config.VIT_FEATURE_DIM, config.HIDDEN_DIM)
        self.eds_projection = nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM)
        self.xrd_projection = nn.Linear(config.HIDDEN_DIM // 2, config.HIDDEN_DIM)

        # 成本矩阵生成网络
        self.cost_net = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, 1)
        )

    def compute_cost_matrix(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """计算成本矩阵"""
        n1, n2 = x1.size(0), x2.size(0)
        x1_expand = x1.unsqueeze(1).expand(-1, n2, -1)
        x2_expand = x2.unsqueeze(0).expand(n1, -1, -1)
        paired = torch.cat([x1_expand, x2_expand], dim=2)
        return self.cost_net(paired).squeeze(-1)

    def sinkhorn(self, cost: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """Sinkhorn迭代算法"""
        n1, n2 = cost.size()
        u = torch.zeros(n1, device=cost.device)
        v = torch.zeros(n2, device=cost.device)

        for _ in range(self.config.OT_MAX_ITER):
            u_old = u.clone()
            u = (torch.log(mu + 1e-8) -
                 torch.logsumexp(-cost / self.config.OT_EPSILON + v.unsqueeze(0), dim=1))
            v = (torch.log(nu + 1e-8) -
                 torch.logsumexp(-cost / self.config.OT_EPSILON + u.unsqueeze(1), dim=0))

            if torch.abs(u - u_old).mean() < self.config.OT_THRESHOLD:
                break

        P = torch.exp(-cost / self.config.OT_EPSILON + u.unsqueeze(1) + v.unsqueeze(0))
        return P

    def align_features(self, features_list: List[torch.Tensor]) -> Tuple[torch.Tensor, List]:
        """对齐多模态特征"""
        img_feat = self.img_projection(features_list[0])
        eds_feat = self.eds_projection(features_list[1])
        xrd_feat = self.xrd_projection(features_list[2])

        batch_size = img_feat.size(0)
        aligned_features = []
        transport_matrices = []

        for i in range(batch_size):
            # 单个样本的特征对齐
            img_i = img_feat[i].unsqueeze(0)
            eds_i = eds_feat[i].unsqueeze(0)
            xrd_i = xrd_feat[i].unsqueeze(0)

            # 计算传输矩阵
            cost_img_eds = self.compute_cost_matrix(img_i, eds_i)
            cost_img_xrd = self.compute_cost_matrix(img_i, xrd_i)

            mu = torch.ones(img_i.size(0), device=img_i.device) / img_i.size(0)
            nu_eds = torch.ones(eds_i.size(0), device=eds_i.device) / eds_i.size(0)
            nu_xrd = torch.ones(xrd_i.size(0), device=xrd_i.device) / xrd_i.size(0)

            P_img_eds = self.sinkhorn(cost_img_eds, mu, nu_eds)
            P_img_xrd = self.sinkhorn(cost_img_xrd, mu, nu_xrd)

            # 特征对齐
            aligned_eds = torch.mm(P_img_eds, eds_i)
            aligned_xrd = torch.mm(P_img_xrd, xrd_i)

            aligned_features.append(torch.cat([img_i, aligned_eds, aligned_xrd], dim=1))
            transport_matrices.append([P_img_eds, P_img_xrd])

        aligned_features = torch.cat(aligned_features, dim=0)
        return aligned_features, transport_matrices

    def forward(self, image_features: torch.Tensor, eds_features: torch.Tensor,
                xrd_features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features_list = [image_features, eds_features, xrd_features]
        aligned_features, _ = self.align_features(features_list)
        return aligned_features


class R2ETransport(nn.Module):
    """特征传输模块"""

    def __init__(self, config):
        super().__init__()
        self.transport_net = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 2, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        )

    def forward(self, source_feat: torch.Tensor, target_feat: torch.Tensor,
                transport_matrix: torch.Tensor) -> torch.Tensor:
        """执行特征传输"""
        return torch.matmul(transport_matrix,
                            self.transport_net(torch.cat([source_feat, target_feat], dim=-1)))