import torch
import torch.nn as nn
from .vit import VisionTransformer
from .fcnn import FCNN
from .ot_align import OTAlignment, R2ETransport


class MultiModalRockModel(nn.Module):
    """多模态岩石分析模型"""

    def __init__(self, numerical_dim: int, config):
        super().__init__()
        self.config = config

        # 视觉特征提取
        self.vit = VisionTransformer(config)

        # 数值特征处理
        self.eds_network = FCNN(numerical_dim // 2, config)
        self.xrd_network = FCNN(numerical_dim // 2, config)

        # 特征对齐和传输
        self.ot_alignment = OTAlignment(config)
        self.r2e_transport = R2ETransport(config)

        # 特征融合
        fusion_dim = config.VIT_FEATURE_DIM + config.HIDDEN_DIM * 2
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE)
        )

        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=config.HIDDEN_DIM,
            num_heads=4,
            dropout=config.DROPOUT_RATE,
            batch_first=True
        )

        # 预测头
        self.regression_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE // 2),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, image: torch.Tensor, numerical: torch.Tensor) -> torch.Tensor:
        # 特征分离与提取
        eds_features = numerical[:, :numerical.size(1) // 2]
        xrd_features = numerical[:, numerical.size(1) // 2:]

        image_features = self.vit(image)
        eds_features = self.eds_network(eds_features)
        xrd_features = self.xrd_network(xrd_features)

        # 特征对齐
        aligned_features = self.ot_alignment(image_features, eds_features, xrd_features)

        # 特征融合
        fused_features = self.fusion_network(aligned_features)
        fused_features = fused_features.unsqueeze(1)
        attn_output, _ = self.attention(fused_features, fused_features, fused_features)
        fused_features = attn_output.squeeze(1)

        # 预测
        output = self.regression_head(fused_features)
        return output.squeeze()