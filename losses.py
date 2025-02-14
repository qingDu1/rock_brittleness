import torch
import torch.nn as nn


class CustomRegressionLoss(nn.Module):
    """自定义回归损失函数"""

    def __init__(self, config):
        super().__init__()
        self.huber = nn.HuberLoss(delta=config.HUBER_DELTA)
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.mse_weight = config.MSE_WEIGHT
        self.mae_weight = config.MAE_WEIGHT

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber_loss = self.huber(pred, target)
        mse_loss = self.mse(pred, target)
        mae_loss = self.mae(pred, target)

        return (self.mse_weight * mse_loss +
                self.mae_weight * mae_loss +
                huber_loss)