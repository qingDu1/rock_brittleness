import torch
from torch import nn, optim
import os
import logging
import json
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

class ModelTrainer:
    """模型训练器"""
    def __init__(self, model: nn.Module, criterion: nn.Module, config):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = config.DEVICE

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            verbose=True
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'best_val_loss': float('inf')
        }

        self.logger = logging.getLogger(__name__)

    def train_epoch(self, train_loader) -> float:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            images, numerical, targets, _ = [x.to(self.device)
                                          if torch.is_tensor(x) else x
                                          for x in batch]

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images, numerical)
            loss = self.criterion(outputs, targets)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    def validate(self, val_loader) -> float:
        """验证模型"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images, numerical, targets, _ = [x.to(self.device)
                                              if torch.is_tensor(x) else x
                                              for x in batch]
                outputs = self.model(images, numerical)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def train(self, train_loader, val_loader):
        """完整训练流程"""
        self.logger.info(f"开始训练 - 使用设备: {self.device}")

        for epoch in range(self.config.NUM_EPOCHS):
            # 训练和验证
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 更新历史记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)

            # 保存最佳模型
            if val_loss < self.history['best_val_loss']:
                self.history['best_val_loss'] = val_loss
                self.save_checkpoint(epoch, 'best_model.pth')
                self.logger.info(f"Epoch {epoch + 1}: 保存新的最佳模型 (验证损失: {val_loss:.4f})")

            # 打印进度
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f'Epoch {epoch + 1}/{self.config.NUM_EPOCHS}: '
                    f'训练损失: {train_loss:.4f}, '
                    f'验证损失: {val_loss:.4f}, '
                    f'学习率: {current_lr:.6f}'
                )

            # 保存训练历史
            self.save_training_history()

    def evaluate(self, test_loader) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """评估模型性能"""
        self.model.eval()
        predictions = []
        actual_values = []
        filenames = []

        with torch.no_grad():
            for batch in test_loader:
                images, numerical, targets, batch_filenames = batch
                images = images.to(self.device)
                numerical = numerical.to(self.device)

                outputs = self.model(images, numerical)
                predictions.extend(outputs.cpu().numpy())
                actual_values.extend(targets.numpy())
                filenames.extend(batch_filenames)

        predictions = np.array(predictions)
        actual_values = np.array(actual_values)

        # 计算评估指标
        metrics = {
            'MSE': mean_squared_error(actual_values, predictions),
            'RMSE': np.sqrt(mean_squared_error(actual_values, predictions)),
            'MAE': mean_absolute_error(actual_values, predictions),
            'R2': r2_score(actual_values, predictions)
        }

        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'Filename': filenames,
            'Actual': actual_values,
            'Predicted': predictions,
            'Absolute_Error': np.abs(predictions - actual_values)
        })

        # 保存预测结果图
        self.plot_predictions(actual_values, predictions, metrics)

        return results_df, metrics

    def plot_predictions(self, actual: np.ndarray, predicted: np.ndarray,
                        metrics: Dict[str, float]):
        """绘制预测结果图"""
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([actual.min(), actual.max()],
                [actual.min(), actual.max()],
                'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('预测结果对比')

        # 添加评估指标文本
        metrics_text = '\n'.join([
            f'{k}: {v:.4f}' for k, v in metrics.items()
        ])
        plt.text(0.05, 0.95, metrics_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')

        plt.savefig(os.path.join(self.config.EXPERIMENT_DIR, 'predictions.png'))
        plt.close()

    def save_checkpoint(self, epoch: int, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(self.config.EXPERIMENT_DIR, filename))

    def load_checkpoint(self, filename: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(os.path.join(self.config.EXPERIMENT_DIR, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['epoch']

    def save_training_history(self):
        """保存训练历史"""
        # 绘制损失曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失')
        plt.plot(self.history['val_loss'], label='验证损失')
        plt.title('损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.history['learning_rates'])
        plt.title('学习率变化')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.EXPERIMENT_DIR, 'training_history.png'))
        plt.close()

        # 保存历史数据
        history_path = os.path.join(self.config.EXPERIMENT_DIR, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)