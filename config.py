import os
from datetime import datetime
import torch


class Config:
    """配置类"""

    def __init__(self):
        # 基础路径配置
        self.PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_ROOT = os.path.join(self.PROJECT_ROOT, 'data')

        # 数据路径
        self.IMAGE_PATH = os.path.join(self.DATA_ROOT, 'image')
        self.EDS_PATH = os.path.join(self.DATA_ROOT, 'EDS_DATA.xlsx')
        self.XRD_PATH = os.path.join(self.DATA_ROOT, 'XRD_data.xlsx')
        self.OUTPUT_PATH = os.path.join(self.DATA_ROOT, 'Brittleness_DATA.xlsx')

        # 训练参数
        self.BATCH_SIZE = 16
        self.NUM_EPOCHS = 100
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 1e-4

        # 模型参数
        self.IMAGE_SIZE = 224
        self.HIDDEN_DIM = 256
        self.DROPOUT_RATE = 0.3
        self.FCNN_LAYERS = [512, 256, 128, 64]
        self.FCNN_DROPOUT_DECAY = 0.5
        self.VIT_FEATURE_DIM = 768

        # 损失函数参数
        self.HUBER_DELTA = 1.0
        self.MSE_WEIGHT = 0.7
        self.MAE_WEIGHT = 0.3

        # 最优传输参数
        self.OT_EPSILON = 0.1
        self.OT_MAX_ITER = 50
        self.OT_THRESHOLD = 1e-6

        # 实验记录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.EXPERIMENT_DIR = os.path.join(self.PROJECT_ROOT, 'experiments', timestamp)
        os.makedirs(self.EXPERIMENT_DIR, exist_ok=True)

        # 设备配置
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

