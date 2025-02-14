import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from extractors.image_extractor import ImageFeatureExtractor
from extractors.eds_extractor import EDSFeatureExtractor
from extractors.xrd_extractor import XRDFeatureExtractor


class MultiModalRockDataset(Dataset):
    """多模态岩石数据集"""

    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform
        self.feature_extractors = {
            'image': ImageFeatureExtractor(),
            'eds': EDSFeatureExtractor(),
            'xrd': XRDFeatureExtractor()
        }
        self.load_data()

    def load_data(self):
        """加载并预处理数据"""
        # 加载图像文件
        self.image_files = sorted([
            f for f in os.listdir(self.config.IMAGE_PATH)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ])
        image_ids = [f.split('.')[0] for f in self.image_files]

        # 加载表格数据
        self.eds_data = pd.read_excel(self.config.EDS_PATH)
        self.xrd_data = pd.read_excel(self.config.XRD_PATH)
        self.output_data = pd.read_excel(self.config.OUTPUT_PATH)

        # 数据清理和对齐
        for df in [self.eds_data, self.xrd_data, self.output_data]:
            df['Sample_ID'] = df['Sample_ID'].astype(str)

        # 数据合并
        data_df = pd.merge(self.eds_data, self.xrd_data, on='Sample_ID', how='inner')
        data_df = pd.merge(data_df, self.output_data, on='Sample_ID', how='inner')
        data_df = data_df[data_df['Sample_ID'].isin(image_ids)]

        # 更新图像文件列表
        self.image_files = [f for f in self.image_files
                            if f.split('.')[0] in data_df['Sample_ID'].values]

        # 特征提取
        self.features = []
        for idx, image_file in enumerate(self.image_files):
            image_path = os.path.join(self.config.IMAGE_PATH, image_file)
            image = np.array(Image.open(image_path).convert('L'))

            features = {
                'image': self.feature_extractors['image'].extract_features(image),
                'eds': self.feature_extractors['eds'].extract_features(
                    self.eds_data.iloc[[idx]]),
                'xrd': self.feature_extractors['xrd'].process_features(
                    self.xrd_data.iloc[[idx]])
            }
            self.features.append(features)

        # 特征标准化
        self.scaler = StandardScaler()
        feature_matrix = np.array([list(f['image'].values()) +
                                   list(f['eds'].values()) +
                                   list(f['xrd'].values())
                                   for f in self.features])
        self.normalized_features = self.scaler.fit_transform(feature_matrix)
        self.feature_dim = self.normalized_features.shape[1]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        # 加载图像
        image_path = os.path.join(self.config.IMAGE_PATH, self.image_files[idx])
        image = Image.open(image_path).convert('L')
        image = Image.merge('RGB', (image, image, image))

        if self.transform:
            image = self.transform(image)

        # 获取特征和目标值
        features = torch.tensor(self.normalized_features[idx], dtype=torch.float32)
        target = torch.tensor(
            self.output_data['Brittleness'].iloc[idx],
            dtype=torch.float32
        )

        return image, features, target, self.image_files[idx]