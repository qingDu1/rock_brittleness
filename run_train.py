import os
import logging
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from config.config import Config
from models.multimodal import MultiModalRockModel
from data.dataset import MultiModalRockDataset
from trainer.trainer import ModelTrainer
from utils.losses import CustomRegressionLoss


def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    try:
        # 加载配置
        config = Config()
        logger.info(f"使用设备: {config.DEVICE}")

        # 数据增强
        transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 创建数据集
        logger.info("正在加载数据集...")
        dataset = MultiModalRockDataset(config, transform=transform)

        # 数据集划分
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=4,
            pin_memory=True
        )

        # 初始化模型
        logger.info("正在初始化模型...")
        model = MultiModalRockModel(
            numerical_dim=dataset.feature_dim,
            config=config
        ).to(config.DEVICE)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")

        # 创建损失函数和训练器
        criterion = CustomRegressionLoss(config)
        trainer = ModelTrainer(model, criterion, config)

        # 开始训练
        logger.info("开始训练...")
        trainer.train(train_loader, val_loader)

        # 保存最终模型
        trainer.save_model('final_model.pth')
        logger.info("训练完成！")

    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()

