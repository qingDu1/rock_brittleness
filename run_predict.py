import argparse
import torch
import pandas as pd
from config.config import Config
from models.multimodal import MultiModalRockModel
from data.dataset import MultiModalRockDataset
from torch.utils.data import DataLoader
import logging


def predict(model_path: str, data_path: str = None):
    """预测函数"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # 加载配置
        config = Config()
        if data_path:
            config.IMAGE_PATH = data_path

        # 加载数据
        dataset = MultiModalRockDataset(config, transform=None)
        dataloader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )

        # 加载模型
        model = MultiModalRockModel(
            numerical_dim=dataset.feature_dim,
            config=config
        ).to(config.DEVICE)

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 预测
        predictions = []
        filenames = []

        with torch.no_grad():
            for batch in dataloader:
                images, numerical, _, batch_filenames = batch
                images = images.to(config.DEVICE)
                numerical = numerical.to(config.DEVICE)

                outputs = model(images, numerical)
                predictions.extend(outputs.cpu().numpy())
                filenames.extend(batch_filenames)

        # 创建预测结果DataFrame
        results_df = pd.DataFrame({
            'Filename': filenames,
            'Predicted_Brittleness': predictions
        })

        # 保存结果
        save_path = os.path.join(config.EXPERIMENT_DIR, 'predictions.csv')
        results_df.to_csv(save_path, index=False)
        logger.info(f"预测结果已保存至: {save_path}")

        return results_df

    except Exception as e:
        logger.error(f"预测过程出错: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='岩石脆性预测')
    parser.add_argument('--model', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--data', type=str, default=None,
                        help='数据目录路径')

    args = parser.parse_args()
    predict(args.model, args.data)