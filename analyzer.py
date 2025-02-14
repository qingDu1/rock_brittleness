import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from typing import Dict, List, Tuple


class FeatureAnalyzer:
    """特征分析工具"""

    def __init__(self, config):
        self.config = config
        self.features_df = None
        self.target = None
        self.feature_importances = None

    def set_data(self, features_df: pd.DataFrame, target: pd.Series):
        """设置要分析的数据"""
        self.features_df = features_df.copy()
        self.target = pd.to_numeric(target, errors='coerce')

        # 移除无效值
        mask = ~(np.isnan(self.target) | np.isinf(self.target))
        self.features_df = self.features_df[mask]
        self.target = self.target[mask]

    def analyze_correlations(self) -> pd.DataFrame:
        """分析特征相关性"""
        if self.features_df is None or self.target is None:
            raise ValueError("请先使用set_data()设置数据")

        correlations = {}
        for column in self.features_df.columns:
            try:
                feat_data = self.features_df[column].values
                target_data = self.target.values
                mask = ~(np.isnan(feat_data) | np.isinf(feat_data))
                feat_data = feat_data[mask]
                target_data = target_data[mask]

                if len(feat_data) > 0:
                    # 计算各种相关性指标
                    pearson_corr, p_value = pearsonr(feat_data, target_data)
                    spearman_corr, sp_value = spearmanr(feat_data, target_data)
                    mutual_info = mutual_info_regression(
                        feat_data.reshape(-1, 1),
                        target_data
                    )[0]

                    correlations[column] = {
                        'pearson': pearson_corr,
                        'pearson_p': p_value,
                        'spearman': spearman_corr,
                        'spearman_p': sp_value,
                        'mutual_info': mutual_info
                    }

            except Exception as e:
                print(f"分析特征 '{column}' 时出错: {str(e)}")
                correlations[column] = {
                    'pearson': np.nan,
                    'pearson_p': np.nan,
                    'spearman': np.nan,
                    'spearman_p': np.nan,
                    'mutual_info': np.nan
                }

        return pd.DataFrame(correlations).T

    def plot_correlation_heatmap(self, save_path: str = None):
        """绘制相关性热图"""
        if self.features_df is None:
            raise ValueError("请先使用set_data()设置数据")

        plt.figure(figsize=(15, 12))
        correlation_matrix = self.features_df.corr()

        # 使用seaborn绘制热图
        sns.heatmap(correlation_matrix,
                    annot=True,
                    cmap='coolwarm',
                    center=0,
                    fmt='.2f')

        plt.title('特征相关性热图')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def analyze_feature_importance(self) -> pd.DataFrame:
        """分析特征重要性"""
        if self.features_df is None or self.target is None:
            raise ValueError("请先使用set_data()设置数据")

        # 获取相关性分析结果
        correlations = self.analyze_correlations()

        # 计算综合重要性分数
        correlations['importance_score'] = (
                                                   np.abs(correlations['pearson']) +
                                                   np.abs(correlations['spearman']) +
                                                   correlations['mutual_info']
                                           ) / 3

        # 按重要性排序
        self.feature_importances = correlations.sort_values('importance_score',
                                                            ascending=False)
        return self.feature_importances

    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """绘制特征重要性图"""
        if self.feature_importances is None:
            self.analyze_feature_importance()

        plt.figure(figsize=(12, 6))
        importance_scores = self.feature_importances['importance_score']
        top_features = importance_scores.head(top_n)

        # 绘制条形图
        sns.barplot(x=top_features.values,
                    y=top_features.index,
                    palette='viridis')

        plt.title(f'Top {top_n} 特征重要性')
        plt.xlabel('重要性分数')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def generate_analysis_report(self):
        """生成分析报告"""
        if self.features_df is None or self.target is None:
            raise ValueError("请先使用set_data()设置数据")

        # 特征重要性分析
        importance_df = self.analyze_feature_importance()

        # 保存相关性热图
        heatmap_path = os.path.join(self.config.EXPERIMENT_DIR,
                                    'correlation_heatmap.png')
        self.plot_correlation_heatmap(heatmap_path)

        # 保存特征重要性图
        importance_path = os.path.join(self.config.EXPERIMENT_DIR,
                                       'feature_importance.png')
        self.plot_feature_importance(save_path=importance_path)

        # 保存分析结果
        importance_df.to_csv(os.path.join(self.config.EXPERIMENT_DIR,
                                          'feature_importance.csv'))

        # 生成报告文本
        report = {
            "数据概况": {
                "特征数量": len(self.features_df.columns),
                "样本数量": len(self.features_df),
                "目标变量范围": f"{self.target.min():.2f} - {self.target.max():.2f}"
            },
            "top_10_特征": importance_df.head(10).to_dict(),
            "分析文件": {
                "相关性热图": heatmap_path,
                "特征重要性图": importance_path,
                "详细分析结果": "feature_importance.csv"
            }
        }

        # 保存报告
        report_path = os.path.join(self.config.EXPERIMENT_DIR, 'analysis_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        return report