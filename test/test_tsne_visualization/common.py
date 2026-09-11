"""
t-SNE 三维分类可视化公共模块。

对每种采样方法：生成合成样本，与原始数据合并（合成样本标记为类别 2，
与原始多数类 0、少数类 1 区分），绘制三分类 t-SNE 图，并把合成后的
原始数据（含标签列）保存为 CSV。

结果按方法分文件夹保存：
    results/<method>/synthetic_data/   合成后的原始数据 CSV
    results/<method>/visualization/     t-SNE 可视化图 PNG
"""

import os
import warnings

# joblib 在 Windows 上探测物理核失败会告警，这里直接指定核心数，从源头避免
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data_preprocess import data_loader, data_preprocess
from config import datasetnames
from smote_variants.gp_smote.visualize import tsne_visualization_binary

# 项目根目录（本文件位于 test/test_tsne_visualization/ 下）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DAT_DIR = os.path.join(PROJECT_ROOT, "datasets", "dat")
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_tsne(method_name, make_sampler, extract_synthetic, datasets=None, n_runs=1):
    """运行 t-SNE 可视化。

    Parameters
    ----------
    method_name : str
        方法名，用于 results/ 下的子文件夹命名。
    make_sampler : callable
        make_sampler(seed) -> sampler，返回一个已设置 res_only=True 的采样器。
    extract_synthetic : callable
        extract_synthetic(sampler, X_train, y_train) -> X_syn，返回合成样本特征矩阵。
    datasets : list or None
        数据集列表，默认 config.datasetnames。
    n_runs : int
        每个数据集的运行次数。
    """
    datasets = datasets if datasets is not None else datasetnames
    synth_dir = os.path.join(RESULTS_ROOT, method_name, "synthetic_data")
    vis_dir = os.path.join(RESULTS_ROOT, method_name, "visualization")
    os.makedirs(synth_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print(f"########\t {method_name} t-SNE 可视化开始！\t########")

    for datasetname in datasets:
        print(f"##########\t 正在处理：{datasetname} \t##########")
        X, y = data_loader(os.path.join(DAT_DIR, datasetname + ".dat"))

        for i in range(n_runs):
            X_train, X_test, y_train, y_test = data_preprocess(
                X, y, standard=True, random_state=42 + i)

            sampler = make_sampler(42 + i)
            X_syn = np.asarray(extract_synthetic(sampler, X_train, y_train))
            if len(X_syn) == 0:
                continue

            # 合成样本标记为类别 2
            y_syn = [2] * len(X_syn)
            X_vis = np.vstack((X_train, X_syn))
            y_vis = np.hstack((y_train, y_syn))

            # 保存合成后的原始数据（最后一列为标签）
            data_with_label = np.hstack((X_vis, np.array([y_vis]).T))
            csv_path = os.path.join(synth_dir, f"{datasetname}_{i + 1}.csv")
            pd.DataFrame(data_with_label).to_csv(csv_path, index=False)

            # 绘制 t-SNE 可视化图
            tsne_visualization_binary(
                X_vis, y_vis, vis_dir, f"{datasetname}_{method_name}_{i + 1}")

    print(f"########\t {method_name} t-SNE 可视化结束！\t########")
