"""
运行时间基准测试的公共模块。

各 SMOTE 变体只需提供一个"单次运行"的可调用对象，其余公共流程
（数据加载、标准化、计时、均值统计、CSV 保存）由 benchmark() 统一完成，
避免在多个测试脚本中重复相同的样板代码。
"""

import os
import time
import warnings

import pandas as pd

from data_preprocess import data_loader, data_preprocess
from config import datasetnames_final_1

warnings.filterwarnings("ignore")

# 项目根目录（本文件位于 test/test_runtime/ 下，向上两级即项目根目录）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DAT_DIR = os.path.join(PROJECT_ROOT, "datasets", "dat")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "test", "results", "runtime")


def benchmark(algorithm_name, run_once, save_dir, num_run=3,
              datasets=None, seed_base=42, standard=True):
    """对指定算法做运行时间基准测试。

    Parameters
    ----------
    algorithm_name : str
        算法名称，仅用于打印提示。
    run_once : callable
        单次运行的函数，签名为 run_once(X_train, y_train, seed) -> None，
        内部完成采样器的构造与拟合，计时由本函数负责。
    save_dir : str
        结果子目录名（相对于 test/results/runtime/）。
    num_run : int
        每个数据集重复运行的次数。
    datasets : list, optional
        数据集名列表，默认取 config 中的 datasetnames_final_1。
    seed_base : int
        随机种子基数，第 i 次运行使用 seed_base + i。
    standard : bool
        是否对特征做标准化。
    """
    datasets = datasets if datasets is not None else datasetnames_final_1
    save_path = os.path.join(RESULTS_DIR, save_dir)
    os.makedirs(save_path, exist_ok=True)

    df_mean = pd.DataFrame(columns=["数据集", "time"])
    print(f"########\t {algorithm_name} 运行时间测试开始！\t########")

    for index, datasetname in enumerate(datasets):
        print(f"##########\t 正在处理：{datasetname} \t##########")

        X, y = data_loader(os.path.join(DAT_DIR, datasetname + ".dat"))
        times = []
        for i in range(num_run):
            X_train, X_test, y_train, y_test = data_preprocess(
                X, y, standard=standard, random_state=seed_base + i)

            start = time.perf_counter()
            run_once(X_train, y_train, seed_base + i)
            times.append(time.perf_counter() - start)

        df_time = pd.DataFrame({"time": times})
        df_mean.loc[index] = [datasetname, df_time["time"].mean()]

        df_time.to_csv(os.path.join(save_path, datasetname + ".csv"),
                       encoding="utf_8_sig", index=False)
        df_mean.to_csv(os.path.join(save_path, "mean.csv"),
                       encoding="utf_8_sig", index=False)

    print(f"########\t {algorithm_name} 运行时间测试结束！\t########")
