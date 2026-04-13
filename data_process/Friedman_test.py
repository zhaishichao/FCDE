import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 配置
# ======================
methods = [
    "GP-SMOTE",
    "DG-SMOTE",
    "RAW",
    "ROS",
    "SMOTE",
    "SMOTEN",
    "Borderline_1",
    "Borderline_2"
]

method_display = {
    "GP-SMOTE": "GP-SMOTE",
    "DG-SMOTE": "DG-SMOTE",
    "RAW": "Original",
    "ROS": "ROS",
    "SMOTE": "SMOTE",
    "SMOTEN": "SMOTEN",
    "Borderline_1": "B-SMOTE-1",
    "Borderline_2": "B-SMOTE-2"
}

metrics = ["F-measure", "AUC"]
root_dir = "your_mean_results_path"


# ======================
# 读取数据
# ======================
def load_data(metric):
    data = []
    for m in methods:
        df = pd.read_csv(os.path.join(root_dir, f"{m}.csv"))
        df = df.sort_values(by="Dataset")
        data.append(df[metric].values)
    return np.array(data).T  # datasets × methods


# ======================
# 计算排名
# ======================
def compute_ranks(data):
    ranks = []
    for row in data:
        rank = pd.Series(row).rank(ascending=False, method="average")
        ranks.append(rank.values)
    return np.array(ranks)


# ======================
# 绘制 Friedman 排名图
# ======================
def plot_friedman_style(ranks, metric):
    # 平均排名
    mean_ranks = ranks.mean(axis=0)

    # 标准误（推荐，比std更常用）
    sem = ranks.std(axis=0) / np.sqrt(ranks.shape[0])

    df = pd.DataFrame({
        "Method": methods,
        "MeanRank": mean_ranks,
        "SEM": sem
    })

    # 排序（从好到差：rank小 → 前）
    df = df.sort_values(by="MeanRank", ascending=True)

    # ===== 画图 =====
    plt.figure(figsize=(6, 4))

    y_pos = np.arange(len(df))

    plt.errorbar(
        df["MeanRank"],
        y_pos,
        xerr=df["SEM"],
        fmt='o',
        capsize=4
    )

    plt.yticks(y_pos, [method_display[m] for m in df["Method"]])

    plt.xlabel("Mean Rank")
    plt.title(f"{metric} Friedman test result")

    plt.gca().invert_yaxis()  # 让最优在上

    plt.tight_layout()
    plt.savefig(f"friedman_{metric}.png", dpi=300)
    plt.show()


# ======================
# 主流程
# ======================
for metric in metrics:
    data = load_data(metric)
    ranks = compute_ranks(data)
    plot_friedman_style(ranks, metric)