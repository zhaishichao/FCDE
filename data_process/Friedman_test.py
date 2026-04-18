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
    "Borderline_1": "Borderline-1",
    "Borderline_2": "Borderline-2"
}

metrics = ["F-measure", "AUC"]
root_dir = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\F1和AUC\\knn\\"


# ======================
# 读取数据
# ======================
def load_data(metric):
    data = []
    for m in methods:
        df = pd.read_csv(os.path.join(root_dir, f"{m}.csv"))
        df = df.sort_values(by="数据集")
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

# ===== 字体（论文推荐）=====
# 将默认衬线字体设为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
def plot_friedman_style(ranks, metric):

    # ===== 平均排名 & 标准误 =====
    mean_ranks = ranks.mean(axis=0)
    sem = ranks.std(axis=0) / np.sqrt(ranks.shape[0])

    df = pd.DataFrame({
        "Method": methods,
        "MeanRank": mean_ranks,
        "SEM": sem
    })

    df = df.sort_values(by="MeanRank", ascending=True)

    # ===== 颜色（每个方法不同）=====
    colors = plt.cm.tab10.colors  # 10种颜色够用

    plt.figure(figsize=(6, 4))

    y_pos = np.arange(len(df))

    # ===== 每一行单独画（实现不同颜色）=====
    for i in range(len(df)):
        plt.errorbar(
            df["MeanRank"].iloc[i],
            y_pos[i],
            xerr=df["SEM"].iloc[i],
            fmt='o',
            color=colors[i % len(colors)],
            ecolor=colors[i % len(colors)],
            capsize=4,
            markersize=6
        )

    # ===== 坐标轴 =====
    plt.yticks(y_pos, [method_display[m] for m in df["Method"]])
    plt.xlabel("Mean Rank")
    plt.title(f"{metric} Average Rank")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    # ===== 保存为PDF =====
    plt.savefig(f"friedman_{metric}.pdf", bbox_inches='tight')

    plt.show()


# ======================
# 主流程
# ======================
for metric in metrics:
    data = load_data(metric)
    ranks = compute_ranks(data)
    plot_friedman_style(ranks, metric)