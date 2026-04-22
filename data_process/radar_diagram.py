import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =============================
# 数据集名称
# =============================
datasets = [f"D{i}" for i in range(1, 26)]

# =============================
# 填入你的数据（25个）
# =============================
rootpath = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\mean\\lw\\knn\\"
df_f = pd.read_csv(os.path.join(rootpath, "f1_1.csv"))  # GP-SMOTE F-measure
df_auc = pd.read_csv(os.path.join(rootpath, "auc_1.csv"))  # GP-SMOTE AUC
df_woc = pd.read_csv(os.path.join(rootpath, "woc.csv"))  # GP-SMOTE-woc

gp_f = list(df_f["GP-SMOTE"])  # GP-SMOTE F-measure
gp_auc = list(df_auc["GP-SMOTE"])  # GP-SMOTE-woc F-measure

woc_f = list(df_woc["F-measure"])  # GP-SMOTE AUC
woc_auc = list(df_woc["AUC"])  # GP-SMOTE-woc AUC


# =============================
# 雷达图函数
# =============================
def radar_plot(labels, values1, values2, title, save_name):
    N = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    values1 = values1 + values1[:1]
    values2 = values2 + values2[:1]
    # sns.set_theme(style='whitegrid')
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    # 起点在顶部
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # 标签
    plt.xticks(angles[:-1], labels, fontsize=9)
    # plt.yticks(np.linspace(40, 100, 20), fontsize=9)

    # 线
    ax.plot(angles, values1, linewidth=2, label="GP-SMOTE")
    ax.fill(angles, values1, alpha=0.15)

    ax.plot(angles, values2, linewidth=2, label="GP-SMOTE-woc")
    ax.fill(angles, values2, alpha=0.15)

    plt.title(title, y=1.08, fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    plt.tight_layout()
    plt.savefig(save_name + ".pdf", dpi=300)
    plt.show()


# =============================
# 画图
# =============================
radar_plot(datasets, gp_f, woc_f,
           "F-measure Comparison",
           "radar_fmeasure")

radar_plot(datasets, gp_auc, woc_auc,
           "AUC Comparison",
           "radar_auc")
