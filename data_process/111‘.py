import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===============================
# 数据
# ===============================
datasets = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10',
            'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19',
            'D20', 'D21', 'D22', 'D23', 'D24', 'D25']

gp_f = [84.31, 82.37, 89.95, 96.08, 96.71, 69.29, 100.0, 73.68, 65.47, 83.75,
        66.13, 95.14, 86.44, 76.8, 94.52, 90.75, 98.12, 78.42, 84.57, 83.91,
        64.13, 99.31, 82.16, 50.85, 100.0]

woc_f = [84.16, 82.27, 89.71, 96.07, 96.66, 69.01, 100.0, 73.32, 65.3, 82.78,
         65.89, 95.1, 86.17, 76.36, 94.42, 90.47, 97.99, 77.57, 84.69, 84.22,
         63.4, 97.77, 82.24, 51.35, 99.24]

gp_auc = [90.13, 88.89, 95.1, 98.39, 98.99, 76.73, 100.0, 83.38, 72.21, 92.13,
          73.92, 99.12, 93.46, 79.51, 99.72, 95.25, 99.76, 83.86, 90.56, 97.35,
          78.84, 99.44, 79.7, 59.57, 100.0]

woc_auc = [90.06, 88.61, 94.89, 98.39, 99.02, 76.54, 100.0, 82.76, 72.23, 91.77,
           73.9, 99.08, 93.28, 79.24, 99.72, 95.17, 99.72, 83.67, 90.16, 96.92,
           78.17, 98.33, 78.85, 60.13, 100.0]

# ===============================
# 构造 DataFrame
# ===============================
df = pd.DataFrame({
    "Dataset": datasets,
    "F_diff": np.array(gp_f) - np.array(woc_f),
    "AUC_diff": np.array(gp_auc) - np.array(woc_auc)
})

# ===============================
# 风格设置
# ===============================
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "pdf.fonttype": 42
})

# ==================================================
# 一、配对差值图（Scatter）
# ==================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# ---------- F1 ----------
colors_f = ["green" if x >= 0 else "red" for x in df["F_diff"]]

sns.scatterplot(
    x="Dataset",
    y="F_diff",
    data=df,
    s=160,
    hue=colors_f,
    palette={"green":"green", "red":"red"},
    legend=False,
    ax=axes[0]
)

axes[0].axhline(0, linestyle="--", linewidth=1.2)
axes[0].set_title("F-measure Difference")
axes[0].set_ylabel("GP-SMOTE - GP-SMOTE-woc")
axes[0].tick_params(axis='x', rotation=65)

# ---------- AUC ----------
colors_auc = ["green" if x >= 0 else "red" for x in df["AUC_diff"]]

sns.scatterplot(
    x="Dataset",
    y="AUC_diff",
    data=df,
    s=160,
    hue=colors_auc,
    palette={"green":"green", "red":"red"},
    legend=False,
    ax=axes[1]
)

axes[1].axhline(0, linestyle="--", linewidth=1.2)
axes[1].set_title("AUC Difference")
axes[1].set_ylabel("GP-SMOTE - GP-SMOTE-woc")
axes[1].tick_params(axis='x', rotation=65)

plt.tight_layout()
plt.savefig("paired_difference_plot.pdf", bbox_inches="tight")
plt.show()


# ==================================================
# 二、棒棒糖图（Lollipop）
# ==================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# ---------- F1 ----------
axes[0].hlines(
    y=df["Dataset"],
    xmin=0,
    xmax=df["F_diff"],
    color=colors_f,
    linewidth=2
)

axes[0].scatter(
    df["F_diff"],
    df["Dataset"],
    s=140,
    c=colors_f
)

axes[0].axvline(0, linestyle="--", linewidth=1.2)
axes[0].set_title("F-measure Difference")
axes[0].set_xlabel("GP-SMOTE - GP-SMOTE-woc")

# ---------- AUC ----------
axes[1].hlines(
    y=df["Dataset"],
    xmin=0,
    xmax=df["AUC_diff"],
    color=colors_auc,
    linewidth=2
)

axes[1].scatter(
    df["AUC_diff"],
    df["Dataset"],
    s=140,
    c=colors_auc
)

axes[1].axvline(0, linestyle="--", linewidth=1.2)
axes[1].set_title("AUC Difference")
axes[1].set_xlabel("GP-SMOTE - GP-SMOTE-woc")

plt.tight_layout()
plt.savefig("lollipop_difference_plot.pdf", bbox_inches="tight")
plt.show()