import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 数据（填写你的真实结果）
# ===============================
datasets = [f"D{i}" for i in range(1, 26)]


rootpath = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\mean\\lw\\knn\\"
df_f = pd.read_csv(os.path.join(rootpath, "f1_1.csv"))  # GP-SMOTE F-measure
df_auc = pd.read_csv(os.path.join(rootpath, "auc_1.csv"))  # GP-SMOTE AUC
df_woc = pd.read_csv(os.path.join(rootpath, "woc.csv"))  # GP-SMOTE-woc

gp_f = list(df_f["GP-SMOTE"])  # GP-SMOTE F-measure
gp_auc = list(df_auc["GP-SMOTE"])  # GP-SMOTE-woc F-measure

woc_f = list(df_woc["F-measure"])  # GP-SMOTE AUC
woc_auc = list(df_woc["AUC"])  # GP-SMOTE-woc AUC

# ===============================
# 计算差值
# ===============================
diff_f = [a - b for a, b in zip(gp_f, woc_f)]
diff_auc = [a - b for a, b in zip(gp_auc, woc_auc)]

df = pd.DataFrame({
    "Dataset": datasets,
    "F-measure Diff": diff_f,
    "AUC Diff": diff_auc
})

# ===============================
# 风格设置
# ===============================
sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "pdf.fonttype": 42
})

# ===============================
# 绘图
# ===============================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ---------- F1 ----------
sns.scatterplot(
    data=df,
    x="Dataset",
    y="F-measure Diff",
    s=70,
    ax=axes[0]
)

axes[0].axhline(0, linestyle="--")
axes[0].set_title("F-measure Difference")
axes[0].set_ylabel("GP-SMOTE - GP-SMOTE-woc")
axes[0].tick_params(axis='x', rotation=60)

# ---------- AUC ----------
sns.scatterplot(
    data=df,
    x="Dataset",
    y="AUC Diff",
    s=70,
    ax=axes[1]
)

axes[1].axhline(0, linestyle="--")
axes[1].set_title("AUC Difference")
axes[1].set_ylabel("GP-SMOTE - GP-SMOTE-woc")
axes[1].tick_params(axis='x', rotation=60)

plt.tight_layout()
plt.savefig("paired_difference_plot.pdf", bbox_inches="tight")
plt.show()