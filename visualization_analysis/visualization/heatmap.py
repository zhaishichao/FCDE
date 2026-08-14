import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


# ============================================================
# 路径与数据（读取数据并赋值）
# ============================================================
DATA_FILE = "average_runtime.csv"   # 数据文件名
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")   # 数据目录
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heatmap_runtime.pdf")   # 保存路径

df = pd.read_csv(os.path.join(RAW_DATA_DIR, DATA_FILE), encoding="utf-8-sig")
df = df.set_index("数据集").T   # 转置为「方法 × 数据集」矩阵


# ============================================================
# 绘图配置（集中在此设置，便于统一修改）
# ============================================================

# --- 字体设置（Times New Roman）---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

# --- 方法顺序（行），留空 [] 表示按 CSV 原始列序 ---
METHODS = ["DG-SMOTE", "MTGP-SMOTE", "Blind-SMOTE", "GP-SMOTE"]

# --- 标题 / 标签 ---
TITLE = "Average Runtime Comparison (s) across 25 Datasets"
XLABEL = "Dataset"
YLABEL = "Method"
CBAR_LABEL = "Avg. Runtime (s)"

# --- 颜色与外观 ---
CMAP = "Blues"
USE_LOG_SCALE = True           # 运行时间跨度大，用对数色阶更易区分（False 为线性）
ANNOT_FMT = ".1f"              # 单元格内数值格式
LINE_WIDTH = 0.5               # 网格线宽
LINE_COLOR = "gray"            # 网格线颜色

# --- 图形尺寸（英寸）---
FIG_WIDTH = 16
FIG_HEIGHT = 4


# 按指定顺序重排方法（依赖上面的 METHODS）
if METHODS:
    df = df.loc[METHODS]


# ============================================================
# 绘制热力图
# ============================================================
norm = LogNorm() if USE_LOG_SCALE else None

plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
ax = sns.heatmap(
    df,
    annot=True,
    fmt=ANNOT_FMT,
    cmap=CMAP,
    norm=norm,
    linewidths=LINE_WIDTH,
    linecolor=LINE_COLOR,
    cbar_kws={"label": CBAR_LABEL, "ticks": [100, 1000]},   # 只保留 10² 和 10³ 两个刻度
)
ax.set_title(TITLE, fontsize=14, pad=12)
ax.set_xlabel(XLABEL)
ax.set_ylabel(YLABEL)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
print(f"热力图已保存至: {OUTPUT_PATH}")
