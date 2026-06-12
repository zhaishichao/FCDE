import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os


# ============================================================
# 0. 加载数据
# ============================================================
data_dir = os.path.join(os.path.dirname(__file__), "results")

# big_pop: 大规模种群运行结果
df_big = pd.read_csv(os.path.join(data_dir, "mean_big_pop.csv"))
# small_pop: 小规模种群运行结果
df_small = pd.read_csv(os.path.join(data_dir, "mean_small_pop.csv"))

# 按数据集名称合并，确保一一对应
df = pd.merge(df_big, df_small, on="数据集", suffixes=("_big", "_small"))

# 提取对比数据，并 ×100 转为百分制
f1_big = df["F-measure_big"].values * 100
f1_small = df["F-measure_small"].values * 100
auc_big = df["AUC_big"].values * 100
auc_small = df["AUC_small"].values * 100

# 数据集标签 (D1~D25)
labels = [f"D{i+1}" for i in range(len(df))]

print(f"加载 {len(df)} 个数据集")
print(f"F1 均值 — Big: {f1_big.mean():.2f}, Small: {f1_small.mean():.2f}")
print(f"AUC 均值 — Big: {auc_big.mean():.2f}, Small: {auc_small.mean():.2f}")


# ============================================================
# 1. 全局绘图参数
# ============================================================

# --- 字体设置（Times New Roman）---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

# --- 图形整体尺寸（英寸）---
FIG_WIDTH = 11
FIG_HEIGHT = 5.5


# --- 标题 / 坐标轴字号 ---
TITLE_SIZE = 15
SUPTITLE_SIZE = 18
AXIS_LABEL_SIZE = 15.5
TICK_SIZE = 14

# --- 散点参数 ---
SCATTER_SIZE = 95            # 点大小
SCATTER_EDGECOLOR = "white"  # 点边缘色
SCATTER_LINEWIDTH = 0.6      # 点边缘宽
SCATTER_ALPHA = 0.85         # 点透明度, 1=完全不透明, 0.3=很淡

# --- 三分类颜色 ---
COLOR_TIE = "#525252"              # 深灰: Large ≈ Small
COLOR_LARGE_BETTER = "#4B9CD3"     # 深蓝: Large > Small
COLOR_SMALL_BETTER = "#5CA05C"     # 深绿: Large < Small

# --- ≈ 阈值（百分制下，差值绝对值小于此值视为约等于）---
EPSILON = 0.5  # 0.5个百分点以内视为近似相等; 调大则更多点归为≈, 调小则更少

# --- 对角线参数 ---
DIAG_COLOR = "black"
DIAG_STYLE = "--"            # "-"实线, "--"虚线, "-."点划线, ":"点线
DIAG_WIDTH = 0.8
DIAG_ALPHA = 0.4

# --- 边距 ---
MARGIN_PCT = 0.05

# --- 顶部图例（颜色含义说明）---
LEGEND_SIZE = 14             # 图例字号
LEGEND_MARKER_SIZE = 10      # 图例点大小 (数值越大点越大)

# --- 图内统计标注（计数）---
STAT_LEGEND_SIZE = 15         # 统计字号
STAT_MARKER_SIZE = 9         # 统计点大小

# --- 输出 ---
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "scatter_comparison.pdf")
DPI = 300



# ============================================================
# 2. 辅助函数：根据差值分配颜色
# ============================================================
def assign_colors(big_vals, small_vals, epsilon=EPSILON):
    """
    逐点比较大小，返回颜色列表
    epsilon: 差值绝对值阈值, |big−small|<=epsilon → ≈ (近似相等)
    """
    colors = []
    for b, s in zip(big_vals, small_vals):
        diff = b - s
        if diff > epsilon:
            colors.append(COLOR_LARGE_BETTER)      # Large 显著更高
        elif diff < -epsilon:
            colors.append(COLOR_SMALL_BETTER)      # Small 显著更高
        else:
            colors.append(COLOR_TIE)               # 近似相等
    return colors


# ============================================================
# 3. 绘图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT),
                        gridspec_kw={"wspace": 0.25})  # wspace: 子图间距, 0=紧贴, 越大间距越大

for ax, metric_name, big_vals, small_vals in [
    (axes[0], "F1-score", f1_big, f1_small),
    (axes[1], "AUC", auc_big, auc_small),
]:
    # ----------------------------------------------------------
    # 3a. 坐标轴范围
    # ----------------------------------------------------------
    all_vals = np.concatenate([big_vals, small_vals])
    val_min = all_vals.min()
    val_max = all_vals.max()
    margin = (val_max - val_min) * MARGIN_PCT
    ax_min = val_min - margin
    ax_max = val_max + margin

    # ----------------------------------------------------------
    # 3b. 对角线 y = x
    # ----------------------------------------------------------
    ax.plot([ax_min, ax_max], [ax_min, ax_max],
            color=DIAG_COLOR, linestyle=DIAG_STYLE,
            linewidth=DIAG_WIDTH, alpha=DIAG_ALPHA, zorder=1)

    # ----------------------------------------------------------
    # 3c. 分配颜色并绘制散点
    #     按类别分层绘制，确保图例正确
    # ----------------------------------------------------------
    point_colors = assign_colors(big_vals, small_vals)

    # 分层绘制（≈ 在底层，避免覆盖显眼的绿/红点）
    for cat_color, cat_label in [
        (COLOR_TIE, "Large ≈ Small"),
        (COLOR_LARGE_BETTER, "Large > Small"),
        (COLOR_SMALL_BETTER, "Large < Small"),
    ]:
        mask = [c == cat_color for c in point_colors]
        if not any(mask):
            continue
        ax.scatter(
            small_vals[mask], big_vals[mask],
            s=SCATTER_SIZE,                        # 点大小
            c=cat_color,                           # 填充色
            edgecolors=SCATTER_EDGECOLOR,          # 边缘色
            linewidth=SCATTER_LINEWIDTH,           # 边缘宽
            alpha=SCATTER_ALPHA,                   # 透明度
            label=cat_label,                       # 图例标签
            zorder=5,
        )

    # ----------------------------------------------------------
    # 3d. 坐标轴标签与样式
    # ----------------------------------------------------------
    ax.set_xlabel("Small Population (%)", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Large Population (%)", fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_title(metric_name, fontsize=TITLE_SIZE, fontweight="bold")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color("gray")

    # ----------------------------------------------------------
    # 3e. 图内统计图例（圆点 + 计数，≈ 排第一）
    # ----------------------------------------------------------
    n_large = np.sum(np.array(point_colors) == COLOR_LARGE_BETTER)
    n_small = np.sum(np.array(point_colors) == COLOR_SMALL_BETTER)
    n_tie = np.sum(np.array(point_colors) == COLOR_TIE)

    stat_order = [
        (COLOR_TIE, f"{n_tie}"),
        (COLOR_LARGE_BETTER, f"{n_large}"),
        (COLOR_SMALL_BETTER, f"{n_small}"),
    ]
    stat_handles = []
    for color, label in stat_order:
        stat_handles.append(
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color,
                   markeredgecolor=SCATTER_EDGECOLOR,
                   markeredgewidth=SCATTER_LINEWIDTH,
                   markersize=STAT_MARKER_SIZE,
                   linestyle="None")
        )

    ax.legend(
        handles=stat_handles,
        labels=[l for _, l in stat_order],
        loc="upper left",
        fontsize=STAT_LEGEND_SIZE,
        title="Counts",
        title_fontsize=STAT_LEGEND_SIZE,
        frameon=True,
        fancybox=True,
        framealpha=0.7,
        edgecolor="gray",
        borderpad=0.3,           # 边框内边距, 越小越紧凑
        handletextpad=0.2,       # 图标与文字间距
        labelspacing=0.2,        # 条目之间的垂直间距
    )

# ----------------------------------------------------------
# 3f. 顶部图例（颜色含义说明，无背景无边框，≈ 排第一）
# ----------------------------------------------------------
legend_order = [
    (COLOR_TIE, " Large ≈ Small"),
    (COLOR_LARGE_BETTER, " Large > Small"),
    (COLOR_SMALL_BETTER, " Large < Small"),
]
top_handles = []
for color, label in legend_order:
    top_handles.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=color,
               markeredgecolor=SCATTER_EDGECOLOR,
               markeredgewidth=SCATTER_LINEWIDTH,
               markersize=LEGEND_MARKER_SIZE,
               linestyle="None")
    )

fig.legend(
    handles=top_handles,
    labels=[l for _, l in legend_order],
    loc="upper center",
    ncol=3,
    fontsize=LEGEND_SIZE,
    frameon=False,
    columnspacing=0.2,         # 列间距, 越小越紧凑
    handletextpad=0.02,          # 图标与文字间距, 越小越紧凑
    bbox_to_anchor=(0.5, 0.95),
)

# ----------------------------------------------------------
# 3g. 总标题
# ----------------------------------------------------------
fig.suptitle("GP-SMOTE: Large vs Small Population Size",
             fontsize=SUPTITLE_SIZE, fontweight="bold", y=1.005)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
print(f"\n图像已保存至: {OUTPUT_PATH}")
plt.show()
