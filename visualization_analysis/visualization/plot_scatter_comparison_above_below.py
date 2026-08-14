import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os


# ============================================================
# 0. 加载数据
# ============================================================
data_dir = os.path.join(os.path.dirname(__file__), "..", "raw_data")

df_big = pd.read_csv(os.path.join(data_dir, "mean_big_pop.csv"))      # 大规模种群运行结果
df_small = pd.read_csv(os.path.join(data_dir, "mean_small_pop.csv"))  # 小规模种群运行结果

# 按数据集名称合并，确保一一对应
df = pd.merge(df_big, df_small, on="数据集", suffixes=("_big", "_small"))

# 提取对比数据，并 ×100 转为百分制
f1_big = df["F-measure_big"].values * 100
f1_small = df["F-measure_small"].values * 100
auc_big = df["AUC_big"].values * 100
auc_small = df["AUC_small"].values * 100

print(f"加载 {len(df)} 个数据集")
print(f"F1 均值 — Big: {f1_big.mean():.2f}, Small: {f1_small.mean():.2f}")
print(f"AUC 均值 — Big: {auc_big.mean():.2f}, Small: {auc_small.mean():.2f}")


# ============================================================
# 1. 全局绘图参数（集中在此设置，便于统一修改）
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

# --- 标题 ---
MAIN_TITLE = "GP-SMOTE: Large vs Small Population Size"   # 总标题（整体）
LEFT_TITLE = "F1-Score"                                   # 左图标题
RIGHT_TITLE = "AUC"                                       # 右图标题

# --- 坐标轴标签 ---
XLABEL = "Small Population (%)"     # 横轴标签
YLABEL = "Large Population (%)"     # 纵轴标签

# --- 图例 ---
SHOW_LEGEND = True                  # 是否绘制顶部图例（颜色含义）
LEGEND_LABELS = ["Large ≈ Small", "Large > Small", "Large < Small"]  # 顶部图例内容（与 CATEGORY_COLORS 顺序对应）

# --- 散点参数 ---
SCATTER_SIZE = 105            # 点大小
SCATTER_EDGECOLOR = "white"  # 点边缘色
SCATTER_LINEWIDTH = 0.6      # 点边缘宽
SCATTER_ALPHA = 0.85         # 点透明度, 1=完全不透明, 0.3=很淡

# --- 三分类颜色（取消注释即可切换，当前使用最后一组）---
# COLOR_TIE = "#525252"              # 深灰: Large ≈ Small
# COLOR_LARGE_BETTER = "#4B9CD3"     # 深蓝: Large > Small
# COLOR_SMALL_BETTER = "#5CA05C"     # 深绿: Large < Small
#
# COLOR_TIE = "#2196F3"              # 深灰: Large ≈ Small
# COLOR_LARGE_BETTER = "#00BCD4"     # 深蓝: Large > Small
# COLOR_SMALL_BETTER = "#FF9800"     # 深绿: Large < Small

COLOR_TIE = "#1E88E5"              # 深灰: Large ≈ Small
COLOR_LARGE_BETTER = "#00ACC1"     # 深蓝: Large > Small
COLOR_SMALL_BETTER = "#EF6C00"     # 深绿: Large < Small

# 三类点的颜色（顺序固定：≈、Large 更优、Small 更优，与 LEGEND_LABELS 对应）
CATEGORY_COLORS = [COLOR_TIE, COLOR_LARGE_BETTER, COLOR_SMALL_BETTER]

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

# --- 图内三类点计数标注（与顶部图例一致）---
SHOW_COUNTS = True               # 是否在子图内标注各类点的计数
COUNTS_FONTSIZE = 11             # 标注字号
COUNTS_POS = (0.55, 0.05)        # 标注位置（axes 坐标，左下角，左对齐）

# --- 输出 ---
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "scatter_comparison_above_below.pdf")
DPI = 300


# ============================================================
# 2. 辅助函数
# ============================================================
def assign_colors(big_vals, small_vals, epsilon=EPSILON):
    """逐点比较大小，返回颜色数组。

    epsilon: 差值绝对值阈值, |big−small|<=epsilon → ≈ (近似相等)
    """
    diff = np.asarray(big_vals) - np.asarray(small_vals)
    return np.where(diff > epsilon, COLOR_LARGE_BETTER,
                    np.where(diff < -epsilon, COLOR_SMALL_BETTER, COLOR_TIE))


def legend_handle(color, markersize):
    """构造一个圆点图例句柄。"""
    return Line2D([0], [0], marker="o", color="w",
                  markerfacecolor=color,
                  markeredgecolor=SCATTER_EDGECOLOR,
                  markeredgewidth=SCATTER_LINEWIDTH,
                  markersize=markersize,
                  linestyle="None")


# ============================================================
# 3. 绘图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT),
                        gridspec_kw={"wspace": 0.25})  # wspace: 子图间距, 0=紧贴, 越大间距越大

for ax, metric_name, big_vals, small_vals in [
    (axes[0], LEFT_TITLE, f1_big, f1_small),
    (axes[1], RIGHT_TITLE, auc_big, auc_small),
]:
    # 3a. 坐标轴范围
    all_vals = np.concatenate([big_vals, small_vals])
    val_min, val_max = all_vals.min(), all_vals.max()
    margin = (val_max - val_min) * MARGIN_PCT
    ax_min, ax_max = val_min - margin, val_max + margin

    # 3b. 对角线 y = x
    ax.plot([ax_min, ax_max], [ax_min, ax_max],
            color=DIAG_COLOR, linestyle=DIAG_STYLE,
            linewidth=DIAG_WIDTH, alpha=DIAG_ALPHA, zorder=1)

    # 3c. 分配颜色并绘制散点（≈ 在底层，避免覆盖显眼的绿/红点）
    point_colors = assign_colors(big_vals, small_vals)
    for cat_color in CATEGORY_COLORS:
        mask = point_colors == cat_color
        if not mask.any():
            continue
        ax.scatter(
            small_vals[mask], big_vals[mask],
            s=SCATTER_SIZE,                        # 点大小
            c=cat_color,                           # 填充色
            edgecolors=SCATTER_EDGECOLOR,          # 边缘色
            linewidth=SCATTER_LINEWIDTH,           # 边缘宽
            alpha=SCATTER_ALPHA,                   # 透明度
            zorder=5,
        )

    # 3d. 坐标轴标签与样式
    ax.set_xlabel(XLABEL, fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_ylabel(YLABEL, fontsize=AXIS_LABEL_SIZE, fontweight="bold")
    ax.set_title(metric_name, fontsize=TITLE_SIZE, fontweight="bold")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color("gray")

    # 3e. 标注三类点的计数（与顶部图例一致）
    if SHOW_COUNTS:
        counts = [int((point_colors == c).sum()) for c in CATEGORY_COLORS]
        text = "\n".join(f"{label}: {count}" for label, count in zip(LEGEND_LABELS, counts))
        ax.text(COUNTS_POS[0], COUNTS_POS[1], text,
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=COUNTS_FONTSIZE,
                bbox=dict(boxstyle="round", facecolor="none", edgecolor="black"))

# ----------------------------------------------------------
# 3f. 顶部图例（颜色含义说明，无背景无边框，≈ 排第一）
# ----------------------------------------------------------
if SHOW_LEGEND:
    legend_order = list(zip(CATEGORY_COLORS, [f" {l}" for l in LEGEND_LABELS]))  # 文字前加空格，与原样式一致
    fig.legend(
        handles=[legend_handle(c, LEGEND_MARKER_SIZE) for c, _ in legend_order],
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
fig.suptitle(MAIN_TITLE, fontsize=SUPTITLE_SIZE, fontweight="bold", y=1.005)

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
print(f"\n图像已保存至: {OUTPUT_PATH}")
plt.show()
