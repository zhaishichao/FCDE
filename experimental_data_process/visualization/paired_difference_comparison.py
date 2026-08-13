import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================================
# 读取数据（从 raw_data/ 下的 CSV 读取并赋值）
# ============================================================================
data_file = 'constrained_unconstrained_1.csv'   # 数据文件名（位于 raw_data/ 下）
raw_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_data')
df_raw = pd.read_csv(os.path.join(raw_data_dir, data_file), encoding='utf-8-sig')

datasets = df_raw['数据集'].tolist()
cons_f = df_raw['cons_f'].tolist()
uncons_f = df_raw['uncons_f'].tolist()
cons_auc = df_raw['cons_auc'].tolist()
uncons_auc = df_raw['uncons_auc'].tolist()

# ============================================================================
# 绘图配置（集中在此设置，便于统一修改；留空 "" 或 None 表示不显示）
# ============================================================================

# --- 排序 ---
# sort_by_difference = True  表示按差值大小排序
# sort_by_difference = False 表示按数据集顺序排列（D1, D2, ..., D25）
sort_by_difference = False

# --- 标题 ---
main_title = 'Paired Difference Analysis'      # 主图（整体）标题
left_title = 'F1-Score'                        # 左图顶部标题
right_title = 'AUC'                            # 右图顶部标题

# --- 坐标轴标签 ---
left_xlabel = 'Constrained - Unconstrained (%)'    # 左图横轴（底部）标签
right_xlabel = 'Constrained - Unconstrained (%)'   # 右图横轴（底部）标签
left_ylabel = 'Dataset'                            # 左图纵轴标签
right_ylabel = 'Dataset'                           # 右图纵轴标签

# --- 图例 ---
show_legend = True                            # 是否绘制图例
legend_title = None                           # 图例标题（None 表示不显示）
legend_labels = ['Negative', 'Positive']      # 图例条目内容（与 colors 顺序对应）

# --- 配色方案（取消注释即可切换，当前使用最后一组） ---
# colors = ['#0DC0C9', '#FD8251']
# colors = ['#FD8251', '#0DC0C9']
# colors = ['#88CAAE', '#45A981']
# colors = ['#0DC0C9', '#A2E9F0']
# colors = ['#99E7EF', '#0DC0C9']
# colors = ['#6FB5E5', '#7ABD7A']
colors = ['#FFBAA7', '#EF7B5B']

# --- 输出 ---
output_filename = '../paired_difference_comparison.pdf'  # 保存文件名

# ============================================================================
# 字体设置
# ============================================================================
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['font.family'] = 'serif'

# 计算差值
f1_diff = np.array(cons_f) - np.array(uncons_f)
auc_diff = np.array(cons_auc) - np.array(uncons_auc)

# 创建数据框
df_f1 = pd.DataFrame({
    'Dataset': datasets,
    'Difference': f1_diff
})

df_auc = pd.DataFrame({
    'Dataset': datasets,
    'Difference': auc_diff
})

# 根据排序选项处理数据
sort_label = "Sorted by Dataset"
if sort_by_difference:
    df_f1 = df_f1.sort_values('Difference').reset_index(drop=True)
    df_auc = df_auc.sort_values('Difference').reset_index(drop=True)
    sort_label = "Sorted by Difference"


def plot_diff(ax, df, colors, title, xlabel, ylabel):
    """在指定坐标轴上绘制配对差值横向条形图。"""
    y_pos = np.arange(len(df))
    bar_colors = [colors[0] if x < 0 else colors[1] for x in df['Difference']]

    ax.barh(y_pos, df['Difference'], color=bar_colors, height=0.7, edgecolor='none')

    # x 轴刻度自适应（9 个刻度，覆盖差值范围）
    lo, hi = df['Difference'].min(), df['Difference'].max()
    margin = (hi - lo) * 0.1  # 两端各留 10% 边距
    xlim = (lo - margin, hi + margin)
    ticks = np.linspace(xlim[0], xlim[1], 9)

    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.2f}" for t in ticks])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Dataset'], fontsize=14)
    ax.set_xlabel(xlabel, fontsize=16, family='serif', fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.axvline(x=0, linestyle='-', color="black", linewidth=0.5)
    ax.grid(axis='x', alpha=0.5, linestyle='--')
    ax.tick_params(axis='both', labelsize=14)
    for spine in ax.spines.values():
        spine.set_color('gray')
    ax.set_xlim(xlim)


# 创建并排的两张图
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
plot_diff(axes[0], df_f1, colors, left_title, left_xlabel, left_ylabel)
plot_diff(axes[1], df_auc, colors, right_title, right_xlabel, right_ylabel)

# 添加共同的图例
if show_legend:
    legend_elements = [Patch(facecolor=colors[0], label=legend_labels[0]),
                       Patch(facecolor=colors[1], label=legend_labels[1])]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               title=legend_title, fontsize=15.5,
               frameon=True, fancybox=False, bbox_to_anchor=(0.5, 0.95))

# 整体标题
fig.suptitle(main_title, fontsize=19, family='serif', fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('./' + output_filename, dpi=300, bbox_inches='tight')
plt.show()
print(f"配对差值图已保存 ({sort_label})")
print(f"文件名：{output_filename}")
plt.close()
