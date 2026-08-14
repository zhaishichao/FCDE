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
sort_by_difference = True

# --- 标题 ---
main_title = 'Lollipop Chart Analysis'     # 主图（整体）标题
left_title = 'F1-Score: Lollipop Chart'    # 左图标题
right_title = 'AUC: Lollipop Chart'        # 右图标题

# --- 坐标轴标签 ---
xlabel = 'Performance Difference (Constrained - Unconstrained) / %'
ylabel = 'Dataset'

# --- 配色 ---
color_positive = '#1f77b4'   # 正差值（约束更好）
color_negative = '#d62728'   # 负差值（非约束更好）

# --- 图例 ---
show_legend = True
legend_labels = ['Positive (Constrained Better)', 'Negative (Unconstrained Better)']

# --- 棒棒糖样式 ---
lollipop_linewidth = 2        # 棒（线）宽
lollipop_size = 120           # 糖（圆点）大小
lollipop_edgecolor = 'black'  # 圆点边缘色
lollipop_edgewidth = 1        # 圆点边缘宽

# --- 图形尺寸（英寸）---
fig_width = 16
fig_height = 8

# --- 输出 ---
output_filename = '../lollipop_chart_comparison.pdf'  # 保存文件名

# ============================================================================
# 字体设置
# ============================================================================
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['font.family'] = 'serif'

# 计算差值并创建数据框
f1_diff = np.array(cons_f) - np.array(uncons_f)
auc_diff = np.array(cons_auc) - np.array(uncons_auc)

df_f1 = pd.DataFrame({'Dataset': datasets, 'Difference': f1_diff})
df_auc = pd.DataFrame({'Dataset': datasets, 'Difference': auc_diff})

# 根据排序选项处理数据
sort_label = "Sorted by Dataset"
if sort_by_difference:
    df_f1 = df_f1.sort_values('Difference').reset_index(drop=True)
    df_auc = df_auc.sort_values('Difference').reset_index(drop=True)
    sort_label = "Sorted by Difference"


def plot_lollipop(ax, df, title):
    """在指定坐标轴上绘制棒棒糖图。"""
    y_pos = np.arange(len(df))
    colors = [color_negative if x < 0 else color_positive for x in df['Difference']]

    for i, (diff, color) in enumerate(zip(df['Difference'], colors)):
        ax.plot([0, diff], [i, i], color=color, linewidth=lollipop_linewidth, zorder=1)
        ax.scatter(diff, i, color=color, s=lollipop_size, zorder=2,
                   edgecolor=lollipop_edgecolor, linewidth=lollipop_edgewidth)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Dataset'], fontsize=9, family='serif')
    ax.set_xlabel(xlabel, fontsize=11, family='serif', fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, family='serif', fontweight='bold')
    ax.set_title(title, fontsize=12, family='serif', fontweight='bold', pad=15)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')


# 创建并排的两张图
fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
plot_lollipop(axes[0], df_f1, left_title)
plot_lollipop(axes[1], df_auc, right_title)

# 添加共同的图例
if show_legend:
    legend_elements = [
        Patch(facecolor=color_positive, edgecolor='black', label=legend_labels[0]),
        Patch(facecolor=color_negative, edgecolor='black', label=legend_labels[1]),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10,
               frameon=True, fancybox=False, edgecolor='black', bbox_to_anchor=(0.5, 0.96))

# 整体标题
fig.suptitle(f'{main_title} ({sort_label})',
             fontsize=14, family='serif', fontweight='bold', y=0.995)

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"棒棒糖图已保存 ({sort_label})")
print(f"文件名：{output_path}")
plt.close()
