import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 数据
datasets = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10',
            'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20',
            'D21', 'D22', 'D23', 'D24', 'D25']
cons_f = [84.31, 82.37, 89.95, 96.08, 96.71, 69.29, 100.0, 73.68, 65.47, 83.75,
          66.13, 95.14, 86.44, 76.8, 94.52, 90.75, 98.12, 78.42, 84.57, 83.91,
          64.13, 99.31, 82.16, 50.85, 100.0]
uncons_f = [84.16, 82.27, 89.71, 96.07, 96.66, 69.01, 100.0, 73.32, 65.3, 82.78,
            65.89, 95.1, 86.17, 76.36, 94.42, 90.47, 97.99, 77.57, 84.69, 84.22,
            63.4, 97.77, 82.24, 51.35, 99.24]
cons_auc = [90.13, 88.89, 95.1, 98.39, 98.99, 76.73, 100.0, 83.38, 72.21, 92.13,
            73.92, 99.12, 93.46, 79.51, 99.72, 95.25, 99.76, 83.86, 90.56, 97.35,
            78.84, 99.44, 79.7, 59.57, 100.0]
uncons_auc = [90.06, 88.61, 94.89, 98.39, 99.02, 76.54, 100.0, 82.76, 72.23, 91.77,
              73.9, 99.08, 93.28, 79.24, 99.72, 95.17, 99.72, 83.67, 90.16, 96.92,
              78.17, 98.33, 78.85, 60.13, 100.0]

# ============================================================================
# 设置参数：排序选项
# sort_by_difference = True  表示按差值大小排序
# sort_by_difference = False 表示按数据集顺序排列（D1, D2, ..., D25）
# ============================================================================
sort_by_difference = False

# 设置字体为New Times Roman
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
if sort_by_difference:
    df_f1 = df_f1.sort_values('Difference').reset_index(drop=True)
    df_auc = df_auc.sort_values('Difference').reset_index(drop=True)
    sort_label = "Sorted by Difference"
else:
    df_f1 = df_f1.reset_index(drop=True)
    df_auc = df_auc.reset_index(drop=True)
    sort_label = "Sorted by Dataset"

# 设置颜色
colors_f1 = ['#d62728' if x < 0 else '#1f77b4' for x in df_f1['Difference']]
colors_auc = ['#d62728' if x < 0 else '#1f77b4' for x in df_auc['Difference']]

# 创建并排的两张图
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# ============================================================================
# 绘制F1指标的配对差值图
# ============================================================================
ax1 = axes[0]
y_pos = np.arange(len(df_f1))

for i, (diff, color) in enumerate(zip(df_f1['Difference'], colors_f1)):
    ax1.barh(i, diff, color=color, height=0.7, edgecolor='black', linewidth=0.5)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(df_f1['Dataset'], fontsize=9, family='serif')
ax1.set_xlabel('Performance Difference (Constrained - Unconstrained) / %',
               fontsize=11, family='serif', fontweight='bold')
ax1.set_ylabel('Dataset', fontsize=11, family='serif', fontweight='bold')
ax1.set_title('F1-Score: Paired Difference Plot', fontsize=12, family='serif', fontweight='bold', pad=15)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# ============================================================================
# 绘制AUC指标的配对差值图
# ============================================================================
ax2 = axes[1]
y_pos = np.arange(len(df_auc))

for i, (diff, color) in enumerate(zip(df_auc['Difference'], colors_auc)):
    ax2.barh(i, diff, color=color, height=0.7, edgecolor='black', linewidth=0.5)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(df_auc['Dataset'], fontsize=9, family='serif')
ax2.set_xlabel('Performance Difference (Constrained - Unconstrained) / %',
               fontsize=11, family='serif', fontweight='bold')
ax2.set_ylabel('Dataset', fontsize=11, family='serif', fontweight='bold')
ax2.set_title('AUC: Paired Difference Plot', fontsize=12, family='serif', fontweight='bold', pad=15)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# 添加共同的图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#1f77b4', edgecolor='black', label='Positive (Constrained Better)'),
                   Patch(facecolor='#d62728', edgecolor='black', label='Negative (Unconstrained Better)')]
fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=10,
           frameon=True, fancybox=False, edgecolor='black', bbox_to_anchor=(0.5, 0.98))

# 整体标题
fig.suptitle(f'Paired Difference Analysis ({sort_label})',
             fontsize=14, family='serif', fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('./paired_difference_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"配对差值图已保存 ({sort_label})")
print(f"文件名：paired_difference_comparison.png")
plt.close()
