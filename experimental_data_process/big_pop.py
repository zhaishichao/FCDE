import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 示例数据
small_f1 = [...]
large_f1 = [...]

small_auc = [...]
large_auc = [...]

# 构造DataFrame
df_f1 = pd.DataFrame({
    'Population': ['Small'] * len(small_f1) +
                  ['Large'] * len(large_f1),
    'F1-score': small_f1 + large_f1
})

df_auc = pd.DataFrame({
    'Population': ['Small'] * len(small_auc) +
                  ['Large'] * len(large_auc),
    'AUC': small_auc + large_auc
})

plt.rcParams['font.family'] = 'Times New Roman'

fig, axes = plt.subplots(
    1, 2,
    figsize=(10,4)
)

# F1
sns.boxplot(
    data=df_f1,
    x='Population',
    y='F1-score',
    ax=axes[0],
    width=0.5
)

sns.stripplot(
    data=df_f1,
    x='Population',
    y='F1-score',
    ax=axes[0],
    color='black',
    size=3,
    alpha=0.5
)

axes[0].set_title('(a) F1-score', fontsize=14)
axes[0].set_xlabel('')
axes[0].set_ylabel('F1-score', fontsize=12)

# AUC
sns.boxplot(
    data=df_auc,
    x='Population',
    y='AUC',
    ax=axes[1],
    width=0.5
)

sns.stripplot(
    data=df_auc,
    x='Population',
    y='AUC',
    ax=axes[1],
    color='black',
    size=3,
    alpha=0.5
)

axes[1].set_title('(b) AUC', fontsize=14)
axes[1].set_xlabel('')
axes[1].set_ylabel('AUC', fontsize=12)

plt.tight_layout()
plt.show()