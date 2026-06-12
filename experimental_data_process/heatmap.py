import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子，确保示例可复现
np.random.seed(42)

# 方法名和数据集名
methods = ["GP-SMOTE", "DG-SMOTE", "MTGP-SMOTE", "Blind-SMOTE"]
datasets = [f"D{i}" for i in range(1, 26)]

# 随机生成平均运行时间（单位：秒）
# 使用对数正态分布，使数据更像真实运行时间（有快有慢，差异明显）
base = np.random.lognormal(mean=1.0, sigma=1.2, size=(4, 25))

# 人为制造一些方法间的差异，例如 Blind-SMOTE 在某些数据集上偏慢
base[3, :] *= 1.8
base[0, 5:10] *= 0.6   # GP-SMOTE 在部分数据集上稍快

# 构建 DataFrame
df = pd.DataFrame(base, index=methods, columns=datasets)
df = df.round(3)   # 保留三位小数

# 绘制热力图
plt.figure(figsize=(16, 4))
ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues",
                 linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Avg. Runtime (s)'})
ax.set_title("Average Runtime Comparison (s) across 25 Datasets", fontsize=14, pad=12)
ax.set_xlabel("Dataset")
ax.set_ylabel("Method")
plt.tight_layout()
plt.show()