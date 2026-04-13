import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# =========================
# 1. 读取 CSV 数据
# =========================
title = 'D27'
dataset_name = 'shuttle_' + title.lower() + '_gp'
root_path = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\tsne\\"
file_name = "shuttle-2-vs-5_1_X_train_res_ds.csv"
file_path = os.path.join(root_path, file_name)  # ← 替换为你的文件路径
df = pd.read_csv(file_path)

# =========================
# 2. 分离特征和标签 判断多数类 / 少数类（仅针对0和1）
# =========================
# 假设最后一列是标签
X = df.iloc[:, :-1].values  # 所有特征
y = df.iloc[:, -1].values  # 标签（0,1,2）
unique, counts = np.unique(y[y != 2], return_counts=True)
class_counts = dict(zip(unique, counts))
majority_class = max(class_counts, key=class_counts.get)
minority_class = min(class_counts, key=class_counts.get)

# =========================
# 4. t-SNE 降维
# =========================
tsne = TSNE(
    n_components=2,  # 降到二维（用于可视化）
    perplexity=30,  # 邻域大小（影响局部/全局结构）
    learning_rate=200,  # 学习率
    max_iter=1000,  # 迭代次数
    random_state=42  # 固定随机性（保证复现）
)

X_tsne = tsne.fit_transform(X)

# =========================
# 5. 构建用于绘图的 DataFrame
# =========================
plot_df = pd.DataFrame({
    "Dim1": X_tsne[:, 0],
    "Dim2": X_tsne[:, 1],
    "Label": y
})

# 映射标签名称（更适合论文展示）
label_map = {
    majority_class: "Majority class",
    minority_class: "Minority class",
    2: "Synthetic instance"
}
plot_df["Class"] = plot_df["Label"].map(label_map)

# =========================
# 6. 设置论文级绘图风格
# =========================
sns.set_theme(
    style="whitegrid",  # 背景风格（white / dark / whitegrid / darkgrid）
    font="Times New Roman",
    font_scale=1.2
)
plt.figure(figsize=(6, 6))

# =========================
# 7. 使用 seaborn 绘制 t-SNE 分布图
# =========================
sns.scatterplot(
    data=plot_df,  # 数据源（DataFrame）
    x="Dim1",  # x轴变量（t-SNE 第一维）
    y="Dim2",  # y轴变量（t-SNE 第二维）
    hue="Class",  # 按类别上色（最关键参数之一）
    # 👉 hue 会根据 Class 自动分组并赋予不同颜色
    palette={
        label_map[majority_class]: "#d62728",  # 蓝色
        label_map[minority_class]: "#1f77b4",  # 红色
        label_map[2]: "#2ca02c"  # 绿色
    },
    # 👉 palette：控制不同类别的颜色（可以是字典或调色板名）
    style="Class",
    # 👉 style：不同类别使用不同 marker（这里其实都是圆，可以去掉）
    markers={
        label_map[majority_class]: "o",
        label_map[minority_class]: "o",
        label_map[2]: "o"
    },
    # 👉 markers：指定不同类别的点形状（o=圆形）
    s=25,
    # 👉 s：点的大小（scalar 或 array）
    alpha=0.8,
    # 👉 alpha：透明度（0~1），用于减少遮挡
    edgecolor="none",
    # 👉 edgecolor：点的边框颜色（none 更干净）
    legend=False
)

# =========================
# 8. 图像美化（论文级）
# =========================
plt.title(title, fontweight="bold", fontsize=24)
plt.xlabel("")
plt.ylabel("")
# plt.xlabel("t-SNE Dimension 1", color="black", fontsize=16)
# plt.ylabel("t-SNE Dimension 2", color="black", fontsize=16)
plt.xticks([], color="black")
plt.yticks([], color="black")

plt.grid(False)
plt.tight_layout()

# =========================
# 9. 保存为高质量 PDF
# =========================
plt.savefig("../results/" + dataset_name + ".pdf", dpi=300)
plt.show()
