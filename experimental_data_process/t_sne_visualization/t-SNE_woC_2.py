import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# ======== 数据配置 ========
titles = [r'w/o $g_{1}$ and $g_{2}$', r'w/o $g_{3}$']
file_names = ['wisconsin_9_X_train_res_border.csv',
              'wisconsin_2_X_train_res_re_g4.csv']
root_path = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\tsne\\woC\\"
save_dir = "./results/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ======== 绘图 ========
sns.set_theme(style="whitegrid", font="Times New Roman",
              font_scale=1.2)  # 缩放全局字体，>1放大 <1缩小
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1行3列，总尺寸18×6英寸

for idx, (title, file_name) in enumerate(zip(titles, file_names)):
    ax = axes[idx]
    df = pd.read_csv(os.path.join(root_path, file_name))

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 确定多数类 / 少数类（排除合成样本标签2）
    unique, counts = np.unique(y[y != 2], return_counts=True)
    class_counts = dict(zip(unique, counts))
    majority_class = max(class_counts, key=class_counts.get)
    minority_class = min(class_counts, key=class_counts.get)

    class_names = {
        majority_class: "Majority class",
        minority_class: "Minority class",
        2: "Synthetic instance"
    }

    # t-SNE降维到2维
    X_tsne = TSNE(
        n_components=2,
        perplexity=30,       # 邻域大小：越大越重全局结构，越小越重局部细节
        learning_rate=200,   # 优化步长：越大收敛越快但可能不稳定
        max_iter=1000,
        random_state=42
    ).fit_transform(X)

    plot_df = pd.DataFrame({
        "Dim1": X_tsne[:, 0],
        "Dim2": X_tsne[:, 1],
        "Class": [class_names[label] for label in y]
    })

    legend_flag = (idx == 0)  # 仅第一个子图显示图例

    sns.scatterplot(
        ax=ax, data=plot_df, x="Dim1", y="Dim2", hue="Class",
        palette={
            "Majority class": "#3FADFF",
            "Minority class": "#0DC0C9",
            "Synthetic instance": "#FD8251"
        },
        style="Class",
        markers={"Majority class": "o", "Minority class": "o", "Synthetic instance": "o"},
        s=36,               # 点的大小：越大点越大
        alpha=1.0,          # 透明度：1=完全不透明
        edgecolor="none",   # 无边框
        legend=legend_flag
    )

    ax.set_title(title, fontsize=30, fontweight='bold', pad=15)  # pad：标题与子图间距
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='both', labelsize=21, colors='black')  # labelsize：刻度数字大小
    ax.grid(False)

    if legend_flag:
        ax.legend(
            fontsize=23,           # 图例文字大小
            loc="lower right",     # 图例位置
            markerscale=1.0,       # 图例中点大小缩放：>1放大
            handletextpad=0.05,    # 点与文字间距：越大越远
            labelspacing=0.2,      # 行间距：越大越宽
            borderpad=0.15,        # 内边距：越大留白越多
            frameon=True           # 显示图例边框
        )

plt.tight_layout()  # 自动调整子图间距
plt.savefig(os.path.join(save_dir, "wisconsin_woc_2.pdf"), dpi=300)  # dpi：分辨率，越大越清晰文件越大
plt.show()
