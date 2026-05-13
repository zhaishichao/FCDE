import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_tsne(X, y, show_legend=False, ax=None, title=None, save_path=None):
    """t-SNE二维降维并绘制散点图。

    Parameters
    ----------
    X : ndarray, 特征矩阵
    y : ndarray, 标签（0/1=原始多数/少数类, 2=合成样本）
    show_legend : bool, 是否显示图例
    ax : matplotlib.axes.Axes or None, 指定子图
    title : str or None, 子图标题
    save_path : str or None, PDF保存路径（仅ax=None时生效）
    """
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
        learning_rate=200,   # 优化步长：越大收敛越快但可能不稳定，越小越慢但更精细
        max_iter=1000,       # 最大迭代次数
        random_state=42
    ).fit_transform(X)

    plot_df = pd.DataFrame({
        "Dim1": X_tsne[:, 0],
        "Dim2": X_tsne[:, 1],
        "Class": [class_names[label] for label in y]
    })

    palette = {
        "Majority class": "#3FADFF",
        "Minority class": "#0DC0C9",
        "Synthetic instance": "#FD8251"
    }
    markers = {"Majority class": "o", "Minority class": "o", "Synthetic instance": "o"}

    if ax is None:
        sns.set_theme(style="whitegrid", font="Times New Roman",
                      font_scale=1.2)  # font_scale：>1放大全局字体，<1缩小
        plt.figure(figsize=(6, 6))     # figsize：英寸，越大图片尺寸越大

    sns.scatterplot(
        data=plot_df, x="Dim1", y="Dim2", hue="Class",
        palette=palette, style="Class", markers=markers,
        s=36,               # 点的大小：越大点越大，越小点越小
        alpha=1.0,          # 透明度：0=全透明（不可见），1=完全不透明（无遮挡穿透）
        edgecolor="none",   # 无边框，设为具体颜色可给每个点加描边
        legend=show_legend,
        ax=ax
    )

    if show_legend:
        legend_kwargs = dict(
            fontsize=18,           # 图例文字大小
            loc="upper left",      # 图例位置：upper left / upper right / lower left / lower right
            markerscale=1.0,       # 图例中点的大小缩放：>1放大点，<1缩小点
            handletextpad=0.05,    # 点与文字间距：越大间距越大
            labelspacing=0.2,      # 行间距：越大行距越宽
            borderpad=0.15,        # 内边距：越大图例边框内留白越多
            frameon=True           # 是否显示图例边框
        )
        if ax is not None:
            ax.legend(**legend_kwargs)
        else:
            plt.legend(**legend_kwargs)

    if ax is not None:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='both', labelsize=18, colors='black')
        ax.grid(False)
        if title is not None:
            ax.set_title(title, fontsize=30, fontweight='bold', pad=15)  # pad：标题与子图间距
    else:
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(color="black", fontsize=18)  # fontsize：刻度数字大小
        plt.yticks(color="black", fontsize=18)
        plt.grid(False)       # 去掉网格线
        plt.tight_layout()    # 自动收紧边距，避免标签溢出裁切
        if save_path is not None:
            plt.savefig(save_path, dpi=300)  # dpi：分辨率，越大越清晰但文件体积越大
        plt.show()
