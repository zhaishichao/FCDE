"""
t-SNE 可视化脚本的公共模块。

包含 t-SNE 绘图底层函数 plot_tsne，以及两个编排函数：
  - plot_single_group：为每个数据集生成一张独立图（bs / dg / gp / mtgp）
  - plot_grid：将多个数据集绘制成一行多子图（woC 系列）
各脚本只需提供数据配置并调用对应函数即可。
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plot_tsne(X, y, show_legend=False, ax=None, title=None, save_path=None,
              tick_labelsize=18, legend_fontsize=18, legend_loc="upper left"):
    """t-SNE二维降维并绘制散点图。

    Parameters
    ----------
    X : ndarray, 特征矩阵
    y : ndarray, 标签（0/1=原始多数/少数类, 2=合成样本）
    show_legend : bool, 是否显示图例
    ax : matplotlib.axes.Axes or None, 指定子图
    title : str or None, 子图标题
    save_path : str or None, PDF保存路径（仅ax=None时生效）
    tick_labelsize : int, 刻度数字大小（默认 18）
    legend_fontsize : int, 图例文字大小（默认 18）
    legend_loc : str, 图例位置（默认 "upper left"）
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
            fontsize=legend_fontsize,  # 图例文字大小
            loc=legend_loc,            # 图例位置：upper left / upper right / lower left / lower right
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
        ax.tick_params(axis='both', labelsize=tick_labelsize, colors='black')
        ax.grid(False)
        if title is not None:
            ax.set_title(title, fontsize=30, fontweight='bold', pad=15)  # pad：标题与子图间距
    else:
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(color="black", fontsize=tick_labelsize)  # fontsize：刻度数字大小
        plt.yticks(color="black", fontsize=tick_labelsize)
        plt.grid(False)       # 去掉网格线
        plt.tight_layout()    # 自动收紧边距，避免标签溢出裁切
        if save_path is not None:
            plt.savefig(save_path, dpi=300)  # dpi：分辨率，越大越清晰但文件体积越大
        plt.show()


def plot_single_group(dataset_names, titles, file_names, root_path, save_dir,
                      suffix, legend_rule):
    """为每个数据集生成一张独立的 t-SNE 散点图。

    Parameters
    ----------
    dataset_names / titles / file_names : list
        三个等长列表，逐项一一对应（数据集名 / 图标题 / CSV 文件名）。
    root_path : str
        CSV 所在目录。
    save_dir : str
        图片保存目录。
    suffix : str
        保存文件名后缀（如 "bs"、"dg"、"gp"）。
    legend_rule : callable
        legend_rule(title) -> bool，决定该子图是否显示图例。
    """
    os.makedirs(save_dir, exist_ok=True)
    for dataset_name, title, file_name in zip(dataset_names, titles, file_names):
        df = pd.read_csv(os.path.join(root_path, file_name))
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        save_name = f"{dataset_name}_{title.lower()}_{suffix}"
        plot_tsne(X, y, show_legend=legend_rule(title),
                  save_path=os.path.join(save_dir, save_name + ".pdf"))


def plot_grid(titles, file_names, root_path, save_dir, output_name,
              ncols, figsize, legend_loc):
    """将多个数据集绘制成一行多子图的 t-SNE 散点图。

    与 plot_single_group 不同，这里所有子图共享同一张图；图例仅显示在
    第一个子图，且字号比单图更大以适配多子图布局。

    Parameters
    ----------
    titles / file_names : list
        两个等长列表，逐项对应（图标题 / CSV 文件名）。
    output_name : str
        输出 PDF 文件名。
    ncols : int
        子图列数。
    figsize : tuple
        整图尺寸（英寸）。
    legend_loc : str
        图例位置（"lower right" / "upper right" 等）。
    """
    os.makedirs(save_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=1.2)
    fig, axes = plt.subplots(1, ncols, figsize=figsize)

    for idx, (title, file_name) in enumerate(zip(titles, file_names)):
        df = pd.read_csv(os.path.join(root_path, file_name))
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        plot_tsne(X, y, show_legend=(idx == 0), ax=axes[idx], title=title,
                  tick_labelsize=21, legend_fontsize=23, legend_loc=legend_loc)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, output_name), dpi=300)
    plt.show()
