import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


# 使用t-SNE进行二维可视化
def tsne_visualization_binary(X, y, save_path, filename,
                              perplexity=30, random_state=42,
                              figsize=(10, 8), dpi=300):
    """
    使用t-SNE对二分类数据进行二维可视化并保存图片

    参数:
    ----------
    X : array-like, shape (n_samples, n_features)
        特征数据矩阵
    y : array-like, shape (n_samples,)
        标签数据，应该是二分类 [0, 1]
    save_path : str
        图片保存目录路径
    filename : str
        保存的文件名（不需要扩展名）
    perplexity : float, optional (default=30)
        t-SNE的困惑度参数
    random_state : int, optional (default=42)
        随机种子
    figsize : tuple, optional (default=(10, 8))
        图像大小
    dpi : int, optional (default=300)
        图像分辨率

    返回:
    -------
    X_tsne : array, shape (n_samples, 2)
        t-SNE降维后的数据
    """

    # 参数检查
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    # 确保保存目录存在
    os.makedirs(save_path, exist_ok=True)

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2,
                perplexity=perplexity,
                random_state=random_state,
                n_iter=1000)

    X_tsne = tsne.fit_transform(X)

    # 创建可视化
    plt.figure(figsize=figsize)

    # 为两个类别设置不同的颜色和样式

    # 为不同类别设置颜色和样式
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']  # 使用Set3色系
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # 多种标记
    # 如果没有提供类别名称，自动生成
    labels = [f'Class {i}' for i in unique_classes]

    # colors = ['red', 'green']  # 红色和青色
    # markers = ['o', 's']  # 圆形和方形
    # labels = ['Class 0', 'Class 1']

    for i, class_label in enumerate(unique_classes):
        mask = (y == class_label)
        plt.scatter(X_tsne[mask, 0],
                    X_tsne[mask, 1],
                    c=colors[i],
                    marker=markers[i],
                    label=labels[i],
                    alpha=0.7,
                    s=60,
                    edgecolors='white',
                    linewidth=0.5)

    # 美化图表
    plt.title('t-SNE 2D Visualization - Binary Classification',
              fontsize=18, fontweight='bold', pad=24)
    plt.xlabel('t-SNE Component 1', fontsize=18)
    plt.ylabel('t-SNE Component 2', fontsize=18)
    plt.legend(fontsize=14, framealpha=0.9)
    plt.grid(alpha=0.2)

    # 添加统计信息到图表
    stats_text = '\n'.join([f'{labels[i]}: {np.sum(y == cls)} samples'
                            for i, cls in enumerate(unique_classes)])

    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=14)

    plt.tight_layout()
    # 保存图片
    full_path = os.path.join(save_path, f"{filename}.png")
    plt.savefig(full_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    # 显示图片
    # plt.show()
    plt.close()  # 关闭图形，避免在内存中积累

    print(f"t-SNE可视化图片已保存至: {full_path}")

    return X_tsne


# 绘制收敛曲线
def curve_fitting(list_cv, file_path, filename):
    # 你的收敛曲线数据（这里用示例数据，请替换为你的实际数据）

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制曲线
    plt.plot(list_cv, 'b-o', linewidth=2, markersize=6,
             markerfacecolor='red', markeredgecolor='red', label='cv')

    # 设置图形属性
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('cv', fontsize=12)
    plt.title('Convergence curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 添加一些辅助线
    plt.axhline(y=min(list_cv), color='r', linestyle='--', alpha=0.5,
                label=f'最优值: {min(list_cv):.3f}')

    full_path = os.path.join(file_path, f"{filename}.png")
    # 保存图形
    plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
    # # 显示图形
    # plt.show()
