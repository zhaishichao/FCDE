import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def find_k_nearest_to_center(csv_path, k, distance_metric='euclidean', return_one_based=True):
    """
    计算标签0的特征中心（均值），然后分别找出距离该中心最近的k个标签1和标签2样本的索引。

    参数:
        csv_path: CSV文件路径
        k: 需要返回的最近邻数量
        distance_metric: 距离度量（如 'euclidean', 'cityblock', 'cosine'）
        return_one_based: True返回行号从1开始，False返回0-based数组索引

    返回:
        indices_label1: 距离中心最近的k个标签1样本的原始行号列表
        indices_label2: 距离中心最近的k个标签2样本的原始行号列表
    """
    # 1. 读取数据
    df = pd.read_csv(csv_path)
    label_col = df.columns[-1]  # 假设最后一列为标签列
    X = df.drop(columns=[label_col]).values
    y = df[label_col].values
    original_indices = np.arange(len(df)) + 1 if return_one_based else np.arange(len(df))

    # 2. 按标签分离
    mask0 = (y == 0)
    mask1 = (y == 1)
    mask2 = (y == 2)

    X0 = X[mask0]
    X1 = X[mask1]
    X2 = X[mask2]

    idx1 = original_indices[mask1]
    idx2 = original_indices[mask2]

    # 3. 检查标签0是否存在
    if len(X0) == 0:
        raise ValueError("没有标签为0的样本，无法计算中心")

    # 4. 计算标签0的中心（均值向量）
    center0 = np.mean(X0, axis=0).reshape(1, -1)  # shape (1, n_features)

    # 5. 计算中心到所有标签1样本的距离
    if len(X1) == 0:
        print("警告：没有标签为1的样本，返回空列表")
        indices_label1 = []
    else:
        dist1 = cdist(center0, X1, metric=distance_metric)[0]  # 一维数组，长度 = n1
        # 取距离最小的k个索引（如果k>n1则取全部）
        k1 = min(k, len(dist1))
        nearest_idx = np.argsort(dist1)[:k1]
        indices_label1 = idx1[nearest_idx].tolist()

    # 6. 计算中心到所有标签2样本的距离
    if len(X2) == 0:
        print("警告：没有标签为2的样本，返回空列表")
        indices_label2 = []
    else:
        dist2 = cdist(center0, X2, metric=distance_metric)[0]
        k2 = min(k, len(dist2))
        nearest_idx = np.argsort(dist2)[:k2]
        indices_label2 = idx2[nearest_idx].tolist()

    return indices_label1, indices_label2



# 使用示例
if __name__ == "__main__":
    indices_1, indices_2 = find_k_nearest_to_center("C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\tsne\\woC\\wisconsin_5_X_train_res_dg.csv", k=5)
    print("标签1的最小距离索引（行号）:", indices_1)
    print("标签2的最小距离索引（行号）:", indices_2)