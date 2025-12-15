from typing import Union, List
from collections import Counter

import numpy as np


########### 数据预处理 ##########

# 分离多数类和少数类
def separate_maj_min(X, y):
    '''
    将原始数据的多数类和少数类分割开，返回值是一个数据字典
    '''
    # 找出两个类别以及各自数量
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) != 2:
        raise ValueError("数据集必须包含两个类别")

    # 判断哪个是少数类，哪个是多数类
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]

    maj_x = X[y == majority_class]
    maj_y = y[y == majority_class]

    min_x = X[y == minority_class]
    min_y = y[y == minority_class]

    return {
        'maj_x': maj_x,
        'maj_y': maj_y,
        'min_x': min_x,
        'min_y': min_y
    }


# 对数据进行采样并计算采样样本的中心点
def random_sampling(x):
    """
    对数据进行随机采样
    参数:
    x: ndarray, 特征数据
    返回:
    sampled_x: ndarray, 采样后的特征数据
    """
    # 生成随机采样数量n，范围在(total_samples/2, total_samples)之间
    n = np.random.randint(len(x) // 2 + 1, len(x))

    # 从原始数据x中随机采样n个样本
    indices = np.random.choice(len(x), size=n, replace=False)
    sampled_x = x[indices]

    return sampled_x


# 计算数据的中心
def calculate_center(x):
    return np.mean(x, axis=0)


# 计算平均最大距离
def calculate_ave_max_distance(X, k):
    """
    紧凑版本的函数实现
    """
    center = np.mean(X, axis=0)
    distances = np.linalg.norm(X - center, axis=1)
    return np.mean(np.partition(distances, -k)[-k:])

# 计算最短距离

def calculate_min_distance(X, new_instance):
    """
    计算new_instance与X中所有点的欧式距离中最小的距离值

    参数:
    X: numpy数组, shape=(n_samples, n_features)
    new_instance: numpy数组, shape=(n_features,)

    返回:
    min_distance: new_instance与X中所有点的最小欧式距离
    """
    distances = np.linalg.norm(X - new_instance, axis=1)
    return np.min(distances)


# 计算k个最短距离的实例中，少数类的占比
def minority_class_proportion(x: Union[List, np.ndarray],
                              y: Union[List, np.ndarray],
                              x_new: Union[List, np.ndarray],
                              k: int) -> float:
    """
    计算新实例在k近邻中少数类所占的比例

    参数:
    x: 特征数据，形状为(n_samples, n_features)
    y: 标签数据，形状为(n_samples,)，一般为2分类（0或1）
    x_new: 新实例的特征数据，形状为(n_features,)
    k: 近邻数量

    返回:
    float: 少数类在k近邻中所占的比例
    """

    # 转换为numpy数组以便处理
    x = np.array(x)
    y = np.array(y)
    x_new = np.array(x_new)

    # 参数检查
    if len(x) != len(y):
        raise ValueError("特征数据x和标签y的长度必须相同")

    if len(x) == 0:
        raise ValueError("输入数据不能为空")

    if k <= 0 or k > len(x):
        raise ValueError(f"k值必须在1到{len(x)}之间")

    # 确定少数类
    class_counts = Counter(y)
    if len(class_counts) != 2:
        raise ValueError("目前只支持2分类问题")

    minority_class = min(class_counts, key=class_counts.get)

    # 计算欧式距离
    distances = []
    for i, sample in enumerate(x):
        # 计算x_new与每个样本之间的欧式距离
        distance = np.sqrt(np.sum((sample - x_new) ** 2))
        distances.append((distance, y[i]))

    # 根据距离排序
    distances.sort(key=lambda x: x[0])

    # 获取前k个最近邻的标签
    k_nearest_labels = [label for _, label in distances[:k]]

    # 计算少数类的比例
    minority_count = k_nearest_labels.count(minority_class)
    proportion = minority_count / k

    return proportion, distances[0][0]


def calculate_cosine_angle(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return np.degrees(np.arccos(-1))
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if cos > 1:
        cos = 1
    if cos < -1:
        cos = -1

    return np.degrees(np.arccos(cos))
