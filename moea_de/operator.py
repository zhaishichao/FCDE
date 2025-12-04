from typing import Union, List
from collections import Counter
import numpy as np
from deap import base, creator, tools, gp, algorithms

# 自定义受保护的除法
def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


# 对数据进行采样并计算采样样本的中心点
def sample_and_center(x, threshold=None):
    """
    对数据进行采样并计算采样样本的中心点

    参数:
    x: ndarray, 特征数据

    返回:
    center: ndarray, 采样样本的中心点（平均值）
    sampled_x: ndarray, 采样后的特征数据
    """
    # 获取数据总数量
    total_samples = len(x)

    # 生成随机采样数量n，范围在(total_samples/2, total_samples)之间
    n = np.random.randint(total_samples // 2 + 1, total_samples)
    if threshold is not None:
        n = threshold

    # 从原始数据中随机采样n个样本
    indices = np.random.choice(total_samples, size=n, replace=False)
    sampled_x = x[indices]

    # 计算采样样本的中心点（平均值）
    center = np.mean(sampled_x, axis=0)

    return center, sampled_x

def remove_duplicate_individuals(individuals):
    seen = set()
    result = []
    for ind in individuals:
        key = str(ind)
        if key not in seen:
            seen.add(key)
            result.append(ind)
    return result


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
    # print(f"少数类是: {minority_class}")

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


def cosine_angle(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if (cos.size > 1):
        print(cos)
        print(a, b)
    if cos > 1:
        cos = 1
    if cos < -1:
        cos = -1
    return np.degrees(np.arccos(cos))


# 更简洁的版本
def calculate_k_min_distances_mean(x, k):
    """
    紧凑版本的函数实现
    """
    center = np.mean(x, axis=0)
    distances = np.linalg.norm(x - center, axis=1)
    return np.mean(np.partition(distances, -k)[-k:])

def selTournament_cv(individuals, k):
    chosen = []
    while len(chosen) < k:
        aspirants = tools.selRandom(individuals, 2)  # 随机选择tournsize个个体
        # print(f'亲本1：', aspirants[0], '亲本2：', aspirants[1])
        if aspirants[0].fitness.cv == 0 and aspirants[1].fitness.cv > 0:
            chosen.append(aspirants[0])
        elif aspirants[0].fitness.cv > 0 and aspirants[1].fitness.cv == 0:
            chosen.append(aspirants[1])
        elif aspirants[0].fitness.cv > 0 and aspirants[1].fitness.cv > 0:
            if aspirants[0].fitness.cv <= aspirants[1].fitness.cv:
                chosen.append(aspirants[0])
            else:
                chosen.append(aspirants[1])
        else:
            chosen.append(aspirants[0])
        if len(chosen) > 1 and str(chosen[-1]) == str(chosen[-2]):
            chosen.pop()
    return chosen


def calculate_statistics_inndividuals(individuals):
    """
    计算individuals列表中各个属性的最大值

    参数:
    individuals -- Individual对象列表

    返回:
    tuple -- (max_a, max_b, max_c, max_d)
    """
    if not individuals:
        return None, None, None, None

    max_minimum_distance = max(-ind.minimum_distance for ind in individuals)
    if max_minimum_distance == 0:
        max_minimum_distance = 1
    max_maj_min_distance = max(-ind.fitness.values[0] for ind in individuals)
    max_min_center_distance = max(ind.min_center_distance for ind in individuals)
    max_cosine_angle = max((ind.cosine_angle - 90) for ind in individuals)

    # 以字典的形式返回
    return {'minimum_distance': max_minimum_distance,
            'maj_min_distance': max_maj_min_distance,
            'min_center_distance': max_min_center_distance,
            'cosine_angle': max_cosine_angle}


def calculate_mean_inndividuals_cv(individuals, thresholds):
    """
    计算individuals列表中各个属性的最大值

    参数:
    individuals -- Individual对象列表

    返回:
    tuple -- (max_a, max_b, max_c, max_d)
    """
    if not individuals:
        return None, None, None, None

    mean_maj_min_distance = sum(
        max(0, -ind.fitness.values[0] / thresholds['maj_min_distance']) for ind in individuals) / len(
        individuals)
    mean_min_center_distance = sum(
        max(0, ind.min_center_distance / thresholds['min_center_distance']) for ind in individuals) / len(individuals)
    mean_cosine_angle = sum(max(0, (ind.cosine_angle - 90) / thresholds['cosine_angle']) for ind in individuals) / len(
        individuals)

    # 以字典的形式返回
    return {
        'mean_maj_min_distance': mean_maj_min_distance,
        'mean_min_center_distance': mean_min_center_distance,
        'mean_cosine_angle': mean_cosine_angle}
