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
