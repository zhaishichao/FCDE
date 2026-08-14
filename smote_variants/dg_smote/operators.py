"""DG-SMOTE 的自定义算子：受保护除法与锦标赛选择。"""

import numpy as np
from deap.tools import selRandom


def protectedDiv(left, right):
    """受保护除法：分母为 0 或结果出现 inf/nan 时返回 1。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def selTournament(individuals, k, tournsize=3):
    """自定义锦标赛选择：先取角度最大，角度相同时再取距离最大。"""
    chosen = []
    for _ in range(k):
        chose = []
        # 找到角度最大的个体
        while len(chose) == 0:
            aspirants = selRandom(individuals, tournsize)
            max_angle = 0
            for j in range(tournsize):
                if aspirants[j].fitness.values[0] > max_angle:
                    max_angle = aspirants[j].fitness.values[0]
            # 找到所有角度最大的个体
            for j in range(tournsize):
                if aspirants[j].fitness.values[0] == max_angle:
                    chose.append(aspirants[j])

        if len(chose) > 1:
            # 角度相同时，取距离最大的个体
            max_index = 0
            max_distance = 0
            for j in range(len(chose)):
                if chose[j].fitness.values[1] > max_distance:
                    max_index = j
                    max_distance = chose[j].fitness.values[1]
            chosen.append(chose[max_index])
        else:
            chosen.append(chose[0])

    return chosen
