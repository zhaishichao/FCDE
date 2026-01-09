from operator import attrgetter


# 计算约束阈值
def calculate_constraint_thresholds(individuals, avg_distance=None):
    """
    计算individuals列表中各个属性的最大值

    参数:
    individuals -- Individual对象列表

    返回:
    tuple -- (max_a, max_b, max_c, max_d)
    """
    max_g1 = max(avg_distance - ind.distance_minority_min for ind in individuals)  # distance_minority_min > 0
    max_g2 = max(0 - ind.fitness.values[0] for ind in individuals)  # ind.fitness.values[0] (第一个目标) > 0
    max_g3 = max(ind.distance_minority_center for ind in individuals)  # distance_minority_center - ave_max_distance > 0
    max_g4 = max((ind.cosine_angle - 90) for ind in individuals)  # cosine_angle < 90

    # 以字典的形式返回
    return {'max_g1': max_g1,
            'max_g2': max_g2,
            'max_g3': max_g3,
            'max_g4': max_g4,
            'avg_distance': avg_distance}


# 计算约束违反程度 cv
def cv(ind, thresholds):
    cvs = []
    cvs.append(max(0, (thresholds['avg_distance'] - ind.distance_minority_min) / thresholds['max_g1']))
    cvs.append(max(0, (0 - ind.fitness.values[0]) / thresholds['max_g2']))
    cvs.append(max(0, ind.distance_minority_center / thresholds['max_g3']))
    cvs.append(max(0, (ind.cosine_angle - 90) / thresholds['max_g4']))
    cv = sum(cvs) / 4  # 求0和cv中的最小值之和，cv=0，表示是一个可行个体
    ind.fitness.cv = cv  # 将cv值保存在个体中
    return cv


# 分离可行解与不可行解
def get_feasible_infeasible(pop, thresholds):
    '''
    :param pop: 种群
    :param constraints: 约束阈值
    :return: 可行解和不可行解
    '''
    index = []
    for i in range(len(pop)):
        if cv(pop[i], thresholds) == 0:  # 判断个体适应度是否都满足约束条件
            index.append(i)  # 将不符合约束条件的个体的索引添加到index中
    feasibles = [ind for j, ind in enumerate(pop) if j in index]  # 可行个体
    infeasibles = [ind for j, ind in enumerate(pop) if j not in index]  # 不可行个体
    infeasibles = sorted(infeasibles, key=attrgetter("fitness.cv"), reverse=False)  # 对不可行个体按cv值升序排列
    return feasibles, infeasibles
