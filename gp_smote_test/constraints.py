from operator import attrgetter

# 计算约束阈值
def calculate_constraint_thresholds(individuals):
    """
    计算individuals列表中各个属性的最大值

    参数:
    individuals -- Individual对象列表

    返回:
    tuple -- (max_a, max_b, max_c, max_d)
    """
    max_minimum_distance = max(-ind.minimum_distance for ind in individuals)
    if max_minimum_distance == 0: # 避免实例重合
        max_minimum_distance = 1
    max_maj_min_distance = max(-ind.fitness.values[0] for ind in individuals)
    max_min_center_distance = max(ind.min_center_distance for ind in individuals)
    max_cosine_angle = max((ind.cosine_angle - 90) for ind in individuals)

    # 以字典的形式返回
    return {'minimum_distance': max_minimum_distance,
            'maj_min_distance': max_maj_min_distance,
            'min_center_distance': max_min_center_distance,
            'cosine_angle': max_cosine_angle}

# 计算约束违反程度 cv
def cv(individual, thresholds):
    constraint_values = []
    constraint_values.append(max(0,  -individual.minimum_distance / thresholds['minimum_distance']))
    constraint_values.append(max(0, -individual.fitness.values[0] / thresholds['maj_min_distance']))
    constraint_values.append(max(0, (individual.cosine_angle - 90) / thresholds['cosine_angle']))
    constraint_values.append(max(0, individual.min_center_distance / thresholds['min_center_distance']))
    cv = sum(constraint_values) / 4  # 求0和cv中的最小值之和，cv=0，表示是一个可行个体
    individual.fitness.cv = cv  # 将cv值保存在个体中
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