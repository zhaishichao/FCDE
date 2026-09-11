"""约束处理：约束阈值计算、约束违反程度（cv）与可行/不可行解分离。

四个约束（约束 1~4）：
  约束1 : 新实例与所有少数类的最小距离应大于 avg_distance
  约束2 : 新实例离少数类中心的距离应小于 ave_max_distance
  约束3 : 新实例与多数类/少数类中心连线的夹角应小于 90°
  约束4 : 第一个目标（maj_min_dis）应大于 0
"""

from operator import attrgetter


def calculate_constraint_thresholds(individuals, avg_distance=None,
                                    remove_constraints=()):
    """计算种群中各个约束的最大违反程度（阈值）。

    remove_constraints: 要去掉的约束编号集合（1~4）。
    被去掉的约束不参与计算，返回的字典里也不包含对应阈值。
    """
    thresholds = {'avg_distance': avg_distance}

    if 1 not in remove_constraints:
        # 约束1：最小距离
        thresholds['max_g1'] = max(avg_distance - ind.distance_minority_min
                                   for ind in individuals)
    if 2 not in remove_constraints:
        # 约束2：中心距离
        thresholds['max_g2'] = max(ind.center_distance_excess
                                   for ind in individuals)
    if 3 not in remove_constraints:
        # 约束3：夹角
        thresholds['max_g3'] = max((ind.cosine_angle - 90)
                                   for ind in individuals)
    if 4 not in remove_constraints:
        # 约束4：第一目标
        thresholds['max_g4'] = max(0 - ind.fitness.values[0]
                                   for ind in individuals)
    return thresholds


def cv(ind, thresholds, remove_constraints=()):
    """计算个体约束违反程度（cv），并保存到 ind.fitness.cv。

    remove_constraints: 要去掉的约束编号集合（1~4）。
    若去掉全部 4 个约束，cv 直接返回 0（视为全部可行）。
    """
    if len(remove_constraints) >= 4:
        ind.fitness.cv = 0
        return 0

    cvs = []
    if 1 not in remove_constraints:
        # 约束1：最小距离
        cvs.append(max(0, (thresholds['avg_distance'] - ind.distance_minority_min)
                       / thresholds['max_g1']))
    if 2 not in remove_constraints:
        # 约束2：中心距离
        cvs.append(max(0, ind.center_distance_excess / thresholds['max_g2']))
    if 3 not in remove_constraints:
        # 约束3：夹角
        cvs.append(max(0, (ind.cosine_angle - 90) / thresholds['max_g3']))
    if 4 not in remove_constraints:
        # 约束4：第一目标
        cvs.append(max(0, (0 - ind.fitness.values[0]) / thresholds['max_g4']))

    cv_val = sum(cvs) / len(cvs)
    ind.fitness.cv = cv_val
    return cv_val


def get_feasible_infeasible(pop, thresholds, remove_constraints=()):
    """分离可行解与不可行解，不可行解按 cv 升序排列。"""
    index = []
    for i in range(len(pop)):
        if cv(pop[i], thresholds, remove_constraints) == 0:
            index.append(i)
    feasibles = [ind for j, ind in enumerate(pop) if j in index]
    infeasibles = [ind for j, ind in enumerate(pop) if j not in index]
    infeasibles = sorted(infeasibles, key=attrgetter("fitness.cv"), reverse=False)
    return feasibles, infeasibles
