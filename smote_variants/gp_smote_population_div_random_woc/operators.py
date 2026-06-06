from deap import tools
import random


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



def remove_duplicate_individuals(individuals):
    seen = set()
    result = []
    for ind in individuals:
        key = str(ind)
        if key not in seen:
            seen.add(key)
            result.append(ind)
    return result


def selTournament_domination(individuals, k):
    """基于 Pareto 支配 + 拥挤距离的锦标赛选择，选 k 个个体"""
    chosen = []
    while len(chosen) < k:
        aspirants = tools.selRandom(individuals, 2)
        ind1, ind2 = aspirants[0], aspirants[1]

        if ind1.fitness.dominates(ind2.fitness):
            chosen.append(ind1)
        elif ind2.fitness.dominates(ind1.fitness):
            chosen.append(ind2)
        else:
            if ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
                chosen.append(ind1)
            elif ind2.fitness.crowding_dist > ind1.fitness.crowding_dist:
                chosen.append(ind2)
            else:
                chosen.append(ind1 if random.random() <= 0.5 else ind2)
    return chosen