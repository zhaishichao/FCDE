from deap import tools


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