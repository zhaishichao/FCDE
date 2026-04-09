import operator

from deap import gp, creator, base, tools


def init_toolbox(arity):
    # 创建GP框架的基本组件
    pset = gp.PrimitiveSet("MAIN", arity, 'x')
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)

    # 创建适应度和GP个体
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti, minimum_distance=None,
                   cosine_angle=None, min_center_distance=None)

    # 初始化toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    # toolbox.register("evaluate", evaluate)
    # toolbox.register("selTournament", selTournament_cv)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=5)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

    toolbox.register("select", tools.selNSGA2)  # NSGA-II选择（非支配排序后）

    return pset, toolbox
