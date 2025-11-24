import operator
from operator import attrgetter

import numpy as np
from deap import base, creator, tools, gp, algorithms
from deap.algorithms import varAnd
from deap.tools import selRandom, selTournamentDCD, selNSGA2, selTournament

from scipy.spatial.distance import cdist
from collections import Counter
from typing import Union, List


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
def sample_and_center(x):
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

    # 从原始数据中随机采样n个样本
    indices = np.random.choice(total_samples, size=n, replace=False)
    sampled_x = x[indices]

    # 计算采样样本的中心点（平均值）
    center = np.mean(sampled_x, axis=0)

    return center


def calculate_distance_stats(sample, x, k):
    """
    计算样本与特征数据中所有样本的欧式距离，并返回前k个最大和最小距离的平均值

    参数:
    sample: ndarray, 单个样本
    x: ndarray, 特征数据
    k: int, 要取的前k个距离

    返回:
    avg_max_dis: float, 前k个最大距离的平均值
    avg_min_dis: float, 前k个最小距离的平均值
    """

    # 确保sample是2D数组（单个样本）
    sample = sample.reshape(1, -1)

    # 计算sample与x中所有样本的欧式距离
    distances = cdist(sample, x, metric='euclidean')[0]

    # 对距离进行排序
    sorted_distances = np.sort(distances)

    # 获取前k个最小距离（包括自身，距离为0）
    min_k_distances = sorted_distances[:k]

    # 获取前k个最大距离
    max_k_distances = sorted_distances[-k:]

    # 计算平均值
    avg_min_dis = np.mean(min_k_distances)
    avg_max_dis = np.mean(max_k_distances)

    return avg_max_dis, avg_min_dis


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

    feasible = False
    if distances[0][0] > 0:
        feasible = True

    # print(f"前{k}个最近邻的标签: {k_nearest_labels}")
    # print(f"少数类出现次数: {minority_count}")
    # print(f"少数类比例: {proportion:.3f}")

    return proportion, feasible, distances[0][0]


def cosine_angle(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(cos))


class DSSMOTE_P_A:
    def __init__(self, X=None, y=None, evol_parameter=None):
        self.X = X
        self.y = y
        self.k = None
        self.total_syn = None
        self.parameter = evol_parameter
        self.data = self.preprocess_data()
        self.maj_center = sample_and_center(self.data['maj_x'])
        self.min_center = sample_and_center(self.data['min_x'])
        self.pset, self.toolbox = self.init_toolbox()

    ####################**********数据预处理**********####################

    # 1. 数据预处理
    def preprocess_data(self):
        '''
        将原始数据的多数类和少数类分割开，返回值是一个数据字典
        '''
        # 找出两个类别以及各自数量
        unique, counts = np.unique(self.y, return_counts=True)
        if len(unique) != 2:
            raise ValueError("数据集必须包含两个类别")

        # 判断哪个是少数类，哪个是多数类
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]

        maj_x = self.X[self.y == majority_class]
        maj_y = self.y[self.y == majority_class]

        min_x = self.X[self.y == minority_class]
        min_y = self.y[self.y == minority_class]

        self.total_syn = len(maj_y) - len(min_y)
        self.k = int(len(min_y) / 1)

        return {
            'maj_x': maj_x,
            'maj_y': maj_y,
            'min_x': min_x,
            'min_y': min_y
        }

    # 4. 评估个体
    def evaluate(self, individuals):
        '''
        :param individuals: 种群或个体
        :param index: 偏移量，用于指示当前子种群所对应的参考目标（多数类，少数类），来计算欧氏距离和角度
        :return: void
        '''
        for j, individual in enumerate(individuals):
            if not individual.fitness.valid:
                func = self.toolbox.compile(expr=individual)
                new_instance = func(*self.data['min_x'])
                # 计算当前实例与多数类和少数类中心的欧氏距离
                maj_dis = np.linalg.norm(self.maj_center - new_instance)
                min_dis = np.linalg.norm(self.min_center - new_instance)
                # 获取少数类的比例
                proportion, feasible, cv = minority_class_proportion(self.X, self.y, new_instance, self.k)
                # 评估新实例与多数类和少数类的距离
                maj_min_dis = maj_dis - min_dis

                individual.fitness.values = (maj_min_dis, proportion)
                individual.feasible = feasible
                # maj_center - min_center表示一条从少数类中心到多数类中心的向量
                individual.cosine_angle = cosine_angle(self.maj_center - self.min_center, new_instance)
                individual.fitness.cv = cv

    def get_feasible_infeasible(self, pop):
        '''
        :param pop: 种群
        :param constraints: 约束阈值
        :return: 可行解和不可行解
        '''
        feasible_inds = []
        infeasible_inds = []
        for ind in pop:
            if ind.feasible and ind.cosine_angle < 90:  # 满足约束条件的个体
                feasible_inds.append(ind)
            else:  # 不满足约束条件的个体
                infeasible_inds.append(ind)
        infeasible_inds = sorted(infeasible_inds, key=attrgetter("fitness.cv"), reverse=True)  # 对不可行个体按cv值降序排序
        return feasible_inds, infeasible_inds

    ####################**********GP进化合成实例**********####################
    # 5. 初始化toolbox
    def init_toolbox(self):
        # 创建GP框架的基本组件
        pset = gp.PrimitiveSet("MAIN", self.data['min_x'].shape[0], 'x')
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protectedDiv, 2)
        # pset.addEphemeralConstant("rand101", ephemeral=lambda: np.random.uniform(0, 1))
        # pset.addEphemeralConstant("rand101", partial(np.random.uniform, 0, 1))

        # 创建适应度和GP个体
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti, feasible=None, cosine_angle=None)

        # 初始化toolbox
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("selTournament", selTournament, tournsize=7)

        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=1, max_=6)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

        toolbox.register("select", selNSGA2)  # NSGA-II选择（非支配排序后）

        return pset, toolbox

    # 6. 进化
    def evolutionary(self):

        # 更新一下多数类和少数类的中心
        self.maj_center = sample_and_center(self.data['maj_x'])
        self.min_center = sample_and_center(self.data['min_x'])

        # 记录一下迭代信息
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "avg", "min", "max"

        # 初始化种群
        population = self.toolbox.population(n=self.parameter.POPSIZE)
        self.toolbox.evaluate(population)  # 评估初始种群
        population = self.toolbox.select(population, self.parameter.POPSIZE)

        # 进化搜索
        print('########### \t Start the evolution! \t ##########')
        for gen in range(0, self.parameter.NGEN):
            parent = self.toolbox.selTournament(population, self.parameter.POPSIZE)  # 选择父本
            offspring = varAnd(parent, self.toolbox, self.parameter.CXPB, self.parameter.MUTPB)  # 交叉、变异
            self.toolbox.evaluate(offspring)  # 评估变异后父本
            population = population + offspring

            population = remove_duplicate_individuals(population)

            if len(population) < self.parameter.POPSIZE:
                for i in range(self.parameter.POPSIZE - len(population)):
                    ind = self.toolbox.individual()
                    self.toolbox.evaluate(ind)
                    population.append(ind)

            # population = self.toolbox.select(population, self.parameter.POPSIZE)

            feasible_pop, infeasible_pop = self.get_feasible_infeasible(population)  # 得到可行个体与不可行个体
            if len(feasible_pop) >= self.parameter.POPSIZE:
                population = self.toolbox.select(feasible_pop, self.parameter.POPSIZE)
            elif len(feasible_pop) > 0:
                population = feasible_pop + infeasible_pop[
                                            :self.parameter.POPSIZE - len(
                                                feasible_pop)]  # 在不可行个体中选取违约程度小的个体，保证pop数量为POPSIZE
            else:
                population = feasible_pop + infeasible_pop[
                                            :self.parameter.POPSIZE - len(
                                                feasible_pop)]  # 加入不可行个体中违约程度小的个体，保证pop数量为POPSIZE
            # 更新记录
            record = stats.compile(population)
            logbook.record(gen=gen, **record)

            if self.parameter.verbose:
                print(logbook.stream)
        # 最后一代种群的最好个体

        pareto_fronts = tools.sortNondominated(population, len(population), first_front_only=True)
        inds_syn = pareto_fronts[0]
        if len(inds_syn) < 20:
            feasible_pop, infeasible_pop = self.get_feasible_infeasible(population)  # 得到可行个体与不可行个体
            if len(feasible_pop) >= 20:
                inds_syn = self.toolbox.select(feasible_pop, 20)
            else:
                inds_syn = feasible_pop + infeasible_pop[
                                          :20 - len(feasible_pop)]

        synthesis_instances = []
        for ind in inds_syn:
            func = self.toolbox.compile(expr=ind)
            synthesis_instance = func(*self.data['min_x'])
            synthesis_instances.append(synthesis_instance)

        print('前沿中个体数：', len(pareto_fronts[0]), '合成实例数：', len(inds_syn))
        return synthesis_instances

    # 7. 获取合成实例
    def synthesis_minority_instance(self):
        X_syn = []
        curr_syn = 0
        index = 1
        while curr_syn < self.total_syn:
            print('第', index, '轮合成')
            X_syn = X_syn + self.evolutionary()
            curr_syn = curr_syn + len(X_syn)
            index = index + 1

        # curr_syn > self.total_syn, 需要截取
        X_syn = X_syn[:self.total_syn]
        y_syn = [self.data['min_y'][0] for _ in range(len(X_syn))]
        return (X_syn, y_syn)

    # 8. 组合训练数据
    def fit_resample(self):
        synthesis_instance = self.synthesis_minority_instance()
        X_resampled = np.vstack((self.X.copy(), synthesis_instance[0]))
        y_resampled = np.hstack((self.y.copy(), synthesis_instance[1]))
        return X_resampled, y_resampled

    def fit_resample_synthesis_only(self):
        synthesis_instance = self.synthesis_minority_instance()
        X_resampled_synthes = np.array(synthesis_instance[0])
        y_resampled_synthes = np.array(synthesis_instance[1])
        return X_resampled_synthes, y_resampled_synthes
