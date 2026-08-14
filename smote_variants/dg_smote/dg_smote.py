"""
DG-SMOTE：基于多目标遗传规划的合成少数类过采样。

参考:
    Hu et al., "DG-SMOTE: A novel oversampling technique based on
    multi-objective genetic programming"（或其他相关文献）

依赖: deap, scikit-learn, numpy
"""

import math
import operator

import numpy as np
from deap import base, creator, gp, tools
from deap.algorithms import varAnd
from sklearn.base import BaseEstimator

from .operators import protectedDiv, selTournament


class DGSMOTE(BaseEstimator):
    """
    DG-SMOTE 过采样器。

    Parameters
    ----------
    random_state : int, default=42
        随机种子。
    pop_size : int, default=30
        种群大小。
    cx_prob : float, default=0.8
        交叉概率。
    mut_prob : float, default=0.2
        变异概率。
    n_gen : int, default=100
        进化代数。
    verbose : bool, default=True
        是否打印进化日志。
    num_generate : int or None
        需要生成的合成样本数，None 表示自动补齐到与多数类平衡。
    res_only : bool, default=False
        是否仅返回合成样本。
        False（默认）→ fit_resample 返回 (X_res, y_res) 完整重采样数据集；
        True         → fit_resample 返回 (X_syn, y_syn) 仅合成样本。
    """

    def __init__(self, random_state=42,
                 pop_size=30, cx_prob=0.8, mut_prob=0.2, n_gen=100,
                 verbose=True, num_generate=None, res_only=False):
        self.random_state = random_state
        self.pop_size = pop_size
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.n_gen = n_gen
        self.verbose = verbose
        self.num_generate = num_generate
        self.res_only = res_only

    # ====================== 数据预处理 ======================

    def preprocess_data(self):
        """找出两个类别并分离多数类 / 少数类。"""
        unique, counts = np.unique(self.y, return_counts=True)
        if len(unique) != 2:
            raise ValueError("数据集必须包含两个类别")

        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]

        return {
            'maj_x': self.X[self.y == majority_class],
            'min_x': self.X[self.y == minority_class],
            'min_class': minority_class,
        }

    @staticmethod
    def sort_by_euclidean(X, center):
        """按到 center 的欧氏距离升序排序。"""
        return X[np.argsort(np.linalg.norm(X - center, axis=1))]

    def generate_ref_target(self):
        """生成参考目标集合（多数类与少数类样本一一配对）。"""
        # 计算中心点并排序
        center_maj = np.mean(self.data['maj_x'], axis=0)
        center_min = np.mean(self.data['min_x'], axis=0)
        sorted_maj_x = self.sort_by_euclidean(self.data['maj_x'], center_maj)
        sorted_min_x = self.sort_by_euclidean(self.data['min_x'], center_min)

        np.random.seed(self.random_state)
        n_major = sorted_maj_x.shape[0]
        n_minor = sorted_min_x.shape[0]

        # 少数类样本数够用：直接打乱后取前 n_major 个
        if n_major <= n_minor:
            shuffled_min = sorted_min_x.copy()
            np.random.shuffle(shuffled_min)
            return (sorted_maj_x, shuffled_min[:n_major])

        # 少数类样本数不够用：按整倍数复制 + 随机抽取剩余部分
        repeat_times = n_major // n_minor
        remainder = n_major % n_minor
        repeated_blocks = [sorted_min_x.copy() for _ in range(repeat_times)]
        ref_target = np.vstack(repeated_blocks)
        np.random.shuffle(ref_target)
        if remainder > 0:
            extra_samples = sorted_min_x[np.random.choice(n_minor, remainder, replace=False)]
            ref_target = np.vstack([ref_target, extra_samples])

        return {'maj_x': sorted_maj_x, 'min_x': ref_target}

    # ====================== GP 进化 ======================

    def evaluate(self, individuals, index):
        """
        评估个体适应度 (angle, distance)。

        Parameters
        ----------
        individuals : list
            待评估的个体列表。
        index : int
            偏移量，指示当前子种群对应的参考目标（多数类，少数类）。
        """
        for individual in individuals:
            if not individual.fitness.valid:
                func = self.toolbox.compile(expr=individual)
                new_instance = func(*self.data['min_x'])
                a = self.dis_ref_target[index]
                b = np.linalg.norm(self.ref_target['maj_x'][index] - new_instance)
                c = np.linalg.norm(self.ref_target['min_x'][index] - new_instance)
                if b == 0 or c == 0:
                    angle = 0
                else:
                    angle = (a * a - b * b - c * c) / (-2 * b * c)
                    angle = min(1, max(-1, angle))
                    angle = math.degrees(math.acos(angle))
                distance = b - c
                individual.fitness.values = (angle, distance)

    def init_toolbox(self):
        """初始化 DEAP GP 工具箱。"""
        pset = gp.PrimitiveSet("MAIN", self.data['min_x'].shape[0], 'x')
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protectedDiv, 2)

        # 创建多目标适应度和 GP 个体
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("selTournament", selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=1, max_=6)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))

        return pset, toolbox

    def evolutionary(self, index):
        """为第 index 个合成样本运行一次 GP 进化，返回合成实例。"""
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "avg", "min", "max"

        population = self.toolbox.population(n=self.pop_size)
        self.toolbox.evaluate(population, index)  # 评估初始种群

        if self.verbose:
            print(f'########### \t Start the {index}th evolution \t ##########')
        for gen in range(self.n_gen):
            offspring = self.toolbox.selTournament(population, self.pop_size)  # 选择父本
            offspring = varAnd(offspring, self.toolbox, self.cx_prob, self.mut_prob)  # 交叉、变异
            self.toolbox.evaluate(offspring, index)  # 评估后代
            population = offspring  # 更新种群

            record = stats.compile(population)
            logbook.record(gen=gen, **record)
            if self.verbose:
                print(logbook.stream)

        # 最后一代种群的最优个体
        final_best_ind = tools.selBest(population, 1)[0]
        if self.verbose:
            print("final_best_ind", final_best_ind)
        func = self.toolbox.compile(expr=final_best_ind)
        return func(*self.data['min_x'])

    # ====================== 合成样本 ======================

    def synthesis_minority_instance(self):
        """生成所有合成样本，返回 (X_syn, y_syn)。"""
        if self.num_generate is None:
            self.num_generate = self.data['maj_x'].shape[0] - self.data['min_x'].shape[0]
        X_syn = [self.evolutionary(i) for i in range(self.num_generate)]
        y_syn = [self.data['min_class']] * self.num_generate
        return X_syn, y_syn

    def fit_resample(self, X, y):
        """
        过采样，接口与 imbalanced-learn 兼容。

        Parameters
        ----------
        X : ndarray
            特征矩阵。
        y : ndarray
            标签向量（二分类）。

        Returns
        -------
        res_only=False（默认）→ (X_res, y_res) 完整重采样数据集；
        res_only=True         → (X_syn, y_syn) 仅合成样本。
        """
        self.X = X
        self.y = y
        self.data = self.preprocess_data()
        self.ref_target = self.generate_ref_target()
        # 计算每对参考目标（多数类、少数类）之间的欧氏距离
        self.dis_ref_target = np.linalg.norm(
            self.ref_target['maj_x'] - self.ref_target['min_x'], axis=1)
        self.pset, self.toolbox = self.init_toolbox()

        X_syn, y_syn = self.synthesis_minority_instance()

        if self.res_only:
            return np.array(X_syn), np.array(y_syn)
        X_resampled = np.vstack((self.X.copy(), X_syn))
        y_resampled = np.hstack((self.y.copy(), y_syn))
        return X_resampled, y_resampled
