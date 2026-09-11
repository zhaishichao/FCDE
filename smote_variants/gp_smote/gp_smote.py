"""
GP-SMOTE：基于多目标遗传规划（带约束）的合成少数类过采样。

四个约束（g1~g4）见 constraints.py，通过 remove_constraints 控制去掉哪些约束；
当去掉全部约束时退化为无约束版本（Pareto 支配 + 拥挤距离选择）。
"""

import numpy as np
from deap import tools
from deap.algorithms import varAnd
from sklearn.base import BaseEstimator

from .constraints import calculate_constraint_thresholds, get_feasible_infeasible
from .data_preprocess import (separate_maj_min, random_sampling, calculate_center,
                              calculate_ave_max_distance, minority_class_proportion,
                              calculate_cosine_angle, calculate_min_distance, compute_avg_distance)
from .initialization import init_toolbox
from .operators import (remove_duplicate_individuals, selTournament_cv,
                        selTournament_domination)
from .visualize import curve_fitting


class GPSMOTE(BaseEstimator):
    """
    GP-SMOTE 过采样器。

    Parameters
    ----------
    random_state : int or None, default=None
        随机种子，None 表示不设置（沿用全局随机）。
    pop_size : int, default=30
        种群大小。
    cx_prob : float, default=0.8
        交叉概率。
    mut_prob : float, default=0.2
        变异概率。
    n_gen : int, default=100
        进化代数。
    verbose : bool, default=False
        是否打印进化日志。
    remove_constraints : tuple of int, default=()
        要去掉的约束编号集合（1~4），用于消融实验。
        四个约束依次为：1=最小距离、2=中心距离、3=夹角、4=第一目标。
        默认空表示启用全部四个约束；(1, 2, 3, 4) 表示去掉全部约束（等价于 woc 版本）。
    res_only : bool, default=False
        是否仅返回合成样本。
        False（默认）→ fit_resample 返回 (X_res, y_res)；
        True         → fit_resample 返回 (X_syn, y_syn)。
    """

    def __init__(self, random_state=None,
                 pop_size=30, cx_prob=0.8, mut_prob=0.2, n_gen=100,
                 verbose=False, remove_constraints=(), res_only=False):
        self.random_state = random_state
        self.pop_size = pop_size
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.n_gen = n_gen
        self.verbose = verbose
        self.remove_constraints = tuple(remove_constraints)
        self.res_only = res_only

    # ====================== 评估个体 ======================

    def evaluate(self, individuals):
        for individual in individuals:
            if not individual.fitness.valid:
                func = self.toolbox.compile(expr=individual)
                new_instance = func(*self.data['min_x'])
                # 计算当前实例与多数类 / 少数类中心的欧氏距离
                maj_dis = np.linalg.norm(self.maj_center - new_instance)
                min_dis = np.linalg.norm(self.min_center - new_instance)
                # 第一个目标：新实例到多数类与少数类中心的距离差
                maj_min_dis = maj_dis - min_dis
                # 第二个目标：新实例 k 近邻中少数类的比例
                proportion, _ = minority_class_proportion(self.X_samples, self.y_samples,
                                                          new_instance, len(self.min_samples))
                individual.fitness.values = (maj_min_dis, proportion)

                # 约束1：新实例与所有少数类（含已合成实例）的最小距离
                individual.distance_minority_min = calculate_min_distance(
                    self.min_samples_and_synthesis, new_instance)

                # 约束2：新实例离少数类中心的距离超出平均最大距离的量
                individual.center_distance_excess = (
                    np.linalg.norm(self.min_center - new_instance) - self.ave_max_distance)

                # 约束3：新实例与少数类/多数类中心连线的夹角
                individual.cosine_angle = calculate_cosine_angle(
                    self.maj_center - self.min_center, new_instance - self.min_center)

    # ====================== 进化 ======================

    def evolutionary(self):
        # 每次进化前重新采样多数类与少数类
        self.maj_samples = random_sampling(self.data['maj_x'])
        self.maj_center = calculate_center(self.maj_samples)
        self.min_samples = random_sampling(self.data['min_x'])
        self.min_center = calculate_center(self.min_samples)
        self.X_samples = np.concatenate([self.maj_samples, self.min_samples], axis=0)
        maj_samples_y = np.array([self.data['maj_y'][0] for _ in range(len(self.maj_samples))])
        min_samples_y = np.array([self.data['min_y'][0] for _ in range(len(self.min_samples))])
        self.y_samples = np.concatenate([maj_samples_y, min_samples_y], axis=0)

        self.ave_max_distance = calculate_ave_max_distance(
            self.min_samples, k=len(self.min_samples) // 5)

        # 初始化种群
        population = self.toolbox.population(n=self.pop_size)
        self.toolbox.evaluate(population)

        # 是否还有启用的约束（决定选择算子和环境选择策略）
        use_constraints = len(self.remove_constraints) < 4
        if use_constraints:
            thresholds = calculate_constraint_thresholds(
                population, self.min_avg_distance, self.remove_constraints)
            get_feasible_infeasible(population, thresholds, self.remove_constraints)

        cv_list = []
        feasible_ratio_list = []

        for gen in range(self.n_gen):
            if use_constraints:
                parent = self.toolbox.selTournament(population, self.pop_size)
            else:
                # 无约束版本：先 NSGA-II 选择，再用 Pareto 支配锦标赛选父本
                population = self.toolbox.select(population, self.pop_size)
                parent = self.toolbox.selTournament(population, self.pop_size)

            offspring = varAnd(parent, self.toolbox, self.cx_prob, self.mut_prob)
            self.toolbox.evaluate(offspring)
            population = population + offspring

            # 去重，并补足种群规模
            population = remove_duplicate_individuals(population)
            while len(population) < self.pop_size:
                for _ in range(self.pop_size - len(population)):
                    ind = self.toolbox.individual()
                    self.toolbox.evaluate(ind)
                    population.append(ind)
                population = remove_duplicate_individuals(population)

            # 环境选择
            if use_constraints:
                feasible_pop, infeasible_pop = get_feasible_infeasible(
                    population, thresholds, self.remove_constraints)
                if len(feasible_pop) >= self.pop_size:
                    population = self.toolbox.select(feasible_pop, self.pop_size)
                elif len(feasible_pop) > 0:
                    population = feasible_pop + infeasible_pop[:self.pop_size - len(feasible_pop)]
                else:
                    population = feasible_pop + infeasible_pop[:self.pop_size - len(feasible_pop)]

                cv_list.append(np.mean([ind.fitness.cv for ind in population]))
                if self.verbose and gen % 20 == 0:
                    feasible_pop, _ = get_feasible_infeasible(
                        population, thresholds, self.remove_constraints)
                    ratio = len(feasible_pop) / len(population)
                    feasible_ratio_list.append(ratio)
                    obj1_lt_zero = sum(1 for ind in population if ind.fitness.values[0] < 0)
                    print(f'第{gen}代 可行解占比: {ratio:.2%} '
                          f'({len(feasible_pop)}/{len(population)})  '
                          f'目标1<0个体数: {obj1_lt_zero}/{len(population)}')
            else:
                population = self.toolbox.select(population, self.pop_size)

        # 规范化检查：去除不包含任何特征变量（x0, x1, ...）的个体
        feature_names = [f'x{i}' for i in range(len(self.data['min_x']))]
        population = [ind for ind in population
                      if any(fn in str(ind) for fn in feature_names)]

        # 最后一代种群
        synthesis_instances = []
        for ind in population:
            func = self.toolbox.compile(expr=ind)
            synthesis_instances.append(func(*self.data['min_x']))

        self.cv_list = cv_list
        self.feasible_ratio_list = feasible_ratio_list
        return synthesis_instances

    # ====================== 合成样本 ======================

    def synthesis_minority_instance(self):
        X_syn = []
        curr_syn = 0
        index = 1
        total_syn = len(self.data['maj_y']) - len(self.data['min_y'])
        while curr_syn < total_syn:
            syn = self.evolutionary()
            self.min_samples_and_synthesis = np.vstack((self.min_samples_and_synthesis, syn))
            X_syn = X_syn + syn
            curr_syn = curr_syn + len(syn)
            index = index + 1
        if self.verbose:
            print(f'共计合成：{index - 1}轮，合成数量为：{total_syn}')
        X_syn = X_syn[:total_syn]
        y_syn = [self.data['min_y'][0] for _ in range(len(X_syn))]
        return X_syn, y_syn

    # ====================== 公共接口 ======================

    def fit_resample(self, X, y):
        """
        过采样，接口与 imbalanced-learn 兼容。

        Returns
        -------
        res_only=False（默认）→ (X_res, y_res) 完整重采样数据集；
        res_only=True         → (X_syn, y_syn) 仅合成样本。
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.X = X
        self.y = y
        self.data = separate_maj_min(self.X, self.y)

        self.maj_samples = None
        self.maj_center = None
        self.min_samples = None
        self.min_samples_and_synthesis = self.data['min_x']
        self.min_center = None
        self.X_samples = None
        self.y_samples = None
        self.ave_max_distance = None
        self.min_avg_distance = compute_avg_distance(self.data['min_x']) / 2

        self.pset, self.toolbox = init_toolbox(len(self.data['min_x']))
        self.toolbox.register("evaluate", self.evaluate)
        if len(self.remove_constraints) < 4:
            self.toolbox.register("selTournament", selTournament_cv)
        else:
            self.toolbox.register("selTournament", selTournament_domination)
        self.cv_list = []

        X_syn, y_syn = self.synthesis_minority_instance()

        if self.res_only:
            return np.array(X_syn), np.array(y_syn)
        X_resampled = np.vstack((self.X.copy(), X_syn))
        y_resampled = np.hstack((self.y.copy(), y_syn))
        return X_resampled, y_resampled

    def curve_fitting(self, file_path, filename, title):
        curve_fitting(self.cv_list, file_path, filename, title)
