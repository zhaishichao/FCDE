import operator
from operator import attrgetter

import numpy as np
from deap import base, creator, tools, gp
from deap.algorithms import varAnd
from deap.tools import selNSGA2

from .data_preprocess import separate_maj_min, random_sampling, calculate_center, calculate_ave_max_distance
from .initialization import init_toolbox


class DSSMOTE:
    def __init__(self, X=None, y=None, evol_parameter=None):
        self.X = X  # 原始数据
        self.y = y  # 原始标签
        self.parameter = evol_parameter # 进化参数
        self.data = separate_maj_min(self.X, self.y) # 分离多数类和少数类后的数据
        self.maj_samples = random_sampling(self.data['maj_x']) # 随机采样多数类样本
        self.maj_center = calculate_center(self.maj_samples) # 多数类中心点
        self.min_samples = random_sampling(self.data['min_x']) # 随机采样少数类样本
        self.min_center = calculate_center(self.maj_samples) # 少数类中心点
        self.mean_min_distance = calculate_ave_max_distance(self.min_samples, k=5) # 计算与少数类中心的平均最大距离
        self.pset, self.toolbox = init_toolbox(len(self.data['min_x']))
        self.cv_list = []

    # 评估个体
    def evaluate(self, individuals):
        for j, individual in enumerate(individuals):
            if not individual.fitness.valid:
                func = self.toolbox.compile(expr=individual)
                new_instance = func(*self.data['min_x'])
                # 计算当前实例与多数类和少数类中心的欧氏距离
                maj_dis = np.linalg.norm(self.maj_center - new_instance)
                min_dis = np.linalg.norm(self.min_center - new_instance)
                # 评估新实例与多数类和少数类的距离
                maj_min_dis = maj_dis - min_dis
                # 计算少数类的比例
                proportion, minimum_distance = minority_class_proportion(self.X, self.y, new_instance,
                                                                         len(self.data['min_x']))
                individual.fitness.values = (maj_min_dis, proportion)
                individual.minimum_distance = minimum_distance

                # 计算角度 maj_center - min_center表示一条从少数类中心到多数类中心的向量
                individual.cosine_angle = cosine_angle(self.maj_center - self.min_center, new_instance)

                # 计算平均最小距离
                dis = np.linalg.norm(self.min_center - new_instance)
                individual.min_center_distance = dis - self.mean_min_distance

    # 计算约束违反程度 cv
    def cv(self, individual, constraint_thresholds):
        difference = []
        # difference.append(max(0,  -individual.minimum_distance / constraint_thresholds['minimum_distance']))
        difference.append(max(0, -individual.fitness.values[0] / constraint_thresholds['maj_min_distance']))
        difference.append(max(0, (individual.cosine_angle - 90) / constraint_thresholds['cosine_angle']))
        # difference.append(max(0, individual.min_center_distance / constraint_thresholds['min_center_distance']))
        cv = sum(difference) / 2  # 求0和cv中的最小值之和，cv=0，表示是一个可行个体
        individual.fitness.cv = cv  # 将cv值保存在个体中
        return cv

    def get_feasible_infeasible(self, pop, constraint_thresholds):
        '''
        :param pop: 种群
        :param constraints: 约束阈值
        :return: 可行解和不可行解
        '''
        index = []
        for i in range(len(pop)):
            if self.cv(pop[i], constraint_thresholds) != 0:  # 判断个体适应度是否都满足约束条件
                index.append(i)  # 将不符合约束条件的个体的索引添加到index中
        feasible_pop = [ind for j, ind in enumerate(pop) if j not in index]  # 可行个体
        infeasible_pop = [ind for j, ind in enumerate(pop) if j in index]  # 不可行个体
        infeasible_pop = sorted(infeasible_pop, key=attrgetter("fitness.cv"), reverse=False)  # 对不可行个体按cv值升序排列
        return feasible_pop, infeasible_pop

    ####################**********GP进化合成实例**********####################
    # 5. 初始化toolbox

    # 6. 进化
    def evolutionary(self):

        # 更新一下多数类和少数类的中心
        self.maj_center, self.maj_samples = sample_and_center(self.data['maj_x'])
        self.min_center, self.min_samples = sample_and_center(self.data['min_x'])
        # 计算少数类中的平均最小距离
        self.mean_min_distance = calculate_k_min_distances_mean(self.min_samples, k=5)

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

        # 计算个体的统计信息
        constraint_thresholds = calculate_statistics_inndividuals(population)
        self.get_feasible_infeasible(population, constraint_thresholds)  # 得到可行个体与不可行个体

        # 进化搜索
        cv_list = []
        print('########### \t Start the evolution! \t ##########')
        for gen in range(0, self.parameter.NGEN):
            parent = self.toolbox.selTournament(population, self.parameter.POPSIZE)  # 选择父本
            offspring = varAnd(parent, self.toolbox, self.parameter.CXPB, self.parameter.MUTPB)  # 交叉、变异
            self.toolbox.evaluate(offspring)  # 评估变异后父本

            population = population + offspring

            population = remove_duplicate_individuals(population)
            # print('重复个体数：', self.parameter.POPSIZE * 2 - len(mixed_pop))

            while len(population) < self.parameter.POPSIZE:
                for i in range(self.parameter.POPSIZE - len(population)):
                    ind = self.toolbox.individual()
                    self.toolbox.evaluate(ind)
                    population.append(ind)
                population = remove_duplicate_individuals(population)

            # population = self.toolbox.select(population, self.parameter.POPSIZE)

            feasible_pop, infeasible_pop = self.get_feasible_infeasible(population,
                                                                        constraint_thresholds)  # 得到可行个体与不可行个体

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
            # print(f'第{gen}代平均约束值', calculate_mean_inndividuals_cv(population, constraint_thresholds))
            mean_list = []
            for individual in population:
                mean_list.append(individual.fitness.cv)
            cv_list.append(np.mean(mean_list))
            # print('第一个个体的cv值：', population[0].fitness.cv)
            # 更新记录
            record = stats.compile(population)
            logbook.record(gen=gen, **record)

            if self.parameter.verbose:
                print(logbook.stream)
        # 最后一代种群的最好个体

        feasible_pop, infeasible_pop = self.get_feasible_infeasible(population, constraint_thresholds)  # 得到可行个体与不可行个体
        pareto_fronts = [[]]
        if len(feasible_pop) == 0:
            inds_syn = infeasible_pop[
                       :10 - len(feasible_pop)]
        else:
            pareto_fronts = tools.sortNondominated(feasible_pop, len(feasible_pop), first_front_only=True)
            inds_syn = pareto_fronts[0]
            if len(inds_syn) < 10:
                if len(inds_syn) + len(feasible_pop) >= 10:
                    inds_syn = inds_syn + feasible_pop[
                                          :10 - len(inds_syn)]
                else:
                    inds_syn = feasible_pop + infeasible_pop[
                                              :10 - (len(feasible_pop) + len(inds_syn))]
        synthesis_instances = []
        for ind in inds_syn:
            func = self.toolbox.compile(expr=ind)
            synthesis_instance = func(*self.data['min_x'])
            synthesis_instances.append(synthesis_instance)

        print('前沿中个体数：', len(pareto_fronts[0]), '合成实例数：', len(inds_syn), '可行解数量：', len(feasible_pop))

        self.cv_list = cv_list

        return synthesis_instances

    # 7. 获取合成实例
    def synthesis_minority_instance(self):
        X_syn = []
        curr_syn = 0
        index = 1
        total_syn = len(self.data['maj_y']) - len(self.data['min_y'])
        while curr_syn < total_syn:
            print('第', index, '轮合成')
            syn = self.evolutionary()
            X_syn = X_syn + syn
            curr_syn = curr_syn + len(syn)
            index = index + 1

        X_syn = X_syn[:total_syn]
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

    def curve_fitting(self, file_path, filename):
        curve_fitting(self.cv_list, file_path, filename)
