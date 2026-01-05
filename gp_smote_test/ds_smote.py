import numpy as np
from deap import tools
from deap.algorithms import varAnd
from .constraints import calculate_constraint_thresholds, get_feasible_infeasible
from .data_preprocess import separate_maj_min, random_sampling, calculate_center, calculate_ave_max_distance, \
    minority_class_proportion, calculate_cosine_angle, calculate_min_distance
from .initialization import init_toolbox

from .operators import remove_duplicate_individuals, selTournament_cv
from .visualize import curve_fitting


class DSSMOTE:
    def __init__(self, X=None, y=None, evol_parameter=None):
        self.X = X  # 原始数据
        self.y = y  # 原始标签
        self.parameter = evol_parameter  # 进化参数
        self.data = separate_maj_min(self.X, self.y)  # 分离多数类和少数类后的数据

        self.maj_samples = None  # 随机采样多数类样本
        self.maj_center = None  # 多数类中心点
        self.min_samples = None  # 随机采样少数类样本
        self.min_samples_and_synthesis = self.data['min_x'] # 少数类样本和少数类合成样本
        self.min_center = None  # 少数类中心点
        self.X_samples = None  # 采样得到的特征
        self.y_samples = None  # 采样得到的标签
        self.ave_max_distance = None  # 少数类中心的平均最大距离

        self.pset, self.toolbox = init_toolbox(len(self.data['min_x']))
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("selTournament", selTournament_cv)

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
                # 评估新实例与多数类和少数类的距离 （第一个目标）
                maj_min_dis = maj_dis - min_dis
                # 计算少数类的比例 （第二个目标）
                proportion, _ = minority_class_proportion(self.X_samples, self.y_samples, new_instance,
                                                                         len(self.min_samples))
                individual.fitness.values = (maj_min_dis, proportion)

                # 计算与所有少数类（包括新合成的实例）的最小距离
                # （第一个约束：新实例与所有的少数类的最小距离大于DIS，依据DIS的值来控制实例的离散程度）
                individual.distance_minority_min = calculate_min_distance(self.min_samples_and_synthesis, new_instance)

                # 计算离少数类中心的距离
                # （第三个约束：新实例与少数类中心的距离，要小于离少数类中心最远的距离）
                distance_minority_center = np.linalg.norm(self.min_center - new_instance)
                individual.distance_minority_center = distance_minority_center - self.ave_max_distance

                # 计算角度 maj_center - min_center表示一条从少数类中心到多数类中心的向量
                # （第四个约束：新实例与多数类中心、少数类中心的夹角小于90°）
                individual.cosine_angle = calculate_cosine_angle(self.maj_center - self.min_center,
                                                                 new_instance - self.min_center)



    # 6. 进化
    def evolutionary(self):

        # 每次执行进化搜索前，对多少类和少数类重新采样
        self.maj_samples = random_sampling(self.data['maj_x'])  # 随机采样多数类样本
        self.maj_center = calculate_center(self.maj_samples)  # 多数类中心点
        self.min_samples = random_sampling(self.data['min_x'])  # 随机采样少数类样本
        self.min_center = calculate_center(self.min_samples)  # 少数类中心点
        self.X_samples = np.concatenate([self.maj_samples, self.min_samples], axis=0)  # 合并特征
        maj_samples_y = np.array([self.data['maj_y'][0] for _ in range(len(self.maj_samples))])
        min_samples_y = np.array([self.data['min_y'][0] for _ in range(len(self.min_samples))])
        self.y_samples = np.concatenate([maj_samples_y, min_samples_y], axis=0)  # 合并标签

        self.ave_max_distance = calculate_ave_max_distance(self.min_samples,
                                                           k=len(self.min_samples) // 5)  # 计算与少数类中心的平均最大距离

        # 初始化种群
        population = self.toolbox.population(n=self.parameter.POPSIZE)
        self.toolbox.evaluate(population)  # 评估初始种群

        thresholds = calculate_constraint_thresholds(population)  # 计算初始种群的最大约束违反程度
        get_feasible_infeasible(population, thresholds)  # 得到可行个体与不可行个体

        # 进化搜索
        cv_list = []  # 存储约束违反程度（用于绘制收敛曲线）
        # print('########### \t Start the evolution! \t ##########')
        for gen in range(0, self.parameter.NGEN):
            parent = self.toolbox.selTournament(population, self.parameter.POPSIZE)  # 选择父本
            offspring = varAnd(parent, self.toolbox, self.parameter.CXPB, self.parameter.MUTPB)  # 交叉、变异
            self.toolbox.evaluate(offspring)  # 评估变异后父本

            population = population + offspring

            # 去重
            population = remove_duplicate_individuals(population)
            # print('重复个体数：', self.parameter.POPSIZE * 2 - len(mixed_pop))
            while len(population) < self.parameter.POPSIZE:
                for i in range(self.parameter.POPSIZE - len(population)):
                    ind = self.toolbox.individual()
                    self.toolbox.evaluate(ind)
                    population.append(ind)
                population = remove_duplicate_individuals(population)

            # 环境选择
            feasible_pop, infeasible_pop = get_feasible_infeasible(population, thresholds)  # 得到可行个体与不可行个体
            if len(feasible_pop) >= self.parameter.POPSIZE:
                population = self.toolbox.select(feasible_pop, self.parameter.POPSIZE)
            elif len(feasible_pop) > 0:
                population = feasible_pop + infeasible_pop[:self.parameter.POPSIZE - len(
                    feasible_pop)]  # 在不可行个体中选取违约程度小的个体，保证pop数量为POPSIZE
            else:
                population = feasible_pop + infeasible_pop[:self.parameter.POPSIZE - len(
                    feasible_pop)]  # 加入不可行个体中违约程度小的个体，保证pop数量为POPSIZE

            # print(f'第{gen}代平均约束值', calculate_mean_inndividuals_cv(population, thresholds))
            # 记录一下约束值去的变化
            cv_list.append(np.mean([ind.fitness.cv for ind in population]))

        # 最后一代种群
        feasible_pop, infeasible_pop = get_feasible_infeasible(population, thresholds)  # 得到可行个体与不可行个体
        pareto_fronts = [[]]
        if len(feasible_pop) == 0:
            inds_syn = infeasible_pop[:5]
        else:
            pareto_fronts = tools.sortNondominated(feasible_pop, len(feasible_pop), first_front_only=True)
            inds_syn = pareto_fronts[0]
            if len(inds_syn) < 5:
                if len(feasible_pop) >= 5:
                    inds_syn = self.toolbox.select(feasible_pop, 5)
                else:
                    inds_syn = feasible_pop + infeasible_pop[:5 - len(feasible_pop)]
        synthesis_instances = []
        for ind in inds_syn:
            func = self.toolbox.compile(expr=ind)
            synthesis_instance = func(*self.data['min_x'])
            synthesis_instances.append(synthesis_instance)

        print('可行解数量：', len(feasible_pop), '前沿中个体数：', len(pareto_fronts[0]), '合成实例数：', len(inds_syn))

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
            self.min_samples_and_synthesis = np.vstack((self.min_samples_and_synthesis, syn))
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

    def curve_fitting(self, file_path, filename, title):
        curve_fitting(self.cv_list, file_path, filename, title)
