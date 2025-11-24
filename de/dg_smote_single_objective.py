import math
import operator

import numpy as np
from deap import base, creator, tools, gp
from deap.algorithms import varAnd
from deap.tools import selRandom


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


def selTournament(individuals, k, tournsize=3):
    chosen = []
    for i in range(k):
        chose = []
        # 找到距离大于等于0且角度最大的个体
        while len(chose) == 0:
            aspirants = selRandom(individuals, tournsize)
            max_angle = 0
            # 找到个体中最大的角度
            for j in range(tournsize):
                # if aspirants[j].fitness.values[1] >= 0:
                if aspirants[j].fitness.values[0] > max_angle:
                    max_angle = aspirants[j].fitness.values[0]
            # 找到所有最大角度的个体
            for j in range(tournsize):
                if aspirants[j].fitness.values[0] == max_angle:
                    chose.append(aspirants[j])

        if len(chose) > 1:
            max_index = 0
            max_distance = 0
            # 找到最大角度的个体中距离最大的个体
            for j in range(len(chose)):
                if chose[j].fitness.values[1] > max_distance:
                    max_index = j
                    max_distance = chose[j].fitness.values[1]
            chosen.append(chose[max_index])
        else:
            chosen.append(chose[0])

    return chosen


class DGSMOTE_SINGLE:
    def __init__(self, X=None, y=None, random_state=42,
                 evol_parameter=None, num_generate=None):
        self.X = X
        self.y = y
        self.random_state = random_state
        self.parameter = evol_parameter
        self.num_generate = num_generate
        self.data = self.preprocess_data()
        self.ref_target = self.generate_ref_target()
        # 计算每对目标样本之间的欧氏距离
        self.dis_ref_target = np.linalg.norm(self.ref_target['maj_x'] - self.ref_target['min_x'], axis=1)
        self.pset, self.toolbox = self.init_toolbox()

    ####################**********数据预处理**********####################

    # 1. 数据预处理
    def preprocess_data(self):
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

        return {
            'maj_x': maj_x,
            'maj_y': maj_y,
            'min_x': min_x,
            'min_y': min_y
        }

    # 2. 计算欧式距离并升序排序
    def sort_by_euclidean(self, X, center):
        distances = np.linalg.norm(X - center, axis=1)  # 每行与中心点的欧氏距离
        sorted_indices = np.argsort(distances)  # 获取升序索引
        return X[sorted_indices]

    # 3. 生成参考目标集合（用于计算欧氏距离和角度）
    def generate_ref_target(self):
        '''
        :return: target（包含多数类和少数类的组合，一一配对）
        '''
        # 计算中心点
        center_maj = np.mean(self.data['maj_x'], axis=0)
        center_min = np.mean(self.data['min_x'], axis=0)
        # 获取排序后的样本
        sorted_maj_x = self.sort_by_euclidean(self.data['maj_x'], center_maj)
        sorted_min_x = self.sort_by_euclidean(self.data['min_x'], center_min)
        # 获取样本数量
        np.random.seed(self.random_state)
        n_major = sorted_maj_x.shape[0]
        n_minor = sorted_min_x.shape[0]
        # 少数类样本数够用
        if n_major <= n_minor:
            # 直接打乱后取前n_major个
            shuffled_min = sorted_min_x.copy()
            np.random.shuffle(shuffled_min)
            return (sorted_maj_x, shuffled_min[:n_major])
        # 少数类样本数不够用，计算整倍数和剩余部分
        repeat_times = n_major // n_minor
        remainder = n_major % n_minor
        # 复制整个 sorted_min_x 的 block 多次
        repeated_blocks = [sorted_min_x.copy() for _ in range(repeat_times)]
        ref_target = np.vstack(repeated_blocks)
        np.random.shuffle(ref_target)
        # 剩余部分：随机从 sorted_min_x 中抽 remainder 个样本
        if remainder > 0:
            extra_samples = sorted_min_x[np.random.choice(n_minor, remainder, replace=False)]
            ref_target = np.vstack([ref_target, extra_samples])

        return {
            'maj_x': sorted_maj_x,
            'min_x': ref_target
        }

    ####################**********GP进化**********####################

    # 4. 评估个体
    def evaluate(self, individuals, index):
        '''
        :param individuals: 种群或个体
        :param index: 偏移量，用于指示当前子种群所对应的参考目标（多数类，少数类），来计算欧氏距离和角度
        :return: void
        '''
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
                    if angle > 1:
                        angle = 1
                    if angle < -1:
                        angle = -1
                    angle = math.degrees(math.acos(angle))
                distance = b - c
                individual.fitness.values = (angle, distance)

    ####################**********GP进化合成实例**********####################

    # 5. 初始化toolbox
    def init_toolbox(self):
        # 创建GP框架的基本组件
        pset = gp.PrimitiveSet("MAIN", self.data['min_x'].shape[0], 'x')
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protectedDiv, 2)
        # pset.addEphemeralConstant("rand101", partial(np.random.uniform, 0, 1))

        # 创建适应度和GP个体
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        # 初始化toolbox
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
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

    # 6. 进化
    def evolutionary(self, index):
        # hof = tools.HallOfFame(1)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "avg", "min", "max"
        # 为每一个要生成的样本创建一个GP子过程
        population = self.toolbox.population(n=self.parameter.POPSIZE)
        self.toolbox.evaluate(population, index)  # 评估初始种群

        # 进化
        if self.parameter.verbose:
            print(f'########### \t Start the {index}th evolution \t ##########')
        for gen in range(0, self.parameter.NGEN):
            offspring = self.toolbox.selTournament(population, self.parameter.POPSIZE)  # 选择父本
            offspring = varAnd(offspring, self.toolbox, self.parameter.CXPB, self.parameter.MUTPB)  # 交叉、变异
            self.toolbox.evaluate(offspring, index)  # 评估变异后父本
            population = offspring  # 更新种群

            # 更新记录
            # hof.update(population)
            record = stats.compile(population)
            logbook.record(gen=gen, **record)

            if self.parameter.verbose:
                print(logbook.stream)
        # 最后一代种群的最好个体
        final_best_ind = tools.selBest(population, 1)[0]
        if self.parameter.verbose:
            print("final_best_ind", final_best_ind)
        func = self.toolbox.compile(expr=final_best_ind)
        synthesis_instance = func(*self.data['min_x'])
        return synthesis_instance

    # 7. 获取合成实例
    def synthesis_minority_instance(self):
        if self.num_generate is None:
            self.num_generate = self.data['maj_x'].shape[0] - self.data['min_x'].shape[0]
        X_syn = []
        for index in range(self.num_generate):
            X_syn.append(self.evolutionary(index))
        y_syn = [self.data['min_y'][0] for _ in range(self.num_generate)]
        return (X_syn, y_syn)

    # 8. 组合训练数据
    def fit_resample(self):
        synthesis_instance = self.synthesis_minority_instance()
        X_resampled = np.vstack((self.X.copy(), synthesis_instance[0]))
        y_resampled = np.hstack((self.y.copy(), synthesis_instance[1]))
        return X_resampled, y_resampled

    # 9. 只返回合成样本
    def fit_resample_synthesis_only(self):
        synthesis_instance = self.synthesis_minority_instance()
        X_resampled_synthes = np.array(synthesis_instance[0])
        y_resampled_synthes = np.array(synthesis_instance[1])
        return X_resampled_synthes, y_resampled_synthes
