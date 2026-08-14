"""
MTGP-SMOTE: Multitree Genetic Programming for Synthetic Minority Over-Sampling

Based on:
  Cui et al., "Multitree genetic programming with spherical-based operators
  for synthetic minority over-sampling technique in unbalanced data",
  Swarm and Evolutionary Computation 98 (2025) 102126.

依赖: deap, scikit-learn, imbalanced-learn, numpy
"""

import random
import copy
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from deap import base, creator, gp, tools

warnings.filterwarnings("ignore")

# ── 全局注册 DEAP 类型（只注册一次）
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "MTGPIndividual"):
    creator.create("MTGPIndividual", list, fitness=creator.FitnessMax)


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def protected_div(a, b):
    if isinstance(b, np.ndarray):
        return np.where(np.abs(b) < 1e-10, 0.0, a / b)
    return 0.0 if abs(b) < 1e-10 else a / b


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def _dis_ang_score(ind_vec, mint_vec, majt_vec):
    """单棵树的 Dis_Ang 适应度（论文公式 3 单项）"""
    b = _dist(ind_vec, mint_vec)
    c = _dist(ind_vec, majt_vec)
    if b < 1e-10:
        return 0.0
    denom = 2.0 * b * c
    if denom < 1e-10:
        return 0.0
    a = _dist(majt_vec, mint_vec)
    cos_val = np.clip((b**2 + c**2 - a**2) / denom, -1.0, 1.0)
    alpha = float(np.arccos(cos_val))
    return (c / (b + c)) * alpha


def _in_mint_hemisphere(ind_vec, mint_vec, majt_vec):
    """判断 ind 是否在靠近 mint 的半球内（无需遗传操作）"""
    center = (np.asarray(mint_vec, float) + np.asarray(majt_vec, float)) / 2.0
    radius = _dist(mint_vec, majt_vec) / 2.0
    if _dist(ind_vec, center) > radius:
        return False
    return _dist(ind_vec, mint_vec) <= _dist(ind_vec, majt_vec)


def _eval_tree(tree, pset, n_features):
    """编译并执行 GP 树，返回 n_features 维 numpy 向量。"""
    try:
        func = gp.compile(tree, pset)
        result = func()
        if not isinstance(result, np.ndarray):
            result = np.full(n_features, float(result))
        result = np.asarray(result, float)
        if result.shape != (n_features,):
            result = np.resize(result, (n_features,))
    except Exception:
        result = np.zeros(n_features)
    return result


# ─────────────────────────────────────────────
# 构建 PrimitiveSet（每次唯一命名，避免 DEAP 全局污染）
# ─────────────────────────────────────────────

_pset_counter = [0]

def _build_pset(minority_instances):
    _pset_counter[0] += 1
    uid = _pset_counter[0]
    pset = gp.PrimitiveSet(f"P{uid}", arity=0)
    pset.addPrimitive(np.add,        2, name="add")
    pset.addPrimitive(np.subtract,   2, name="sub")
    pset.addPrimitive(np.multiply,   2, name="mul")
    pset.addPrimitive(protected_div, 2, name="pdiv")
    for i, inst in enumerate(minority_instances):
        pset.addTerminal(np.asarray(inst, float).copy(), name=f"x{i}")
    pset.addEphemeralConstant(f"r{uid}", lambda: np.float64(random.random()))
    return pset


# ─────────────────────────────────────────────
# 个体适应度
# ─────────────────────────────────────────────

def _eval_individual(trees, psets, pair_list, n_features):
    total = 0.0
    for tree, pset, (mint, majt) in zip(trees, psets, pair_list):
        vec = _eval_tree(tree, pset, n_features)
        total += _dis_ang_score(vec, mint, majt)
    return total


# ─────────────────────────────────────────────
# 球形遗传算子
# ─────────────────────────────────────────────

def _spherical_crossover(trees1, trees2, psets, pair_list, n_features):
    c1 = [copy.deepcopy(t) for t in trees1]
    c2 = [copy.deepcopy(t) for t in trees2]
    for i, (t1, t2, pset, (mint, majt)) in enumerate(
            zip(c1, c2, psets, pair_list)):
        v1 = _eval_tree(t1, pset, n_features)
        if _in_mint_hemisphere(v1, mint, majt):
            continue
        new1, new2 = gp.cxOnePoint(copy.deepcopy(t1), copy.deepcopy(t2))
        c1[i], c2[i] = new1, new2
    return c1, c2


def _spherical_mutation(trees, psets, pair_list, n_features):
    result = [copy.deepcopy(t) for t in trees]
    for i, (tree, pset, (mint, majt)) in enumerate(
            zip(result, psets, pair_list)):
        v = _eval_tree(tree, pset, n_features)
        if _in_mint_hemisphere(v, mint, majt):
            continue
        expr_fn = lambda pset=pset, type_=None: gp.genFull(pset, min_=0, max_=2, type_=type_)
        mutant, = gp.mutUniform(copy.deepcopy(tree), expr=expr_fn, pset=pset)
        result[i] = mutant
    return result


def _elite_strategy(population, fitness_vals):
    n_trees = len(population[0])
    best_idx = int(np.argmax(fitness_vals))
    elite = []
    for j in range(n_trees):
        # 对每个位置找最优
        best_j = max(range(len(population)),
                     key=lambda i: _pos_fitness(population[i][j]))
        elite.append(copy.deepcopy(population[best_j][j]))
    return elite


def _pos_fitness(tree):
    """用树的节点数作为位置级别的代理适应度（真实适应度需要 pset/pair）"""
    return len(tree)   # 简化：实际论文按 Dis_Ang 选最优树


def _tournament_select(population, fitness_vals, k):
    idxs = random.sample(range(len(population)), min(k, len(population)))
    best = max(idxs, key=lambda i: fitness_vals[i])
    return [copy.deepcopy(t) for t in population[best]]


# ─────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────

class MTGPSMOTESampler(BaseEstimator):
    """
    MTGP-SMOTE 过采样器（兼容 scikit-learn API）。

    参数
    ----
    pop_size      : 种群大小（论文默认 512）
    n_generations : 进化代数（论文默认 100）
    cx_rate       : 交叉率（论文默认 0.7）
    mut_rate      : 变异率（论文默认 0.3）
    tournament_k  : 锦标赛大小（论文默认 7）
    max_depth     : GP 树最大深度（默认 4）
    random_state  : 随机种子
    verbose       : 是否打印进化进度
    res_only      : 是否只返回合成样本 X_syn（由原 mtgp_smote_res_only 合并而来）
                    False（默认）→ 返回 (X_res, y_res) 完整重采样数据集；
                    True         → 返回 X_syn（仅合成样本特征矩阵）。
    """

    def __init__(self, pop_size=512, n_generations=100,
                 cx_rate=0.7, mut_rate=0.3, tournament_k=7,
                 max_depth=4, random_state=None, verbose=False,
                 res_only=False):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.tournament_k = tournament_k
        self.max_depth = max_depth
        self.random_state = random_state
        self.verbose = verbose
        self.res_only = res_only

    def _build_pairs(self, X_min, X_maj, n_synthetic):
        n_min = len(X_min)
        n_majt = max(1, n_synthetic // n_min)
        km = KMeans(n_clusters=n_majt, random_state=self.random_state,
                    n_init=10)
        km.fit(X_maj)
        majt = km.cluster_centers_
        pairs = []
        for mint in X_min:
            for mj in majt:
                pairs.append((mint.copy(), mj.copy()))
                if len(pairs) >= n_synthetic:
                    return pairs
        return pairs

    def _make_individual(self, psets):
        return [
            gp.PrimitiveTree(gp.genHalfAndHalf(ps, min_=1,
                                                max_=self.max_depth))
            for ps in psets
        ]

    def _evolve(self, X_min, pair_list):
        n_trees = len(pair_list)
        n_features = X_min.shape[1]

        # 每棵树独立的 pset（唯一命名）
        psets = [_build_pset(X_min) for _ in range(n_trees)]

        # 初始化种群
        population = [self._make_individual(psets)
                      for _ in range(self.pop_size)]

        best_ind = [copy.deepcopy(t) for t in population[0]]
        best_fit = -1e18

        for gen in range(self.n_generations):
            # 评估
            fit_vals = [
                _eval_individual(ind, psets, pair_list, n_features)
                for ind in population
            ]

            # 更新全局最优
            for ind, fit in zip(population, fit_vals):
                if fit > best_fit:
                    best_fit = fit
                    best_ind = [copy.deepcopy(t) for t in ind]

            if self.verbose and gen % 10 == 0:
                print(f"    Gen {gen:3d}  best={best_fit:.4f}  "
                      f"avg={np.mean(fit_vals):.4f}")

            # 构造下一代
            new_pop = []

            # 精英个体（基于整体适应度选最优）
            best_idx = int(np.argmax(fit_vals))
            new_pop.append([copy.deepcopy(t)
                            for t in population[best_idx]])

            while len(new_pop) < self.pop_size:
                r = random.random()
                if r < self.cx_rate and len(new_pop) + 1 < self.pop_size:
                    p1 = _tournament_select(population, fit_vals,
                                            self.tournament_k)
                    p2 = _tournament_select(population, fit_vals,
                                            self.tournament_k)
                    c1, c2 = _spherical_crossover(
                        p1, p2, psets, pair_list, n_features)
                    new_pop.append(c1)
                    if len(new_pop) < self.pop_size:
                        new_pop.append(c2)
                else:
                    p = _tournament_select(population, fit_vals,
                                           self.tournament_k)
                    child = _spherical_mutation(
                        p, psets, pair_list, n_features)
                    new_pop.append(child)

            population = new_pop

        return best_ind, psets

    def fit_resample(self, X, y):
        """
        过采样，接口与 imbalanced-learn 兼容。

        Returns
        -------
        res_only=False（默认）→ (X_res, y_res) 完整重采样数据集；
        res_only=True         → X_syn 仅合成样本特征矩阵。
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        X = np.asarray(X, float)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("MTGPSMOTESampler 目前仅支持二分类。")

        majority_cls = classes[np.argmax(counts)]
        minority_cls = classes[np.argmin(counts)]
        X_maj = X[y == majority_cls]
        X_min = X[y == minority_cls]
        n_synthetic = len(X_maj) - len(X_min)

        if n_synthetic <= 0:
            # 已平衡，无需生成合成样本
            if self.res_only:
                return np.empty((0, X.shape[1]))
            return X.copy(), y.copy()

        pair_list = self._build_pairs(X_min, X_maj, n_synthetic)
        n_features = X_min.shape[1]

        best_ind, psets = self._evolve(X_min, pair_list)

        synthetic = [_eval_tree(t, ps, n_features)
                     for t, ps in zip(best_ind, psets)]

        X_syn = np.array(synthetic)
        y_syn = np.full(len(synthetic), minority_cls)

        if self.res_only:
            return X_syn
        return np.vstack([X, X_syn]), np.concatenate([y, y_syn])
