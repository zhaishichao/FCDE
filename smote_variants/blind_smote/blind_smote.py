"""
BlindSMOTE: Synthetic minority oversampling based only on evolutionary computation
================================================================================
完整实现 —— 基于 scikit-learn + NumPy，内置 DEAP 风格进化框架

参考:
    García-Pedrajas et al., "BlindSMOTE: Synthetic minority oversampling
    based only on evolutionary computation", Evolutionary Computation, 2025.
    https://doi.org/10.1162/evco_a_00374

代码结构:
    Part 1  GeneticEngine 工具类（HallOfFame、Statistics）
            —— 对应 deap.tools 核心功能
    Part 2  Individual —— 封装四部分染色体: N, nn, R, s
    Part 3  工具函数（G-mean、适应度）
    Part 4  BlindSMOTE —— scikit-learn 风格重采样器（Algorithm 1 + 2）
    Part 5  演示 __main__
"""

from __future__ import annotations
import warnings
import time
import numpy as np
from copy import deepcopy
from typing import List, Optional, Tuple

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.utils import check_random_state

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  Part 1 ── 轻量级 DEAP 风格工具（HallOfFame / Statistics / FitnessMax）
# ══════════════════════════════════════════════════════════════════════════════

class FitnessMax:
    """
    单目标最大化适应度容器。
    对应 DEAP:  creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    """
    weights = (1.0,)

    def __init__(self, value: float = 0.0):
        self.values = (value,)

    @property
    def value(self) -> float:
        return self.values[0]

    def __gt__(self, other): return self.value > other.value
    def __ge__(self, other): return self.value >= other.value
    def __lt__(self, other): return self.value < other.value
    def __repr__(self): return f"FitnessMax({self.value:.6f})"


class HallOfFame:
    """
    保存进化过程中出现过的最优 maxsize 个个体。
    对应 deap.tools.HallOfFame。
    """

    def __init__(self, maxsize: int = 1):
        self.maxsize = maxsize
        self._items: List = []
        self._fits: List[float] = []

    def update(self, population: List, fitness_list: List[float]):
        for ind, fit in zip(population, fitness_list):
            if len(self._items) < self.maxsize or fit > min(self._fits):
                self._items.append(deepcopy(ind))
                self._fits.append(fit)
                paired = sorted(zip(self._fits, self._items), reverse=True)
                self._fits = [p[0] for p in paired[:self.maxsize]]
                self._items = [p[1] for p in paired[:self.maxsize]]

    @property
    def best(self):
        return self._items[0] if self._items else None

    @property
    def best_fitness(self) -> float:
        return self._fits[0] if self._fits else 0.0


class Statistics:
    """
    收集每代统计信息。
    对应 deap.tools.Statistics。
    """

    def __init__(self):
        self.history: List[dict] = []

    def record(self, gen: int, fitness_list: List[float]):
        arr = np.array(fitness_list)
        self.history.append({
            "gen": gen,
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "std": float(arr.std()),
        })

    def logbook_str(self, last_n: int = 0) -> str:
        data = self.history[-last_n:] if last_n > 0 else self.history
        lines = ["gen   max     mean    min     std"]
        lines.append("-" * 40)
        for r in data:
            lines.append(
                f"{r['gen']:5d} {r['max']:.4f}  {r['mean']:.4f}"
                f"  {r['min']:.4f}  {r['std']:.4f}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Part 2 ── 个体 (Individual / Chromosome)
# ══════════════════════════════════════════════════════════════════════════════

class Individual:
    """
    BlindSMOTE 染色体，由四部分组成（论文 Table 1）:

        N   : int  — 每个少数类样本生成的合成样本数，N ∈ [N_min, N_max]

        nn  : ndarray (n_min, N)
              邻居索引矩阵，值域 [-k, k] \\ {0}。
              负值表示该对 (xi, xj) 不生成合成样本（symmetric evolution）。

        R   : List[ndarray (N, m)]，长度 n_min
              插值权重矩阵，值域 [0,1]。
              新实例 x_n = x_i + R[i][j] ⊙ (x_j - x_i)

        s   : ndarray (n_maj + n_min*N,)  dtype int8
              二值选择向量:
                s[:n_maj]  → 多数类样本是否保留（下采样）
                s[n_maj:]  → 合成样本是否保留（合成样本筛选）

    fitness : FitnessMax — 个体对应的适应度值
    """

    __slots__ = ("N", "nn", "R", "s", "fitness")

    def __init__(self, N: int, nn: np.ndarray,
                 R: List[np.ndarray], s: np.ndarray):
        self.N = N
        self.nn = nn
        self.R = R
        self.s = s
        self.fitness = FitnessMax()

    def clone(self) -> "Individual":
        c = Individual(
            self.N,
            self.nn.copy(),
            [r.copy() for r in self.R],
            self.s.copy(),
        )
        c.fitness = FitnessMax(self.fitness.value)
        return c

    def __repr__(self):
        return (f"Individual(N={self.N}, nn_shape={self.nn.shape}, "
                f"fitness={self.fitness})")


# ══════════════════════════════════════════════════════════════════════════════
#  Part 3 ── 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def gmean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """G-mean = sqrt(sensitivity × specificity)"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(np.sqrt(sn * sp))


def combined_fitness(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """论文适应度 = (G-mean + F1) / 2"""
    gm = gmean_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return float((gm + f1) / 2.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Part 4 ── BlindSMOTE 主类
# ══════════════════════════════════════════════════════════════════════════════

class BlindSMOTE(BaseEstimator, TransformerMixin):
    """
    BlindSMOTE 重采样器（完整实现 Algorithm 1 + Algorithm 2）。

    Parameters
    ----------
    k : int, default=5
        固定超参数：生成合成样本时考虑的近邻数（论文 k=5）。
    N_min : int, default=1
        每个少数类样本最少生成的合成样本数（进化变量下界）。
    N_max : int, default=5
        每个少数类样本最多生成的合成样本数（进化变量上界）。
    pop_size : int, default=100
        种群大小（论文 100）。
    n_gen : int, default=10000
        最大进化代数（论文 10000）。
    cx_prob : float, default=0.8
        交叉概率。
    mut_prob : float, default=0.05
        变异概率（论文 5%）。
    mut_bit_rate : float, default=0.01
        位变异率（论文 1%）。
    elitism_ratio : float, default=0.1
        精英保留比例（论文 top 10%）。
    stagnation_gens : int, default=100
        最优适应度连续无改善的代数阈值，超过后重置最差 10% 个体（论文 100 代）。
    classifier : sklearn estimator or None
        Wrapper 分类器（论文使用 SVM / C4.5 / RF，默认 DecisionTreeClassifier）。
    time_limit : float or None
        运行时间上限（秒），None 表示不限。
    random_state : int or None
        随机种子，保证可复现。
    verbose : bool, default=False
        是否打印进化日志。
    res_only : bool, default=False
        是否额外返回合成样本列表（由原 blind_smote_res_only 模块合并而来）。
        False（默认）→ fit_resample 返回 (X_res, y_res)；
        True         → fit_resample 返回 (X_res, y_res, synth_rows)，
                        其中 synth_rows 为合成样本特征向量的列表。
    """

    def __init__(
        self,
        k: int = 5,
        N_min: int = 1,
        N_max: int = 5,
        pop_size: int = 100,
        n_gen: int = 10000,
        cx_prob: float = 0.8,
        mut_prob: float = 0.05,
        mut_bit_rate: float = 0.01,
        elitism_ratio: float = 0.10,
        stagnation_gens: int = 100,
        classifier=None,
        time_limit: Optional[float] = None,
        random_state=None,
        verbose: bool = False,
        res_only: bool = False,
    ):
        self.k = k
        self.N_min = N_min
        self.N_max = N_max
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.mut_bit_rate = mut_bit_rate
        self.elitism_ratio = elitism_ratio
        self.stagnation_gens = stagnation_gens
        self.classifier = classifier
        self.time_limit = time_limit
        self.random_state = random_state
        self.verbose = verbose
        self.res_only = res_only

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _build_knn(self):
        """计算少数类内部 k 近邻，排除自身。返回 (n_min, k_) 索引矩阵。"""
        nbrs = NearestNeighbors(n_neighbors=self.k_ + 1).fit(self.X_min_)
        _, idx = nbrs.kneighbors(self.X_min_)
        return idx[:, 1:]   # 排除第 0 列（自身）

    def _nn_choices(self) -> list:
        """[-k_, ..., -1, 1, ..., k_]"""
        return list(range(-self.k_, 0)) + list(range(1, self.k_ + 1))

    # ── 个体初始化 ────────────────────────────────────────────────────────

    def _init_individual(self, rng: np.random.RandomState) -> Individual:
        """
        论文 §3：随机初始化种群中的一个个体。
          - N ∈ [N_min, N_max] 均匀随机
          - nn ∈ [-k,k]\\{0} 均匀随机
          - R  ∈ [0,1]^(N×m) 均匀随机
          - s  = 全 1（初始选择全部）
        """
        n_min, m = self.X_min_.shape
        N = int(rng.randint(self.N_min, self.N_max + 1))
        choices = self._nn_choices()
        nn = np.array(rng.choice(choices, size=(n_min, N)), dtype=np.int32)
        R = [rng.uniform(0, 1, (N, m)).astype(np.float32) for _ in range(n_min)]
        s = np.ones(self.n_maj_ + n_min * N, dtype=np.int8)
        return Individual(N, nn, R, s)

    # ── Algorithm 1：从个体解码出增强训练集 ──────────────────────────────

    def _decode(self, ind: Individual) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Algorithm 1: BlindSMOTE procedure to obtain dataset T^A from individual i.

        [1]  T^A = T                             (复制原始训练集)
        [2]  T^A -= { x_j : s_j=0 ∧ y_j=0 }    (多数类下采样)
        [3]  φ = {x_j ∈ T | y_j=1}              (少数类集合)
        [4]  p = n+1                             (新样本追加指针)
        foreach x_i ∈ φ:
          for j=1..N:
        [5]    x_j = nn[i,j]                    (取邻居)
        [6]    x_p = x_i + R[i][j] ⊙ (x_j-x_i)(插值生成)
        [7]    p++
        [8]  T^A -= { x_j : s_j=0 ∧ j>n }      (合成样本筛选)
        [9]  return T^A
        """
        X_min, X_maj = self.X_min_, self.X_maj_
        knn = self.knn_idx_
        n_min = X_min.shape[0]
        N = ind.N

        # [2] 多数类下采样
        s_maj = ind.s[:self.n_maj_]
        X_maj_sel = X_maj[s_maj == 1]

        # [3-7] 生成合成样本（使用 ind.nn 的实际列数，与 N 保持一致）
        N_actual = ind.nn.shape[1]   # 防御性取实际列数
        s_synth = ind.s[self.n_maj_:]
        synth_rows = []
        ptr = 0
        for i in range(n_min):
            xi = X_min[i]
            for j in range(N_actual):
                nn_ij = int(ind.nn[i, j])
                keep = (ptr < len(s_synth)) and (s_synth[ptr] == 1)
                ptr += 1
                if nn_ij < 0 or not keep:
                    # 负值（symmetric evolution）或未被 s 选中：跳过
                    continue
                nb = min(abs(nn_ij) - 1, self.k_ - 1)   # 0-based 索引
                xj = X_min[knn[i, nb]]
                xn = xi + ind.R[i][j] * (xj - xi)       # 公式 (2)
                synth_rows.append(xn)

        # [9] 拼装 T^A
        X_parts = [X_min, X_maj_sel]
        y_parts = [
            np.full(n_min, self.min_class_, dtype=int),
            np.full(len(X_maj_sel), self.maj_class_, dtype=int),
        ]
        if synth_rows:
            X_parts.append(np.array(synth_rows, dtype=np.float32))
            y_parts.append(np.full(len(synth_rows), self.min_class_, dtype=int))

        return np.vstack(X_parts), np.concatenate(y_parts), synth_rows

    # ── 适应度评估（论文 §3，公式 3-8）──────────────────────────────────

    def _evaluate(self, ind: Individual) -> float:
        """
        1. 解码 ind → (X_aug, y_aug)
        2. 用 X_aug 训练 wrapper 分类器
        3. 在原始训练集 (X_, y_) 上预测，计算 (G-mean + F1) / 2
        """
        X_aug, y_aug, _ = self._decode(ind)
        if len(np.unique(y_aug)) < 2:
            return 0.0
        clf = deepcopy(self.clf_)
        try:
            clf.fit(X_aug, y_aug)
            y_pred = clf.predict(self.X_)
            return combined_fitness(self.y_, y_pred)
        except Exception:
            return 0.0

    # ── 遗传算子 ──────────────────────────────────────────────────────────

    def _crossover(
        self,
        p1: Individual,
        p2: Individual,
        rng: np.random.RandomState,
    ) -> Tuple[Individual, Individual]:
        """
        论文 §3 中的两种交叉方案（等概率选择）:

        N:  50% 直接交换 | 50% BLX-α (α=0.5)

        nn + R（方案 A）: 独立列均匀交叉 nn，BLX-α 交叉 R
        nn + R（方案 B）: 交换 nn[i,j] 时同时携带对应 R[i][j]
                         （即整个合成样本一起交换）

        s:  HUX 风格——仅随机交换一半不同位
        """
        c1, c2 = p1.clone(), p2.clone()
        use_b = rng.random() < 0.5

        # —— N ——
        if rng.random() < 0.5:
            c1.N, c2.N = c2.N, c1.N
        else:
            lo, hi = float(min(c1.N, c2.N)), float(max(c1.N, c2.N))
            sp = (hi - lo) * 0.5
            c1.N = int(np.clip(rng.uniform(lo - sp, hi + sp), self.N_min, self.N_max))
            c2.N = int(np.clip(rng.uniform(lo - sp, hi + sp), self.N_min, self.N_max))

        n_min = self.X_min_.shape[0]
        N_cols = min(p1.nn.shape[1], p2.nn.shape[1])
        m = self.X_min_.shape[1]

        for i in range(n_min):
            for j in range(N_cols):
                if rng.random() < 0.5:
                    j1 = j % c1.nn.shape[1]
                    j2 = j % c2.nn.shape[1]
                    c1.nn[i, j1], c2.nn[i, j2] = int(c2.nn[i, j2]), int(c1.nn[i, j1])
                    if use_b:
                        # 方案 B：同时交换对应 R 行
                        r1j = j1 % c1.R[i].shape[0]
                        r2j = j2 % c2.R[i].shape[0]
                        tmp = c1.R[i][r1j].copy()
                        c1.R[i][r1j] = c2.R[i][r2j]
                        c2.R[i][r2j] = tmp

            if not use_b:
                # 方案 A：BLX-α 交叉 R
                Ni = min(c1.R[i].shape[0], c2.R[i].shape[0])
                alpha = 0.5
                lo_r = np.minimum(c1.R[i][:Ni], c2.R[i][:Ni])
                hi_r = np.maximum(c1.R[i][:Ni], c2.R[i][:Ni])
                span = (hi_r - lo_r) * alpha
                noise1 = rng.uniform(0, 1, (Ni, m)).astype(np.float32)
                noise2 = rng.uniform(0, 1, (Ni, m)).astype(np.float32)
                c1.R[i][:Ni] = np.clip(lo_r - span + noise1 * (hi_r - lo_r + 2 * span), 0, 1)
                c2.R[i][:Ni] = np.clip(lo_r - span + noise2 * (hi_r - lo_r + 2 * span), 0, 1)

        # —— s：HUX ——
        min_len = min(len(c1.s), len(c2.s))
        diff = np.where(c1.s[:min_len] != c2.s[:min_len])[0]
        if len(diff) > 1:
            swap = rng.choice(diff, size=len(diff) // 2, replace=False)
            for idx in swap:
                c1.s[idx], c2.s[idx] = c2.s[idx], c1.s[idx]

        # —— 交叉结束后，把 nn 和 R 的列数对齐到新的 N ——
        for c in (c1, c2):
            new_N = c.N
            old_cols = c.nn.shape[1]
            if new_N != old_cols:
                n_min_local = c.nn.shape[0]
                choices_local = self._nn_choices()
                if new_N > old_cols:
                    # 补列：随机采样
                    extra = new_N - old_cols
                    extra_nn = np.array(
                        rng.choice(choices_local, size=(n_min_local, extra)),
                        dtype=np.int32)
                    c.nn = np.hstack([c.nn, extra_nn])
                    for i in range(n_min_local):
                        extra_R = rng.uniform(0, 1, (extra, self.X_min_.shape[1])).astype(np.float32)
                        c.R[i] = np.vstack([c.R[i], extra_R])
                else:
                    # 截断列
                    c.nn = c.nn[:, :new_N]
                    for i in range(n_min_local):
                        c.R[i] = c.R[i][:new_N]
            # 更新 s 长度
            n_min_local = c.nn.shape[0]
            needed_s = self.n_maj_ + n_min_local * c.N
            if len(c.s) < needed_s:
                c.s = np.concatenate([c.s, np.ones(needed_s - len(c.s), dtype=np.int8)])
            elif len(c.s) > needed_s:
                c.s = c.s[:needed_s]

        return c1, c2

    def _mutate(
        self,
        ind: Individual,
        rng: np.random.RandomState,
    ) -> Individual:
        """
        论文 §3.1 变异算子：随机选择一个部分进行变异。

        N  : 非均匀变异（±1 或小幅高斯扰动）
        nn : 50% 非均匀（值域内±1 调整）| 50% 随机（替换或翻转符号）
        R  : 非均匀变异（小幅高斯噪声，clamp 到 [0,1]）
        s  : 随机位翻转（概率 mut_bit_rate）
        """
        part = rng.randint(4)
        choices = self._nn_choices()

        if part == 0:                               # N
            old_N = ind.N
            if rng.random() < 0.5:
                ind.N = int(np.clip(ind.N + rng.choice([-1, 1]),
                                    self.N_min, self.N_max))
            else:
                ind.N = int(np.clip(
                    ind.N + int(np.round(rng.normal(0, 0.8))),
                    self.N_min, self.N_max))
            new_N = ind.N
            if new_N != old_N:
                n_min_local = ind.nn.shape[0]
                m_local = self.X_min_.shape[1]
                choices_local = self._nn_choices()
                if new_N > old_N:
                    extra = new_N - old_N
                    extra_nn = np.array(rng.choice(choices_local,
                                                   size=(n_min_local, extra)),
                                        dtype=np.int32)
                    ind.nn = np.hstack([ind.nn, extra_nn])
                    for i in range(n_min_local):
                        extra_R = rng.uniform(0, 1, (extra, m_local)).astype(np.float32)
                        ind.R[i] = np.vstack([ind.R[i], extra_R])
                else:
                    ind.nn = ind.nn[:, :new_N]
                    for i in range(n_min_local):
                        ind.R[i] = ind.R[i][:new_N]
                needed_s = self.n_maj_ + n_min_local * new_N
                if len(ind.s) < needed_s:
                    ind.s = np.concatenate([ind.s, np.ones(needed_s - len(ind.s), dtype=np.int8)])
                else:
                    ind.s = ind.s[:needed_s]

        elif part == 1:                             # nn
            mask = rng.random(ind.nn.shape) < self.mut_bit_rate
            if not mask.any():
                return ind
            if rng.random() < 0.5:                 # 非均匀
                delta = rng.choice([-1, 0, 1], size=ind.nn.shape)
                new_nn = (ind.nn + mask * delta).astype(np.int32)
                new_nn = np.where(new_nn == 0, 1, new_nn)
                ind.nn = np.clip(new_nn, -self.k_, self.k_)
            else:                                   # 随机
                for i in range(ind.nn.shape[0]):
                    for j in range(ind.nn.shape[1]):
                        if mask[i, j]:
                            if rng.random() < 0.5:
                                ind.nn[i, j] = rng.choice(choices)
                            else:
                                ind.nn[i, j] = -ind.nn[i, j]

        elif part == 2:                             # R
            for i in range(len(ind.R)):
                m_mask = rng.random(ind.R[i].shape) < self.mut_bit_rate
                if m_mask.any():
                    noise = rng.normal(0, 0.05, ind.R[i].shape).astype(np.float32)
                    ind.R[i] = np.clip(ind.R[i] + m_mask.astype(np.float32) * noise, 0, 1)

        else:                                       # s
            flip = rng.random(ind.s.shape) < self.mut_bit_rate
            ind.s = np.where(flip, (1 - ind.s).astype(np.int8), ind.s)

        return ind

    # ── Algorithm 2：主进化循环 ───────────────────────────────────────────

    def _run_ga(self, rng: np.random.RandomState) -> Individual:
        """
        Algorithm 2: BlindSMOTE genetic algorithm.

        [1]  随机初始化种群
        [2-5] 评估初始种群，计算适应度
        while #gen < G and runtime < tl:
          [6]  HUX 交叉，种群扩至 2S
          [7-10] 评估新个体
          [11] μ+λ 精英选择，保留最优 S 个
          [12-13] 变异
          [14] 停滞检测，重新初始化最差 10%
        [15] 返回最优个体
        """
        t_start = time.time()
        n_min = self.X_min_.shape[0]
        hof = HallOfFame(maxsize=1)
        stats = Statistics()

        # [1] 初始化种群
        pop = [self._init_individual(rng) for _ in range(self.pop_size)]

        # [2-5] 评估初始种群
        fit_list = [self._evaluate(ind) for ind in pop]
        for ind, f in zip(pop, fit_list):
            ind.fitness = FitnessMax(f)
        hof.update(pop, fit_list)
        stats.record(0, fit_list)

        log_interval = max(1, self.n_gen // 20)
        if self.verbose:
            print(f"  gen={0:5d}  best={hof.best_fitness:.4f}"
                  f"  mean={np.mean(fit_list):.4f}")

        stagnation = 0

        for gen in range(1, self.n_gen + 1):

            # 时间限制
            if self.time_limit and (time.time() - t_start) > self.time_limit:
                if self.verbose:
                    print(f"  达到时间限制 {self.time_limit}s，停止于 gen={gen}")
                break

            # [6] 交叉，产生 pop_size 个后代
            idx = rng.permutation(self.pop_size)
            offspring = []
            for i in range(0, self.pop_size, 2):
                pa = pop[idx[i]]
                pb = pop[idx[min(i + 1, self.pop_size - 1)]]
                if rng.random() < self.cx_prob:
                    c1, c2 = self._crossover(pa, pb, rng)
                else:
                    c1, c2 = pa.clone(), pb.clone()
                offspring.extend([c1, c2])
            offspring = offspring[:self.pop_size]

            # [12-13] 变异
            for ind in offspring:
                if rng.random() < self.mut_prob:
                    self._mutate(ind, rng)

            # [7-10] 评估后代
            off_fit = [self._evaluate(ind) for ind in offspring]
            for ind, f in zip(offspring, off_fit):
                ind.fitness = FitnessMax(f)

            # [11] μ+λ 精英选择
            combined = list(zip(pop + offspring, fit_list + off_fit))
            combined.sort(key=lambda x: x[1], reverse=True)
            pop = [c[0] for c in combined[:self.pop_size]]
            fit_list = [c[1] for c in combined[:self.pop_size]]

            prev_best = hof.best_fitness
            hof.update(pop, fit_list)
            stagnation = 0 if hof.best_fitness > prev_best else stagnation + 1

            # [14] 停滞：重新初始化最差 10%
            if stagnation >= self.stagnation_gens:
                n_reinit = max(1, int(self.pop_size * 0.10))
                for i in range(self.pop_size - n_reinit, self.pop_size):
                    pop[i] = self._init_individual(rng)
                    fit_list[i] = self._evaluate(pop[i])
                    pop[i].fitness = FitnessMax(fit_list[i])
                stagnation = 0
                if self.verbose:
                    print(f"  gen={gen:5d}  [停滞] 重置最差 {n_reinit} 个体")

            stats.record(gen, fit_list)
            if self.verbose and gen % log_interval == 0:
                print(f"  gen={gen:5d}  best={hof.best_fitness:.4f}"
                      f"  mean={np.mean(fit_list):.4f}"
                      f"  std={np.std(fit_list):.4f}")

        self.hof_ = hof
        self.stats_ = stats
        return hof.best

    # ── 公共接口 ──────────────────────────────────────────────────────────

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        对 (X, y) 执行 BlindSMOTE 进化重采样（二分类）。

        Returns
        -------
        X_res : ndarray — 重采样后的特征矩阵
        y_res : ndarray — 重采样后的标签向量
        synth_rows : list — 合成样本特征向量列表（仅 res_only=True 时返回）
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)

        # 随机数生成器
        base_rng = check_random_state(self.random_state)
        seed = int(base_rng.randint(0, 2**31))
        rng = np.random.RandomState(seed)

        # Wrapper 分类器
        self.clf_ = (deepcopy(self.classifier)
                     if self.classifier is not None
                     else DecisionTreeClassifier(random_state=0))

        # 识别少数类 / 多数类
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("BlindSMOTE 当前仅支持二分类问题。")
        self.min_class_ = int(classes[np.argmin(counts)])
        self.maj_class_ = int(classes[np.argmax(counts)])

        self.X_ = X
        self.y_ = y
        self.X_min_ = X[y == self.min_class_]
        self.X_maj_ = X[y == self.maj_class_]
        self.n_maj_ = int(self.X_maj_.shape[0])

        n_min = self.X_min_.shape[0]
        if n_min <= 1:
            warnings.warn("少数类样本数 ≤ 1，直接返回原始数据。")
            return (X, y, []) if self.res_only else (X, y)

        # 构建 k 近邻（固定超参数）
        self.k_ = min(self.k, n_min - 1)
        self.knn_idx_ = self._build_knn()   # (n_min, k_)

        if self.verbose:
            ir = self.n_maj_ / n_min
            print(f"[BlindSMOTE] 少数类={n_min}, 多数类={self.n_maj_}, "
                  f"IR={ir:.2f}, k={self.k_}")
            print(f"             种群={self.pop_size}, 代数={self.n_gen}, "
                  f"N∈[{self.N_min},{self.N_max}]")
            print()

        # 运行进化算法（Algorithm 2）
        best_ind = self._run_ga(rng)
        self.best_individual_ = best_ind

        # 解码最优个体 → 增强训练集
        X_res, y_res, synth_rows = self._decode(best_ind)

        if self.verbose:
            print(f"\n[BlindSMOTE] 完成。"
                  f"增强后: 多数类={np.sum(y_res == self.maj_class_)}, "
                  f"少数类={np.sum(y_res == self.min_class_)}")
            print(f"             最优适应度: {self.hof_.best_fitness:.4f}")

        return (X_res, y_res, synth_rows) if self.res_only else (X_res, y_res)

    def print_logbook(self, last_n: int = 20):
        """打印进化历史（最后 last_n 代）"""
        if hasattr(self, "stats_"):
            print(self.stats_.logbook_str(last_n=last_n))
        else:
            print("尚未运行 fit_resample。")


# ══════════════════════════════════════════════════════════════════════════════
#  Part 5 ── 演示
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    print("=" * 65)
    print("  BlindSMOTE 演示 (基于 García-Pedrajas et al., 2025)")
    print("=" * 65)

    # ── 生成不均衡数据集 ──
    X, y = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        weights=[0.85, 0.15],
        flip_y=0.01,
        random_state=42,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"训练集: 多数类={np.sum(y_tr==0)}, 少数类={np.sum(y_tr==1)}, "
          f"IR={np.sum(y_tr==0)/np.sum(y_tr==1):.1f}")

    # ── 基线 ──
    clf_base = DecisionTreeClassifier(random_state=0)
    clf_base.fit(X_tr, y_tr)
    y_pred_base = clf_base.predict(X_te)
    gm_base = gmean_score(y_te, y_pred_base)
    f1_base = f1_score(y_te, y_pred_base, zero_division=0)

    print("\n── 基线（无重采样，DecisionTree）──")
    print(classification_report(y_te, y_pred_base,
                                target_names=["Majority", "Minority"]))

    # ── BlindSMOTE ──
    print("── BlindSMOTE 进化重采样 ──")
    bs = BlindSMOTE(
        k=5,
        N_min=1, N_max=5,
        pop_size=30,        # 演示用小种群，论文建议 100
        n_gen=100,           # 演示用少代数，论文建议 10000
        cx_prob=0.8,
        mut_prob=0.05,
        mut_bit_rate=0.02,
        elitism_ratio=0.10,
        stagnation_gens=25,
        classifier=DecisionTreeClassifier(random_state=0),
        time_limit=180,
        random_state=42,
        verbose=True,
    )

    t0 = time.time()
    X_res, y_res = bs.fit_resample(X_tr, y_tr)
    elapsed = time.time() - t0

    clf_bs = DecisionTreeClassifier(random_state=0)
    clf_bs.fit(X_res, y_res)
    y_pred_bs = clf_bs.predict(X_te)
    gm_bs = gmean_score(y_te, y_pred_bs)
    f1_bs = f1_score(y_te, y_pred_bs, zero_division=0)

    print(f"\n增强后: 多数类={np.sum(y_res==0)}, 少数类={np.sum(y_res==1)}")
    print(f"总耗时: {elapsed:.1f}s\n")

    print("── BlindSMOTE 结果 ──")
    print(classification_report(y_te, y_pred_bs,
                                target_names=["Majority", "Minority"]))

    print("┌──────────────┬─────────┬───────────┐")
    print("│  指标        │  基线   │ BlindSMOTE│")
    print("├──────────────┼─────────┼───────────┤")
    print(f"│  G-mean      │ {gm_base:.4f}  │  {gm_bs:.4f}   │")
    print(f"│  F1 (少数类) │ {f1_base:.4f}  │  {f1_bs:.4f}   │")
    print("└──────────────┴─────────┴───────────┘")

    print("\n── 进化历史（最后 10 代）──")
    bs.print_logbook(last_n=10)
