"""
BlindSMOTE: Synthetic minority oversampling based only on evolutionary computation
================================================================================
严格遵照论文原文实现，基于 scikit-learn + NumPy，内置 DEAP 风格遗传引擎。

参考:
    García-Pedrajas et al., "BlindSMOTE: Synthetic minority oversampling
    based only on evolutionary computation", Evolutionary Computation, 2025.
    https://doi.org/10.1162/evco_a_00374

适应度评估（论文 §3，Algorithm 2 第 3-5 行）：
    用增强集 T^A 训练分类器，在原始训练集 T 上评估 (G-mean + F1) / 2。
    论文原文如此，本实现忠实复现，不做额外修改。

代码结构:
    Part 1  DEAP 风格工具类  —— FitnessMax / HallOfFame / Statistics
    Part 2  Individual       —— 四部分染色体: N, nn, R, s
    Part 3  工具函数         —— gmean_score / combined_fitness
    Part 4  BlindSMOTE       —— 重采样器主类 (Algorithm 1 + Algorithm 2)
    Part 5  演示 __main__
"""

from __future__ import annotations

import time
import warnings
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  Part 1 ── DEAP 风格工具类
# ══════════════════════════════════════════════════════════════════════════════

class FitnessMax:
    """
    单目标最大化适应度。
    对应 DEAP: creator.create('FitnessMax', base.Fitness, weights=(1.0,))
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
    保存历史最优 maxsize 个个体。
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
                self._fits  = [p[0] for p in paired[:self.maxsize]]
                self._items = [p[1] for p in paired[:self.maxsize]]

    @property
    def best(self):
        return self._items[0] if self._items else None

    @property
    def best_fitness(self) -> float:
        return self._fits[0] if self._fits else 0.0


class Statistics:
    """
    按代记录种群适应度统计信息。
    对应 deap.tools.Statistics。
    """

    def __init__(self):
        self.history: List[dict] = []

    def record(self, gen: int, fitness_list: List[float]):
        arr = np.array(fitness_list)
        self.history.append({
            "gen":  gen,
            "max":  float(arr.max()),
            "mean": float(arr.mean()),
            "min":  float(arr.min()),
            "std":  float(arr.std()),
        })

    def logbook_str(self, last_n: int = 0) -> str:
        rows = self.history[-last_n:] if last_n > 0 else self.history
        lines = ["  gen    max     mean    min     std",
                 "  " + "-" * 38]
        for r in rows:
            lines.append(f"  {r['gen']:4d}  {r['max']:.4f}  {r['mean']:.4f}"
                         f"  {r['min']:.4f}  {r['std']:.4f}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Part 2 ── Individual（论文 Table 1）
# ══════════════════════════════════════════════════════════════════════════════

class Individual:
    """
    BlindSMOTE 染色体，由四部分组成（论文 Table 1）：

    N   : int
          每个少数类样本生成的合成样本数，N ∈ [N_min, N_max]（进化变量）。

    nn  : ndarray, shape (n_min, N), dtype int32
          邻居索引矩阵，值域 [-k, k] \\ {0}。
          负值 → 该对 (x_i, x_j) 不生成合成样本（symmetric evolution）。

    R   : List[ndarray (N, m)], 长度 n_min
          插值权重矩阵，值域 [0, 1]。
          生成公式（论文公式 2）：x_nl = x_il + R[i][j,l] * (x_jl - x_il)

    s   : ndarray, shape (n_maj + n_min * N,), dtype int8
          二值选择向量：
            s[:n_maj]  → 多数类样本是否保留（多数类下采样）
            s[n_maj:]  → 合成样本是否保留（合成样本筛选）

    fitness : FitnessMax
    """

    __slots__ = ("N", "nn", "R", "s", "fitness")

    def __init__(self, N: int, nn: np.ndarray,
                 R: List[np.ndarray], s: np.ndarray):
        self.N  = N
        self.nn = nn
        self.R  = R
        self.s  = s
        self.fitness = FitnessMax()

    def clone(self) -> "Individual":
        c = Individual(self.N, self.nn.copy(),
                       [r.copy() for r in self.R], self.s.copy())
        c.fitness = FitnessMax(self.fitness.value)
        return c

    def __repr__(self):
        return (f"Individual(N={self.N}, nn={self.nn.shape}, "
                f"fitness={self.fitness})")


# ══════════════════════════════════════════════════════════════════════════════
#  Part 3 ── 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def gmean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """G-mean = sqrt(sensitivity × specificity)，论文公式 (7)"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(np.sqrt(sn * sp))


def combined_fitness(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """论文适应度 = (G-mean + F1) / 2，Algorithm 2 第 5/10 行"""
    gm = gmean_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return float((gm + f1) / 2.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Part 4 ── BlindSMOTE
# ══════════════════════════════════════════════════════════════════════════════

class BlindSMOTE(BaseEstimator, TransformerMixin):
    """
    BlindSMOTE 重采样器，严格复现论文 Algorithm 1 + Algorithm 2。

    适应度评估（论文原文做法）
    --------------------------
    用增强集 T^A 训练分类器，在完整原始训练集 T 上评估 (G-mean + F1) / 2。
    Algorithm 2 第 3-5 行：
        [3] 用 T^A_i 训练得到模型 h(x)
        [4] 在 T 上评估 G-mean_i 和 F1_i
        [5] fitness_i = (G_i + F_i) / 2

    Parameters
    ----------
    k : int, default=5
        固定超参数：生成合成样本时考虑的近邻数（论文 k=5）。
    N_min : int, default=1
        N 的进化下界（论文 N ∈ [1, 10]）。
    N_max : int, default=10
        N 的进化上界。
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
    stagnation_gens : int, default=100
        最优适应度连续无改善代数阈值，超过后重置最差 10% 个体（论文 100 代）。
    cv_folds : int, default=3
        适应度评估使用的分层 k 折交叉验证折数。
        每折在不含验证集样本的增强集上训练，在留出验证折上评估，无信息泄露。
    classifier : sklearn estimator or None
        Wrapper 分类器，默认 DecisionTreeClassifier（近似 C4.5）。
    time_limit : float or None
        运行时间上限（秒），None 表示不限。
    random_state : int or None
        随机种子。
    verbose : bool, default=False
        打印进化日志。
    """

    def __init__(
        self,
        k: int = 5,
        N_min: int = 1,
        N_max: int = 10,
        pop_size: int = 100,
        n_gen: int = 10000,
        cx_prob: float = 0.8,
        mut_prob: float = 0.05,
        mut_bit_rate: float = 0.01,
        stagnation_gens: int = 100,
        cv_folds: int = 3,
        classifier=None,
        time_limit: Optional[float] = None,
        random_state=None,
        verbose: bool = False,
    ):
        self.k               = k
        self.N_min           = N_min
        self.N_max           = N_max
        self.pop_size        = pop_size
        self.n_gen           = n_gen
        self.cx_prob         = cx_prob
        self.mut_prob        = mut_prob
        self.mut_bit_rate    = mut_bit_rate
        self.stagnation_gens = stagnation_gens
        self.cv_folds        = cv_folds
        self.classifier      = classifier
        self.time_limit      = time_limit
        self.random_state    = random_state
        self.verbose         = verbose

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _nn_choices(self) -> list:
        """[-k_, ..., -1, 1, ..., k_]  （symmetric evolution 的取值域）"""
        return list(range(-self.k_, 0)) + list(range(1, self.k_ + 1))

    def _resize_individual(self, ind: Individual,
                           new_N: int,
                           rng: np.random.RandomState) -> Individual:
        """
        当 N 发生变化时，同步调整 nn、R、s 的尺寸。
          new_N > old_N → 补充随机列/行
          new_N < old_N → 截断
        """
        old_N   = ind.nn.shape[1]
        n_min   = ind.nn.shape[0]
        m       = self.X_min_.shape[1]
        choices = self._nn_choices()

        if new_N > old_N:
            extra = new_N - old_N
            extra_nn = np.array(
                rng.choice(choices, size=(n_min, extra)), dtype=np.int32)
            ind.nn = np.hstack([ind.nn, extra_nn])
            for i in range(n_min):
                extra_R = rng.uniform(0, 1, (extra, m)).astype(np.float32)
                ind.R[i] = np.vstack([ind.R[i], extra_R])
        elif new_N < old_N:
            ind.nn = ind.nn[:, :new_N]
            for i in range(n_min):
                ind.R[i] = ind.R[i][:new_N]

        # s 尺寸对齐
        needed = self.n_maj_ + n_min * new_N
        cur    = len(ind.s)
        if needed > cur:
            ind.s = np.concatenate(
                [ind.s, np.ones(needed - cur, dtype=np.int8)])
        elif needed < cur:
            ind.s = ind.s[:needed]

        ind.N = new_N
        return ind

    # ── 个体初始化 ────────────────────────────────────────────────────────

    def _init_individual(self, rng: np.random.RandomState) -> Individual:
        """
        论文 §3 Population initialization：
          N ~ Uniform[N_min, N_max]
          nn[i,j] ~ Uniform([-k,k]\\{0})
          R[i][j,l] ~ Uniform[0,1]
          s = 全 1（初始保留全部多数类 + 所有合成样本）
        """
        n_min, m = self.X_min_.shape
        N       = int(rng.randint(self.N_min, self.N_max + 1))
        choices = self._nn_choices()
        nn      = np.array(rng.choice(choices, size=(n_min, N)), dtype=np.int32)
        R       = [rng.uniform(0, 1, (N, m)).astype(np.float32)
                   for _ in range(n_min)]
        s       = np.ones(self.n_maj_ + n_min * N, dtype=np.int8)
        return Individual(N, nn, R, s)

    # ── Algorithm 1：解码个体 → 增强训练集 ───────────────────────────────

    def _decode(self, ind: Individual):
        """
        Algorithm 1: BlindSMOTE procedure to obtain dataset T^A from individual i.

        [1]  T^A = T
        [2]  T^A -= { x_j : s_j=0 ∧ y_j=0 }          多数类下采样
        [3]  φ = { x_j ∈ T | y_j=1 }
        [4]  p = n+1
        foreach x_i ∈ φ:
          for j = 1..N:
        [5]    x_j = nn[i,j] 对应的邻居
        [6]    x_p = x_i + R[i][j] ⊙ (x_j - x_i)     论文公式 (2)
        [7]    p++
        [8]  T^A -= { x_j : s_j=0 ∧ j>n }             合成样本筛选
        [9]  return T^A

        Returns（六元组，三组各自独立）
        -------
        X_min_orig : ndarray (n_min, m)        原始少数类特征
        y_min_orig : ndarray (n_min,)          原始少数类标签
        X_maj_sel  : ndarray (n_maj_kept, m)   下采样后多数类特征
        y_maj_sel  : ndarray (n_maj_kept,)     下采样后多数类标签
        X_synth    : ndarray (n_synth, m)      合成少数类特征（无则 shape=(0,m)）
        y_synth    : ndarray (n_synth,)        合成少数类标签（无则 shape=(0,)）
        """
        X_min, X_maj = self.X_min_, self.X_maj_
        knn          = self.knn_idx_          # (n_min, k_)
        n_min        = X_min.shape[0]
        N_actual     = ind.nn.shape[1]        # 以实际列数为准（防御性）

        # [2] 多数类下采样
        s_maj     = ind.s[:self.n_maj_]
        X_maj_sel = X_maj[s_maj == 1]
        y_maj_sel = np.full(len(X_maj_sel), self.maj_class_, dtype=int)

        # [3-7] 生成合成样本，[8] 同步筛选
        s_synth    = ind.s[self.n_maj_:]
        synth_rows = []
        ptr        = 0

        for i in range(n_min):
            xi = X_min[i]
            for j in range(N_actual):
                nn_ij = int(ind.nn[i, j])
                keep  = (ptr < len(s_synth)) and (s_synth[ptr] == 1)
                ptr  += 1

                if nn_ij < 0 or not keep:
                    # 负值（不生成）或被 s 筛掉
                    continue

                nb_idx = min(abs(nn_ij) - 1, self.k_ - 1)   # 0-based
                xj     = X_min[knn[i, nb_idx]]
                xn     = xi + ind.R[i][j] * (xj - xi)       # 公式 (2)
                synth_rows.append(xn)

        # 合成样本（可能为空）
        if synth_rows:
            X_synth = np.array(synth_rows, dtype=np.float32)
        else:
            X_synth = np.empty((0, X_min.shape[1]), dtype=np.float32)
        y_synth = np.full(len(X_synth), self.min_class_, dtype=int)

        # 原始少数类（全部保留，论文不对少数类原始样本做筛选）
        X_min_orig = X_min.copy()
        y_min_orig = np.full(n_min, self.min_class_, dtype=int)

        return X_min_orig, y_min_orig, X_maj_sel, y_maj_sel, X_synth, y_synth

    # ── 适应度评估：分层 k 折交叉验证 ────────────────────────────────────

    def _evaluate(self, ind: Individual) -> float:
        """
        分层 k 折交叉验证适应度估计，无信息泄露。

        每折流程：
          1. 将原始训练集 T 分为 T_tr_fold（训练折）和 T_val_fold（验证折）
          2. 在 T_tr_fold 上重放 ind 的操作：
               - 按 ind.s[:n_maj] 的保留比例对该折多数类下采样
               - 用该折少数类重建临时 k 近邻，按 ind.nn/R 生成合成样本
               - 按 ind.s[n_maj:] 的保留比例筛选合成样本
             → 得到增强折训练集 T^A_tr_fold
          3. clf.fit(T^A_tr_fold) → 在 T_val_fold 上预测
          4. 计算 (G-mean + F1) / 2

        最终适应度 = 各折得分的均值。
        """
        from sklearn.model_selection import StratifiedKFold

        # 原始保留率（用于等比映射到各折子集）
        maj_keep_ratio   = float(ind.s[:self.n_maj_].mean())
        synth_keep_ratio = (float(ind.s[self.n_maj_:].mean())
                            if len(ind.s) > self.n_maj_ else 1.0)
        N_actual         = ind.nn.shape[1]
        n_ind_min        = ind.nn.shape[0]   # ind 中少数类样本数（来自完整训练集）

        skf    = StratifiedKFold(n_splits=self.cv_folds, shuffle=False)
        scores = []

        for tr_idx, val_idx in skf.split(self.X_, self.y_):
            X_tr_f, y_tr_f   = self.X_[tr_idx], self.y_[tr_idx]
            X_val_f, y_val_f = self.X_[val_idx], self.y_[val_idx]

            X_min_f = X_tr_f[y_tr_f == self.min_class_]
            X_maj_f = X_tr_f[y_tr_f == self.maj_class_]
            n_min_f = len(X_min_f)
            n_maj_f = len(X_maj_f)

            if n_min_f == 0 or n_maj_f == 0:
                continue

            # 1. 多数类下采样（等比映射）
            n_maj_keep = max(1, int(round(n_maj_f * maj_keep_ratio)))
            maj_idx    = self.rng_cv_.choice(n_maj_f, n_maj_keep, replace=False)
            X_maj_sel  = X_maj_f[maj_idx]
            y_maj_sel  = np.full(n_maj_keep, self.maj_class_, dtype=int)

            # 2. 为该折少数类重建临时 k 近邻
            k_f = min(self.k_, n_min_f - 1)
            if k_f < 1:
                X_aug_f = np.vstack([X_min_f, X_maj_sel])
                y_aug_f = np.concatenate([
                    np.full(n_min_f, self.min_class_, dtype=int), y_maj_sel])
            else:
                nbrs_f   = NearestNeighbors(n_neighbors=k_f + 1).fit(X_min_f)
                _, knn_f = nbrs_f.kneighbors(X_min_f)
                knn_f    = knn_f[:, 1:]      # 排除自身，shape (n_min_f, k_f)

                # 3. 重放合成样本生成（nn/R 映射到该折少数类）
                synth_f = []
                s_syn   = ind.s[self.n_maj_:]
                s_len   = max(len(s_syn), 1)

                ptr = 0
                for i in range(n_min_f):
                    i_src = i % n_ind_min      # 循环复用 ind 的参数行
                    xi    = X_min_f[i]
                    for j in range(N_actual):
                        # 等比映射：按 synth_keep_ratio 随机决定是否保留
                        keep  = self.rng_cv_.random() < synth_keep_ratio
                        nn_ij = int(ind.nn[i_src, j])
                        ptr  += 1
                        if nn_ij < 0 or not keep:
                            continue
                        nb  = min(abs(nn_ij) - 1, k_f - 1)
                        xj  = X_min_f[knn_f[i, nb]]
                        r   = ind.R[i_src][j]
                        synth_f.append(xi + r * (xj - xi))

                X_parts = [X_min_f, X_maj_sel]
                y_parts = [np.full(n_min_f, self.min_class_, dtype=int), y_maj_sel]
                if synth_f:
                    X_parts.append(np.array(synth_f, dtype=np.float32))
                    y_parts.append(np.full(len(synth_f), self.min_class_, dtype=int))
                X_aug_f = np.vstack(X_parts)
                y_aug_f = np.concatenate(y_parts)

            if len(np.unique(y_aug_f)) < 2:
                continue

            # 4. 训练并在验证折上评估
            clf = deepcopy(self.clf_)
            try:
                clf.fit(X_aug_f, y_aug_f)
                y_pred = clf.predict(X_val_f)
                scores.append(combined_fitness(y_val_f, y_pred))
            except Exception:
                pass

        return float(np.mean(scores)) if scores else 0.0

    # ── 遗传算子 ──────────────────────────────────────────────────────────

    def _crossover(self, p1: Individual, p2: Individual,
                   rng: np.random.RandomState
                   ) -> Tuple[Individual, Individual]:
        """
        论文 §3 交叉算子——等概率选择两种方案：

        N：
          50% 直接交换 | 50% BLX-α (α=0.5)

        nn + R：
          方案 A  对每列独立均匀交换 nn；R 做 BLX-α 交叉
          方案 B  交换 nn[i,j] 时同时携带对应 R[i][j]（整个合成样本一起换）

        s：
          HUX 风格——仅随机交换恰好一半的不同位
        """
        c1, c2   = p1.clone(), p2.clone()
        use_b    = rng.random() < 0.5
        n_min    = self.X_min_.shape[0]
        m        = self.X_min_.shape[1]

        # —— N ——
        if rng.random() < 0.5:
            c1.N, c2.N = c2.N, c1.N
        else:
            lo, hi = float(min(c1.N, c2.N)), float(max(c1.N, c2.N))
            span   = (hi - lo) * 0.5
            c1.N   = int(np.clip(rng.uniform(lo - span, hi + span),
                                 self.N_min, self.N_max))
            c2.N   = int(np.clip(rng.uniform(lo - span, hi + span),
                                 self.N_min, self.N_max))

        # —— nn + R ——
        N_cols = min(p1.nn.shape[1], p2.nn.shape[1])
        for i in range(n_min):
            for j in range(N_cols):
                if rng.random() < 0.5:
                    j1 = j % c1.nn.shape[1]
                    j2 = j % c2.nn.shape[1]
                    c1.nn[i, j1], c2.nn[i, j2] = (
                        int(c2.nn[i, j2]), int(c1.nn[i, j1]))
                    if use_b:                       # 方案 B：同步换 R 行
                        r1 = j1 % c1.R[i].shape[0]
                        r2 = j2 % c2.R[i].shape[0]
                        c1.R[i][r1], c2.R[i][r2] = (
                            c2.R[i][r2].copy(), c1.R[i][r1].copy())

            if not use_b:                           # 方案 A：BLX-α 交叉 R
                Ni    = min(c1.R[i].shape[0], c2.R[i].shape[0])
                lo_r  = np.minimum(c1.R[i][:Ni], c2.R[i][:Ni])
                hi_r  = np.maximum(c1.R[i][:Ni], c2.R[i][:Ni])
                span  = (hi_r - lo_r) * 0.5
                c1.R[i][:Ni] = np.clip(
                    lo_r - span + rng.uniform(0,1,(Ni,m)) * (hi_r-lo_r+2*span),
                    0, 1).astype(np.float32)
                c2.R[i][:Ni] = np.clip(
                    lo_r - span + rng.uniform(0,1,(Ni,m)) * (hi_r-lo_r+2*span),
                    0, 1).astype(np.float32)

        # —— 对齐 nn/R/s 尺寸 ——
        c1 = self._resize_individual(c1, c1.N, rng)
        c2 = self._resize_individual(c2, c2.N, rng)

        # —— s：HUX ——
        min_len = min(len(c1.s), len(c2.s))
        diff    = np.where(c1.s[:min_len] != c2.s[:min_len])[0]
        if len(diff) > 1:
            swap = rng.choice(diff, size=len(diff) // 2, replace=False)
            for idx in swap:
                c1.s[idx], c2.s[idx] = c2.s[idx], c1.s[idx]

        return c1, c2

    def _mutate(self, ind: Individual,
                rng: np.random.RandomState) -> Individual:
        """
        论文 §3.1 变异算子：随机选择四部分之一进行变异。

        N   非均匀变异（±1 或高斯扰动），变化后同步 nn/R/s 尺寸
        nn  50% 非均匀（值域内 ±1 调整）| 50% 随机（替换值或翻转符号）
        R   非均匀变异（小幅高斯噪声，clamp 至 [0,1]）
        s   随机位翻转（概率 mut_bit_rate）
        """
        part    = rng.randint(4)
        choices = self._nn_choices()

        if part == 0:                               # —— N ——
            old_N = ind.N
            delta = (rng.choice([-1, 1]) if rng.random() < 0.5
                     else int(np.round(rng.normal(0, 0.8))))
            new_N = int(np.clip(old_N + delta, self.N_min, self.N_max))
            if new_N != old_N:
                ind = self._resize_individual(ind, new_N, rng)

        elif part == 1:                             # —— nn ——
            mask = rng.random(ind.nn.shape) < self.mut_bit_rate
            if mask.any():
                if rng.random() < 0.5:             # 非均匀
                    delta  = rng.choice([-1, 0, 1], size=ind.nn.shape)
                    new_nn = (ind.nn + mask * delta).astype(np.int32)
                    new_nn = np.where(new_nn == 0, 1, new_nn)
                    ind.nn = np.clip(new_nn, -self.k_, self.k_)
                else:                               # 随机
                    for i in range(ind.nn.shape[0]):
                        for j in range(ind.nn.shape[1]):
                            if mask[i, j]:
                                if rng.random() < 0.5:
                                    ind.nn[i, j] = rng.choice(choices)
                                else:
                                    ind.nn[i, j] = -ind.nn[i, j]

        elif part == 2:                             # —— R ——
            for i in range(len(ind.R)):
                m_mask = rng.random(ind.R[i].shape) < self.mut_bit_rate
                if m_mask.any():
                    noise    = rng.normal(0, 0.05, ind.R[i].shape).astype(np.float32)
                    ind.R[i] = np.clip(ind.R[i] + m_mask * noise, 0, 1)

        else:                                       # —— s ——
            flip   = rng.random(ind.s.shape) < self.mut_bit_rate
            ind.s  = np.where(flip, (1 - ind.s).astype(np.int8), ind.s)

        return ind

    # ── Algorithm 2：主进化循环 ───────────────────────────────────────────

    def _run_ga(self, rng: np.random.RandomState) -> Individual:
        """
        Algorithm 2: BlindSMOTE genetic algorithm.

        [1]  随机初始化种群
        [2-5] 评估初始种群适应度
        while #gen < G and runtime < tl:
          [6]  交叉，种群扩至 2S
          [7-10] 评估后代适应度
          [11] μ+λ 精英选择，保留最优 S 个
          [12-13] 变异（概率 p_mutation）
          [14] 停滞检测：无改善 100 代 → 重置最差 10%
        [15] 返回最优个体对应的 T^A
        """
        t_start = time.time()
        hof     = HallOfFame(maxsize=1)
        stats   = Statistics()

        # [1] 初始化种群
        pop = [self._init_individual(rng) for _ in range(self.pop_size)]

        # [2-5] 评估初始种群
        fit_list = [self._evaluate(ind) for ind in pop]
        for ind, f in zip(pop, fit_list):
            ind.fitness = FitnessMax(f)
        hof.update(pop, fit_list)
        stats.record(0, fit_list)

        log_every = max(1, self.n_gen // 20)
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
            idx      = rng.permutation(self.pop_size)
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

            # [11] μ+λ 精英选择：合并后保留最优 pop_size 个
            combined = sorted(
                zip(pop + offspring, fit_list + off_fit),
                key=lambda x: x[1], reverse=True)
            pop      = [c[0] for c in combined[:self.pop_size]]
            fit_list = [c[1] for c in combined[:self.pop_size]]

            prev_best  = hof.best_fitness
            hof.update(pop, fit_list)
            stagnation = 0 if hof.best_fitness > prev_best else stagnation + 1

            # [14] 停滞：重置最差 10%
            if stagnation >= self.stagnation_gens:
                n_reinit = max(1, self.pop_size // 10)
                for i in range(self.pop_size - n_reinit, self.pop_size):
                    pop[i]      = self._init_individual(rng)
                    fit_list[i] = self._evaluate(pop[i])
                    pop[i].fitness = FitnessMax(fit_list[i])
                stagnation = 0
                if self.verbose:
                    print(f"  gen={gen:5d}  [停滞] 重置最差 {n_reinit} 个体")

            stats.record(gen, fit_list)
            if self.verbose and gen % log_every == 0:
                print(f"  gen={gen:5d}  best={hof.best_fitness:.4f}"
                      f"  mean={np.mean(fit_list):.4f}"
                      f"  std={np.std(fit_list):.4f}")

        self.hof_   = hof
        self.stats_ = stats
        return hof.best

    # ── 公共接口 ──────────────────────────────────────────────────────────

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray,
               np.ndarray, np.ndarray,
               np.ndarray, np.ndarray]:
        """
        对 (X, y) 执行 BlindSMOTE 进化重采样（二分类）。

        Returns（六元组）
        -------
        X_min_orig : ndarray (n_min, m)       原始少数类特征
        y_min_orig : ndarray (n_min,)         原始少数类标签
        X_maj_sel  : ndarray (n_maj_kept, m)  下采样后的多数类特征
        y_maj_sel  : ndarray (n_maj_kept,)    下采样后的多数类标签
        X_synth    : ndarray (n_synth, m)     合成的新少数类特征
        y_synth    : ndarray (n_synth,)       合成的新少数类标签

        使用示例
        --------
        X_min, y_min, X_maj, y_maj, X_syn, y_syn = bs.fit_resample(X_tr, y_tr)

        # 需要完整增强集时自行拼接：
        X_aug = np.vstack([X_min, X_maj, X_syn])
        y_aug = np.concatenate([y_min, y_maj, y_syn])
        """
        X   = np.asarray(X, dtype=np.float32)
        y   = np.asarray(y, dtype=int)
        base_seed = check_random_state(self.random_state).randint(0, 2**31)
        rng          = np.random.RandomState(base_seed)
        self.rng_cv_ = np.random.RandomState(base_seed ^ 0xDEAD)  # CV 专用，独立流

        # Wrapper 分类器
        self.clf_ = (deepcopy(self.classifier)
                     if self.classifier is not None
                     else DecisionTreeClassifier(random_state=0))

        # 识别少数类 / 多数类
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("BlindSMOTE 仅支持二分类。")
        self.min_class_ = int(classes[np.argmin(counts)])
        self.maj_class_ = int(classes[np.argmax(counts)])

        self.X_    = X
        self.y_    = y
        self.X_min_ = X[y == self.min_class_]
        self.X_maj_ = X[y == self.maj_class_]
        self.n_maj_ = int(self.X_maj_.shape[0])

        n_min = self.X_min_.shape[0]
        if n_min <= 1:
            warnings.warn("少数类样本数 ≤ 1，返回原始数据。")
            return X, y

        # 构建 k 近邻（固定超参数，论文 k=5）
        self.k_ = min(self.k, n_min - 1)
        nbrs    = NearestNeighbors(n_neighbors=self.k_ + 1).fit(self.X_min_)
        _, idx  = nbrs.kneighbors(self.X_min_)
        self.knn_idx_ = idx[:, 1:]          # 排除自身，shape (n_min, k_)

        if self.verbose:
            ir = self.n_maj_ / n_min
            print(f"[BlindSMOTE] 少数类={n_min}, 多数类={self.n_maj_},"
                  f" IR={ir:.2f}, k={self.k_}")
            print(f"             种群={self.pop_size}, 最大代数={self.n_gen},"
                  f" N∈[{self.N_min},{self.N_max}]")
            print()

        # 运行进化算法（Algorithm 2）
        best_ind = self._run_ga(rng)
        self.best_individual_ = best_ind

        # [15] 解码最优个体 → 三组独立数据
        X_min_orig, y_min_orig, X_maj_sel, y_maj_sel, X_synth, y_synth = \
            self._decode(best_ind)

        if self.verbose:
            print(f"\n[BlindSMOTE] 进化完成。")
            print(f"  原始少数类:  {len(X_min_orig)} 条")
            print(f"  下采样多数类: {len(X_maj_sel)} 条"
                  f"  (原始 {self.n_maj_} 条，"
                  f"保留 {len(X_maj_sel)/self.n_maj_*100:.1f}%)")
            print(f"  合成少数类:  {len(X_synth)} 条")
            print(f"  最优适应度:  {self.hof_.best_fitness:.4f}")

        return X_min_orig, y_min_orig, X_maj_sel, y_maj_sel, X_synth, y_synth

    def print_logbook(self, last_n: int = 20):
        """打印进化历史统计（最后 last_n 代）"""
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
    print("  BlindSMOTE 演示（论文原文适应度评估方式）")
    print("=" * 65)

    # 生成不均衡二分类数据集
    X, y = make_classification(
        n_samples=600, n_features=12, n_informative=6, n_redundant=2,
        weights=[0.85, 0.15], flip_y=0.01, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    print(f"训练集: 多数类={np.sum(y_tr==0)}, 少数类={np.sum(y_tr==1)},"
          f" IR={np.sum(y_tr==0)/np.sum(y_tr==1):.1f}")

    # 基线
    clf_base = DecisionTreeClassifier(random_state=0)
    clf_base.fit(X_tr, y_tr)
    y_base = clf_base.predict(X_te)
    print("\n── 基线（无重采样）──")
    print(classification_report(y_te, y_base, target_names=["Majority", "Minority"]))

    # BlindSMOTE（演示用小规模参数，论文建议 pop=100, gen=10000）
    bs = BlindSMOTE(
        k=5, N_min=1, N_max=10,
        pop_size=30, n_gen=100,
        cx_prob=0.8, mut_prob=0.05, mut_bit_rate=0.01,
        stagnation_gens=30,
        cv_folds=5,
        classifier=DecisionTreeClassifier(random_state=0),
        time_limit=300, random_state=42, verbose=True)

    t0 = time.time()
    X_min_orig, y_min_orig, X_maj_sel, y_maj_sel, X_synth, y_synth = \
        bs.fit_resample(X_tr, y_tr)
    elapsed = time.time() - t0

    # 展示三组各自的数量
    print(f"\n── 重采样结果明细（耗时 {elapsed:.1f}s）──")
    print(f"  原始少数类:   X_min_orig {X_min_orig.shape}  "
          f"标签唯一值={np.unique(y_min_orig)}")
    print(f"  下采样多数类: X_maj_sel  {X_maj_sel.shape}  "
          f"标签唯一值={np.unique(y_maj_sel)}")
    print(f"  合成少数类:   X_synth    {X_synth.shape}  "
          f"标签唯一值={np.unique(y_synth) if len(y_synth) else '(空)'}")

    # 拼接成完整增强集用于训练
    X_aug = np.vstack([X_min_orig, X_maj_sel] +
                      ([X_synth] if len(X_synth) > 0 else []))
    y_aug = np.concatenate([y_min_orig, y_maj_sel] +
                           ([y_synth] if len(y_synth) > 0 else []))
    print(f"  拼接后总计:   X_aug      {X_aug.shape}")

    clf_bs = DecisionTreeClassifier(random_state=0)
    clf_bs.fit(X_aug, y_aug)
    y_bs = clf_bs.predict(X_te)
    print("\n── BlindSMOTE 结果 ──")
    print(classification_report(y_te, y_bs, target_names=["Majority", "Minority"]))

    gm_b  = gmean_score(y_te, y_base)
    f1_b  = f1_score(y_te, y_base, zero_division=0)
    gm_bs = gmean_score(y_te, y_bs)
    f1_bs = f1_score(y_te, y_bs,   zero_division=0)

    print(f"{'指标':<12} {'基线':>8} {'BlindSMOTE':>12}")
    print("-" * 34)
    print(f"{'G-mean':<12} {gm_b:>8.4f} {gm_bs:>12.4f}")
    print(f"{'F1(少数类)':<12} {f1_b:>8.4f} {f1_bs:>12.4f}")
    print(f"{'进化适应度':<12} {'—':>8} {bs.hof_.best_fitness:>12.4f}")

    print("\n── 进化历史（最后 10 代）──")
    bs.print_logbook(last_n=10)
