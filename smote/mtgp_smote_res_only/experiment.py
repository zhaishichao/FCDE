"""
MTGP-SMOTE 实验脚本
复现论文第 4、5 节的实验流程（精简版，使用内置数据集演示）

运行: python experiment.py
"""

import warnings
import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from mtgp_smote import MTGPSMOTESampler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 评估函数
# ─────────────────────────────────────────────

def geometric_mean_score(y_true, y_pred_proba, pos_label=1):
    """GMean = sqrt(sensitivity * specificity)"""
    classes = np.unique(y_true)
    if len(classes) != 2:
        return 0.0
    neg_label = [c for c in classes if c != pos_label][0]

    tp = np.sum((y_true == pos_label) & (y_pred_proba >= 0.5))
    fn = np.sum((y_true == pos_label) & (y_pred_proba < 0.5))
    tn = np.sum((y_true == neg_label) & (y_pred_proba < 0.5))
    fp = np.sum((y_true == neg_label) & (y_pred_proba >= 0.5))

    sens = tp / (tp + fn + 1e-10)
    spec = tn / (tn + fp + 1e-10)
    return float(np.sqrt(sens * spec))


def evaluate_sampler(sampler, X_train, y_train, X_test, y_test, clf):
    """用采样器对训练集重采样，训练分类器，在测试集上评估 AUC 和 GMean。"""
    try:
        if sampler is None:
            X_res, y_res = X_train, y_train
        else:
            X_res, y_res = sampler.fit_resample(X_train, y_train)

        clf.fit(X_res, y_res)

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test)[:, 1]
        else:
            proba = clf.decision_function(X_test)

        auc = roc_auc_score(y_test, proba)
        gmean = geometric_mean_score(y_test, proba)
        return auc, gmean
    except Exception as e:
        print(f"    [警告] {type(sampler).__name__} 出错: {e}")
        return 0.0, 0.0


# ─────────────────────────────────────────────
# 数据集生成（模拟论文中的不平衡场景）
# ─────────────────────────────────────────────

def make_imbalanced_dataset(n_samples=300, n_features=6, ir=10, random_state=42):
    """生成不平衡二分类数据集，IR = ir。"""
    n_minority = n_samples // (ir + 1)
    n_majority = n_samples - n_minority
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features // 2),
        n_redundant=min(2, n_features // 4),
        weights=[n_majority / n_samples, n_minority / n_samples],
        flip_y=0.01,
        random_state=random_state,
    )
    return X, y


# ─────────────────────────────────────────────
# 主实验
# ─────────────────────────────────────────────

def run_experiment():
    print("=" * 65)
    print("  MTGP-SMOTE 实验（基于论文算法的复现）")
    print("=" * 65)

    # 实验数据集配置（模拟论文 Table 1 的部分场景）
    datasets = [
        {"name": "IR=5  (n=300, f=6)",  "n": 300,  "f": 6,  "ir": 5,  "seed": 0},
        {"name": "IR=10 (n=300, f=7)",  "n": 300,  "f": 7,  "ir": 10, "seed": 1},
        {"name": "IR=15 (n=400, f=7)",  "n": 400,  "f": 7,  "ir": 15, "seed": 2},
        {"name": "IR=20 (n=500, f=8)",  "n": 500,  "f": 8,  "ir": 20, "seed": 3},
    ]

    # 分类器（对应论文 LR / RF / GBDT / AdaBoost）
    classifiers = {
        "LR":       LogisticRegression(max_iter=1000, random_state=42),
        "RF":       RandomForestClassifier(n_estimators=100, random_state=42),
        "GBDT":     GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    }

    # 采样方法（对应论文 baseline + MTGP-SMOTE）
    samplers = {
        "Original":    None,
        "SMOTE":       SMOTE(random_state=42),
        "ADASYN":      ADASYN(random_state=42),
        "BSMOTE1":     BorderlineSMOTE(kind="borderline-1", random_state=42),
        "BSMOTE2":     BorderlineSMOTE(kind="borderline-2", random_state=42),
        "SMOTE+ENN":   SMOTEENN(random_state=42),
        "SMOTE+Tomek": SMOTETomek(random_state=42),
        "MTGP-SMOTE":  MTGPSMOTESampler(
                           pop_size=50,        # 论文 512，此处缩小以加快演示
                           n_generations=30,   # 论文 100
                           cx_rate=0.7,
                           mut_rate=0.3,
                           tournament_k=7,
                           max_depth=4,
                           random_state=42,
                       ),
    }

    all_results = {}

    for ds in datasets:
        print(f"\n{'─'*65}")
        print(f"  数据集: {ds['name']}")
        print(f"{'─'*65}")

        X, y = make_imbalanced_dataset(
            n_samples=ds["n"], n_features=ds["f"],
            ir=ds["ir"], random_state=ds["seed"]
        )
        classes, counts = np.unique(y, return_counts=True)
        real_ir = max(counts) / min(counts)
        print(f"  样本数: {len(y)}, 特征数: {ds['f']}, "
              f"不平衡比: {real_ir:.1f}, 少数类: {min(counts)}")

        # 分层划分 70% / 30%
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=ds["seed"]
        )

        ds_results = {}

        for clf_name, clf in classifiers.items():
            print(f"\n  ── 分类器: {clf_name} ──")
            clf_results = {}
            for samp_name, sampler in samplers.items():
                t0 = time.time()
                auc, gmean = evaluate_sampler(
                    sampler, X_train.copy(), y_train.copy(),
                    X_test, y_test,
                    type(clf)(**clf.get_params())
                )
                elapsed = time.time() - t0
                clf_results[samp_name] = {"AUC": auc, "GMean": gmean,
                                          "time": elapsed}
                flag = " ◀ 最优" if samp_name == "MTGP-SMOTE" else ""
                print(f"    {samp_name:<15} AUC={auc:.4f}  GMean={gmean:.4f}"
                      f"  [{elapsed:.1f}s]{flag}")

            ds_results[clf_name] = clf_results

        all_results[ds["name"]] = ds_results

    # ── 汇总表格 ──
    print(f"\n\n{'='*65}")
    print("  汇总：各采样方法平均 AUC（所有数据集 × 所有分类器）")
    print(f"{'='*65}")

    samp_names = list(samplers.keys())
    auc_sums = {s: [] for s in samp_names}

    for ds_name, ds_res in all_results.items():
        for clf_name, clf_res in ds_res.items():
            for samp_name in samp_names:
                auc_sums[samp_name].append(clf_res[samp_name]["AUC"])

    print(f"  {'方法':<16} {'平均AUC':>10} {'排名':>6}")
    print(f"  {'─'*34}")

    sorted_methods = sorted(samp_names,
                            key=lambda s: np.mean(auc_sums[s]), reverse=True)
    for rank, samp_name in enumerate(sorted_methods, 1):
        avg = np.mean(auc_sums[samp_name])
        marker = " ★" if samp_name == "MTGP-SMOTE" else ""
        print(f"  {samp_name:<16} {avg:>10.4f} {rank:>6}{marker}")

    print("\n  实验完成！")
    return all_results


if __name__ == "__main__":
    run_experiment()
