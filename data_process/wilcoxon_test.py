import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# ======================
# 配置
# ======================
methods = [
    "RAW",
    "ROS",
    "SMOTE",
    "SMOTEN",
    "Borderline_1",
    "Borderline_2"
]

method_display = {
    "RAW": "RAW",
    "ROS": "ROS",
    "SMOTE": "SMOTE",
    "SMOTEN": "SMOTEN",
    "Borderline_1": "Borderline-SMOTE-1",
    "Borderline_2": "Borderline-SMOTE-2",
    "GP-SMOTE": "GP-SMOTE"
}

metrics = ["F-measure", "AUC"]

datasetnames = [  # 你的27个数据集
    "sonar", "banana", "australian", "heart", "spambase", "wdbc", "wisconsin", "pima",
    "iris0", "glass0", "german", "phoneme", "yeast1", "vehicle2", "ecoli1", "appendicitis",
    "new-thyroid1", "ecoli2", "segment0", "yeast-0-2-5-6-vs-3-7-8-9",
    "led7digit-0-2-4-5-6-7-8-9-vs-1", "cleveland-0-vs-4", "yeast-1-vs-7",
    "shuttle-6-vs-2-3", "yeast-2-vs-8", "winequality-red-8-vs-6-7", "shuttle-2-vs-5"]

# 路径
mean_root = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\F1和AUC\\knn"  # 平均结果（换成dt/knn也行）
raw_root = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\原始数据\\knn"  # 原始30次结果


# ======================
# Wilcoxon符号判断
# ======================
def get_symbol(gp_vals, other_vals):
    if np.array_equal(gp_vals, other_vals):
        return "$\\approx$"
    try:
        stat, p = wilcoxon(gp_vals, other_vals)
    except ValueError:
        # 兜底（极端情况下）
        return "$\\approx$"
    if p < 0.05:
        if np.mean(gp_vals) > np.mean(other_vals):
            return "$+$"
        else:
            return "$-$"
    else:
        return "$\\approx$"


# ======================
# 读取平均值
# ======================
def load_mean(method, metric):
    df = pd.read_csv(os.path.join(mean_root, f"{method}.csv"))
    return dict(zip(df["数据集"], df[metric]))


# ======================
# 读取30次原始数据
# ======================
def load_raw(method, dataset, metric):
    file = os.path.join(raw_root, method, f"{dataset}.csv")
    df = pd.read_csv(file)
    return df[metric].values


# ======================
# 计算标准差
# ======================
def format_std(std):
    if std == 0:
        return "0.00"

    exp = int(np.floor(np.log10(std)))
    base = std / (10 ** exp)

    return f"{base:.2f}e{exp}"


# ======================
# 生成LaTeX表
# ======================
def generate_table(metric):
    # 加载平均值
    mean_data = {m: load_mean(m, metric) for m in methods}
    mean_data["GP-SMOTE"] = load_mean("GP-SMOTE", metric)

    latex = []

    # 表头
    header = "Dataset & RAW & ROS & SMOTE & SMOTEN & Borderline-SMOTE-1 & Borderline-SMOTE-2 & GP-SMOTE \\\\"
    latex.append(header)
    latex.append("\\hline")

    for idx, ds in enumerate(datasetnames):

        # 收集所有值
        values = []
        for m in methods + ["GP-SMOTE"]:
            values.append(mean_data[m][ds])

        max_val = max(values)

        row = [f"D{idx + 1}"]

        # 逐方法处理
        for m in methods:
            val = mean_data[m][ds]

            # Wilcoxon
            gp_raw = load_raw("GP-SMOTE", ds, metric)
            other_raw = load_raw(m, ds, metric)
            symbol = get_symbol(gp_raw, other_raw)

            # ===== 标准差 =====
            std = np.std(other_raw, ddof=1)
            std_str = format_std(std)

            text = f"{100 * val:.2f}({std_str}){symbol}"

            if val == max_val:
                text = f"\\hl{{{100 * val:.2f}({std_str}){symbol}}}"

            row.append(text)

        # GP-SMOTE（无符号）
        gp_val = mean_data["GP-SMOTE"][ds]

        gp_raw = np.array(load_raw("GP-SMOTE", ds, metric))
        gp_std = np.std(gp_raw, ddof=1)
        gp_std_str = format_std(gp_std)

        if gp_val == max_val:
            row.append(f"\\hl{{{100 * gp_val:.2f}({gp_std_str})}}")
        else:
            row.append(f"{100 * gp_val:.2f}({gp_std_str})")

        latex.append(" & ".join(row) + " \\\\")

    return "\n".join(latex)


# ======================
# 输出两个表
# ======================
for metric in metrics:
    print(f"\n===== {metric} =====\n")
    table = generate_table(metric)
    print(table)
