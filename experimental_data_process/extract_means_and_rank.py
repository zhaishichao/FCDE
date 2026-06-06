import re
import os
import pandas as pd
import numpy as np


# ======================
# 解析函数
# ======================
def parse_cell(cell):
    """返回 (mean, symbol)"""
    cell = cell.strip()
    has_hl = cell.startswith("\\hl{")
    if has_hl:
        inner = cell[4:-1]
    else:
        inner = cell

    symbol = None
    if inner.endswith("$+$"):
        symbol = "+"
        inner = inner[:-3]
    elif inner.endswith("$-$"):
        symbol = "-"
        inner = inner[:-3]
    elif inner.endswith("$\\approx$"):
        symbol = "≈"
        inner = inner[:-9]

    m = re.match(r"([\d.]+)\(.*\)", inner)
    mean = float(m.group(1)) if m else None
    return mean, symbol


def parse_line(line):
    """返回 (d_label, fmeasure_data, auc_data)
    每组 data 为 [(mean, symbol), ...] 4个元素"""
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split(" & ")
    if parts[-1].endswith(" \\\\"):
        parts[-1] = parts[-1][:-3]
    if len(parts) != 9:
        return None
    d_label = parts[0].strip()
    fmeasure = [parse_cell(parts[i]) for i in range(1, 5)]
    auc = [parse_cell(parts[i]) for i in range(5, 9)]
    return d_label, fmeasure, auc


def rank_values(values, method_names):
    """对 values 从高到低排名（最高=1），平局取平均"""
    s = pd.Series(values, index=method_names)
    return s.rank(ascending=False, method="average")


# ======================
# 主流程
# ======================
def main():
    txt_path = os.path.join(os.path.dirname(__file__), "table.txt")
    out_dir = os.path.dirname(__file__)

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    method_names = ["DG-SMOTE", "MTGP-SMOTE", "Blind-SMOTE", "GP-SMOTE"]

    # 存储
    fmeasure_means = []  # 每行: {Dataset: xx, DG-SMOTE: xx, ...}
    auc_means = []
    fmeasure_ranks = []
    auc_ranks = []
    symbol_counts_fm = {"DG-SMOTE": {"+": 0, "≈": 0, "-": 0},
                        "MTGP-SMOTE": {"+": 0, "≈": 0, "-": 0},
                        "Blind-SMOTE": {"+": 0, "≈": 0, "-": 0}}
    symbol_counts_auc = {"DG-SMOTE": {"+": 0, "≈": 0, "-": 0},
                         "MTGP-SMOTE": {"+": 0, "≈": 0, "-": 0},
                         "Blind-SMOTE": {"+": 0, "≈": 0, "-": 0}}

    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            continue
        d_label, fmeasure, auc = parsed

        # F-measure 均值
        fm_means = {m: fmeasure[i][0] for i, m in enumerate(method_names)}
        fm_means["Dataset"] = d_label
        fmeasure_means.append(fm_means)

        # AUC 均值
        auc_m = {m: auc[i][0] for i, m in enumerate(method_names)}
        auc_m["Dataset"] = d_label
        auc_means.append(auc_m)

        # F-measure 排名
        fm_vals = [fmeasure[i][0] for i in range(4)]
        fm_rank = rank_values(fm_vals, method_names)
        fm_rank["Dataset"] = d_label
        fmeasure_ranks.append(fm_rank)

        # AUC 排名
        auc_vals = [auc[i][0] for i in range(4)]
        auc_r = rank_values(auc_vals, method_names)
        auc_r["Dataset"] = d_label
        auc_ranks.append(auc_r)

        # 符号统计（排除 GP-SMOTE），F-measure 和 AUC 分开
        for i, m in enumerate(method_names[:3]):
            symbol = fmeasure[i][1]
            if symbol:
                symbol_counts_fm[m][symbol] += 1
            symbol = auc[i][1]
            if symbol:
                symbol_counts_auc[m][symbol] += 1

    # ======================
    # 保存均值 CSV
    # ======================
    df_fm = pd.DataFrame(fmeasure_means, columns=["Dataset"] + method_names)
    df_auc = pd.DataFrame(auc_means, columns=["Dataset"] + method_names)

    # 合并 F-measure 和 AUC 到一个文件
    df_fm.insert(0, "Metric", "F-measure")
    df_auc.insert(0, "Metric", "AUC")
    df_means = pd.concat([df_fm, df_auc], ignore_index=True)
    means_path = os.path.join(out_dir, "table_mean_values.csv")
    df_means.to_csv(means_path, index=False, encoding="utf-8-sig")
    print(f"均值已保存至: {means_path}")

    # ======================
    # 保存排名 CSV
    # ======================
    df_fm_rank = pd.DataFrame(fmeasure_ranks, columns=["Dataset"] + method_names)
    df_auc_rank = pd.DataFrame(auc_ranks, columns=["Dataset"] + method_names)

    # 计算平均排名
    fm_avg = {m: df_fm_rank[m].mean() for m in method_names}
    auc_avg = {m: df_auc_rank[m].mean() for m in method_names}

    # 添加平均排名行
    fm_avg_row = {"Dataset": "Average Rank"}
    fm_avg_row.update({m: round(v, 2) for m, v in fm_avg.items()})
    df_fm_rank = pd.concat([df_fm_rank, pd.DataFrame([fm_avg_row])], ignore_index=True)

    auc_avg_row = {"Dataset": "Average Rank"}
    auc_avg_row.update({m: round(v, 2) for m, v in auc_avg.items()})
    df_auc_rank = pd.concat([df_auc_rank, pd.DataFrame([auc_avg_row])], ignore_index=True)

    # 合并到一个文件
    df_fm_rank.insert(0, "Metric", "F-measure")
    df_auc_rank.insert(0, "Metric", "AUC")
    df_ranks = pd.concat([df_fm_rank, df_auc_rank], ignore_index=True)
    ranks_path = os.path.join(out_dir, "table_rankings.csv")
    df_ranks.to_csv(ranks_path, index=False, encoding="utf-8-sig")
    print(f"排名已保存至: {ranks_path}")

    # ======================
    # 符号统计输出（F-measure 和 AUC 分开）
    # ======================
    print("\n===== Wilcoxon符号统计 ($+$/$\approx$/$-$) =====")
    for metric_name, counts in [("F-measure", symbol_counts_fm), ("AUC", symbol_counts_auc)]:
        parts = []
        for m in method_names[:3]:
            c = counts[m]
            parts.append(f"&\\textbf{{{c['+']}/{c['≈']}/{c['-']}}} ")
        parts.append("&\\textbf{--}")
        print(f"{metric_name}: " + " ".join(parts))

    # 每个方法的详细统计
    for metric_name, counts in [("F-measure", symbol_counts_fm), ("AUC", symbol_counts_auc)]:
        print(f"\n{metric_name}:")
        for m in method_names[:3]:
            c = counts[m]
            total = c['+'] + c['≈'] + c['-']
            print(f"  {m}: +={c['+']}, ≈={c['≈']}, -={c['-']} (共{total})")


if __name__ == "__main__":
    main()
