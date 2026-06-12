import re
import os
import pandas as pd


# ============================================================
# 表格结构配置（修改此处切换格式）
# ============================================================
# 格式A：双指标（F-measure + AUC），每组4个方法，GP-SMOTE 在每组最后一列
# CONFIG = {
#     "table_file": "table.txt",
#     "mean_output": "table_mean_values_table2.csv",
#     "rank_output": "table_rankings_table2.csv",
#     "groups": [
#         {
#             "name": "F-measure",
#             "methods": ["DG-SMOTE", "MTGP-SMOTE", "Blind-SMOTE", "GP-SMOTE"],
#             "col_range": (1, 5),
#         },
#         {
#             "name": "AUC",
#             "methods": ["DG-SMOTE", "MTGP-SMOTE", "Blind-SMOTE", "GP-SMOTE"],
#             "col_range": (5, 9),
#         },
#     ],
# }

# 格式B：单指标，7个方法，GP-SMOTE 在最后一列
CONFIG = {
    "table_file": "table2.txt",
    "mean_output": "table_mean_values_table2.csv",
    "rank_output": "table_rankings_table2.csv",
    "groups": [
        {
            "name": "AUC",
            "methods": ["Original", "ROS", "SMOTE", "KMeans-SMOTE",
                        "Borderline-1", "Borderline-2", "GP-SMOTE"],
            "col_range": (1, 8),
        },
    ],
}
# ============================================================


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


def parse_line(line, expected_cols):
    """返回 (d_label, all_cells)"""
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split(" & ")
    if parts[-1].endswith(" \\\\"):
        parts[-1] = parts[-1][:-3]
    if len(parts) != expected_cols:
        return None
    d_label = parts[0].strip()
    cells = [parse_cell(parts[i]) for i in range(1, expected_cols)]
    return d_label, cells


def rank_values(values, method_names):
    """对 values 从高到低排名（最高=1），平局取平均"""
    s = pd.Series(values, index=method_names)
    return s.rank(ascending=False, method="average")


def main():
    out_dir = os.path.dirname(__file__)
    txt_path = os.path.join(out_dir, CONFIG["table_file"])

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 计算列数
    max_col = max(g["col_range"][1] for g in CONFIG["groups"])

    # 准备存储（每个 group 独立存储）
    group_means = {}   # group_name -> [{Dataset: xx, method1: xx, ...}, ...]
    group_ranks = {}   # group_name -> [{Dataset: xx, method1: xx, ...}, ...]
    group_symbols = {} # group_name -> {method: {"+": 0, "≈": 0, "-": 0}}

    for g_cfg in CONFIG["groups"]:
        name = g_cfg["name"]
        group_means[name] = []
        group_ranks[name] = []
        group_symbols[name] = {m: {"+": 0, "≈": 0, "-": 0}
                               for m in g_cfg["methods"][:-1]}  # 排除 GP-SMOTE

    for line in lines:
        parsed = parse_line(line, max_col)
        if parsed is None:
            continue
        d_label, all_cells = parsed

        for g_cfg in CONFIG["groups"]:
            g_name = g_cfg["name"]
            methods = g_cfg["methods"]
            start, end = g_cfg["col_range"]
            cells = all_cells[start - 1:end - 1]
            n = len(cells)

            # 均值
            means = {methods[i]: cells[i][0] for i in range(n)}
            means["Dataset"] = d_label
            group_means[g_name].append(means)

            # 排名
            vals = [cells[i][0] for i in range(n)]
            rank = rank_values(vals, methods)
            rank["Dataset"] = d_label
            group_ranks[g_name].append(rank)

            # 符号统计（排除 GP-SMOTE，即最后一列）
            for i in range(n - 1):
                symbol = cells[i][1]
                if symbol:
                    group_symbols[g_name][methods[i]][symbol] += 1

    # ======================
    # 保存均值 CSV
    # ======================
    df_means_list = []
    for g_cfg in CONFIG["groups"]:
        g_name = g_cfg["name"]
        df = pd.DataFrame(group_means[g_name], columns=["Dataset"] + g_cfg["methods"])
        df.insert(0, "Metric", g_name)
        df_means_list.append(df)
    df_means = pd.concat(df_means_list, ignore_index=True)
    means_path = os.path.join(out_dir, CONFIG["mean_output"])
    df_means.to_csv(means_path, index=False, encoding="utf-8-sig")
    print(f"均值已保存至: {means_path}")

    # ======================
    # 保存排名 CSV
    # ======================
    df_ranks_list = []
    for g_cfg in CONFIG["groups"]:
        g_name = g_cfg["name"]
        methods = g_cfg["methods"]
        df = pd.DataFrame(group_ranks[g_name], columns=["Dataset"] + methods)

        # 计算平均排名
        avg = {m: df[m].mean() for m in methods}
        avg_row = {"Dataset": "Average Rank"}
        avg_row.update({m: round(v, 2) for m, v in avg.items()})
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        df.insert(0, "Metric", g_name)
        df_ranks_list.append(df)
    df_ranks = pd.concat(df_ranks_list, ignore_index=True)
    ranks_path = os.path.join(out_dir, CONFIG["rank_output"])
    df_ranks.to_csv(ranks_path, index=False, encoding="utf-8-sig")
    print(f"排名已保存至: {ranks_path}")

    # ======================
    # 符号统计输出（按 metric 分开）
    # ======================
    print("\n===== Wilcoxon符号统计 ($+$/$\approx$/$-$) =====")
    for g_cfg in CONFIG["groups"]:
        g_name = g_cfg["name"]
        counts = group_symbols[g_name]
        parts = []
        for m in g_cfg["methods"][:-1]:  # 排除 GP-SMOTE
            c = counts[m]
            parts.append(f"&\\textbf{{{c['+']}/{c['≈']}/{c['-']}}} ")
        parts.append("&\\textbf{--}")
        print(f"{g_name}: " + " ".join(parts))

    # 详细统计
    for g_cfg in CONFIG["groups"]:
        g_name = g_cfg["name"]
        counts = group_symbols[g_name]
        total_per = len(group_means[g_name])
        print(f"\n{g_name} (共{total_per}数据集):")
        for m in g_cfg["methods"][:-1]:
            c = counts[m]
            total = c['+'] + c['≈'] + c['-']
            print(f"  {m}: +={c['+']}, ≈={c['≈']}, -={c['-']} (共{total})")


if __name__ == "__main__":
    main()
