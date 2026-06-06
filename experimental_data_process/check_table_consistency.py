import re
import csv
import os


def parse_cell(cell):
    """解析一个单元格，返回 (mean, has_hl, symbol)"""
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
    return mean, has_hl, symbol


def parse_line(line, expected_cols):
    """解析一行，返回 (d_label, cells)"""
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


def check_group(cells, method_names, group_name):
    """检查一组 cell（最后一个为 GP-SMOTE），返回异常列表"""
    results = []
    n = len(cells)
    gp_mean = cells[n - 1][0]

    means = [c[0] for c in cells if c[0] is not None]
    max_val = max(means) if means else None

    for i, (mean, has_hl, symbol) in enumerate(cells):
        if mean is None:
            continue

        col_name = method_names[i]
        anomalies = []

        # === Wilcoxon 符号检查（仅非 GP-SMOTE 列） ===
        if i < n - 1 and symbol is not None and gp_mean is not None:
            if symbol == "+" and mean >= gp_mean:
                anomalies.append(f"符号为+但{mean:.2f}>={gp_mean:.2f}")
            elif symbol == "-" and mean <= gp_mean:
                anomalies.append(f"符号为-但{mean:.2f}<={gp_mean:.2f}")

        # === \hl{} 高亮检查 ===
        if has_hl:
            if max_val is not None and mean < max_val:
                anomalies.append(f"非最大值却被\\hl高亮(max={max_val:.2f})")
        else:
            if max_val is not None and mean == max_val:
                anomalies.append(f"是最大值但未被\\hl高亮")

        anomaly_str = "; ".join(anomalies) if anomalies else "OK"
        results.append({
            "Column": col_name,
            "Mean": mean,
            "Symbol": symbol if symbol else "",
            "GP_Mean": gp_mean if i < n - 1 else "",
            "Has_HL": has_hl,
            "Is_Max": (mean == max_val) if max_val is not None else False,
            "Status": anomaly_str
        })
    return results


# ============================================================
# 表格结构配置（修改此处切换格式）
# ============================================================
# 格式A：双指标（F-measure + AUC），每组4个方法
# CONFIG = {
#     "table_file": "table.txt",
#     "output_file": "table_check_result.csv",
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

# 格式B：单指标，7个方法（GP-SMOTE 在最后一列）
CONFIG = {
    "table_file": "table.txt",
    "output_file": "table_check_result.csv",
    "groups": [
        {
            "name": "F-measure",
            "methods": ["Original", "ROS", "SMOTE", "KMeans-SMOTE",
                        "Borderline-1", "Borderline-2", "GP-SMOTE"],
            "col_range": (1, 8),
        },
    ],
}

# 格式A：双指标（F-measure + AUC），每组4个方法
# CONFIG = {
#     "table_file": "table.txt",
#     "output_file": "table_check_result.csv",
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
# ============================================================


def main():
    out_dir = os.path.dirname(__file__)
    txt_path = os.path.join(out_dir, CONFIG["table_file"])
    output_path = os.path.join(out_dir, CONFIG["output_file"])

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 计算总列数 = 1(数据集) + 各组列数之和
    total_cols = 1 + sum(g["col_range"][1] - g["col_range"][0] for g in CONFIG["groups"])
    # 取所有组中最大的结束列号作为预期列数
    max_col = max(g["col_range"][1] for g in CONFIG["groups"])
    expected_cols = 1 + max_col - min(g["col_range"][0] for g in CONFIG["groups"])
    expected_cols = max_col  # 总共有 max_col 列（数据集在 0）

    all_rows = []
    for line in lines:
        parsed = parse_line(line, max_col)
        if parsed is None:
            continue
        d_label, all_cells = parsed

        for group_cfg in CONFIG["groups"]:
            start, end = group_cfg["col_range"]
            group_cells = all_cells[start - 1:end - 1]
            group_results = check_group(group_cells, group_cfg["methods"], group_cfg["name"])
            for r in group_results:
                r["Dataset"] = d_label
                r["Group"] = group_cfg["name"]
            all_rows.extend(group_results)

    fieldnames = ["Dataset", "Group", "Column", "Mean", "GP_Mean", "Symbol", "Has_HL", "Is_Max", "Status"]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    anomalies = [r for r in all_rows if r["Status"] != "OK"]
    print(f"总检查项: {len(all_rows)}, 异常项: {len(anomalies)}")
    if anomalies:
        print("\n异常详情:")
        for r in anomalies:
            print(f"  {r['Dataset']} {r['Group']} {r['Column']}: {r['Status']}")
    else:
        print("全部通过检查。")

    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
