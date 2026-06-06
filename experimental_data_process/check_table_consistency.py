import re
import csv
import os


def parse_cell(cell):
    """解析一个单元格，返回 (mean, has_hl, symbol)"""
    cell = cell.strip()
    has_hl = cell.startswith("\\hl{")
    if has_hl:
        inner = cell[4:-1]  # 去掉 \hl{ 和 }
    else:
        inner = cell

    # 提取符号
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

    # 提取均值：括号前的数字
    m = re.match(r"([\d.]+)\(.*\)", inner)
    if m:
        mean = float(m.group(1))
    else:
        mean = None
    return mean, has_hl, symbol


def parse_line(line):
    """解析一行 LaTeX 表格数据，返回 (d_label, fmeasure_cells, auc_cells)
    每个 cell 为 (mean, has_hl, symbol)"""
    stripped = line.strip()
    if not stripped:
        return None

    parts = stripped.split(" & ")
    # 去掉末尾的 \\
    if parts[-1].endswith(" \\\\"):
        parts[-1] = parts[-1][:-3]

    if len(parts) != 9:
        return None

    d_label = parts[0].strip()
    fmeasure_cells = [parse_cell(parts[i]) for i in range(1, 5)]
    auc_cells = [parse_cell(parts[i]) for i in range(5, 9)]
    return d_label, fmeasure_cells, auc_cells


def check_group(cells, group_name):
    """检查一组 4 个 cell（最后一个为 GP-SMOTE），返回异常列表"""
    results = []
    gp_mean = cells[3][0]  # GP-SMOTE 的均值

    # 找出该组最大值
    means = [c[0] for c in cells if c[0] is not None]
    max_val = max(means) if means else None

    for i, (mean, has_hl, symbol) in enumerate(cells):
        if mean is None:
            continue

        col_name = f"{group_name}-{'M' + str(i+1) if i < 3 else 'GP'}"
        anomalies = []

        # === Wilcoxon 符号检查（仅非 GP-SMOTE 列） ===
        if i < 3 and symbol is not None and gp_mean is not None:
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
            "GP_Mean": gp_mean if i < 3 else "",
            "Has_HL": has_hl,
            "Is_Max": (mean == max_val) if max_val is not None else False,
            "Status": anomaly_str
        })
    return results


def main():
    txt_path = os.path.join(os.path.dirname(__file__), "table.txt")
    output_path = os.path.join(os.path.dirname(__file__), "table_check_result.csv")

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_rows = []
    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            continue
        d_label, fmeasure_cells, auc_cells = parsed

        f_results = check_group(fmeasure_cells, "F-measure")
        for r in f_results:
            r["Dataset"] = d_label
            r["Group"] = "F-measure"
        all_rows.extend(f_results)

        auc_results = check_group(auc_cells, "AUC")
        for r in auc_results:
            r["Dataset"] = d_label
            r["Group"] = "AUC"
        all_rows.extend(auc_results)

    # 写 CSV
    fieldnames = ["Dataset", "Group", "Column", "Mean", "GP_Mean", "Symbol", "Has_HL", "Is_Max", "Status"]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # 统计汇总
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
