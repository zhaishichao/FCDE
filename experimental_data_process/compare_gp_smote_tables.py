import re
import os
import csv


def parse_cell(cell):
    """返回 (mean, std_str)"""
    cell = cell.strip()
    has_hl = cell.startswith("\\hl{")
    if has_hl:
        inner = cell[4:-1]
    else:
        inner = cell

    # 去掉 Wilcoxon 符号
    for sym in ["$+$", "$-$", "$\\approx$"]:
        if inner.endswith(sym):
            inner = inner[:-len(sym)]
            break

    m = re.match(r"([\d.]+)\(([^)]+)\)", inner)
    if m:
        return float(m.group(1)), m.group(2)
    return None, None


def parse_table(filepath, gp_indices):
    """
    解析表格，返回 {dataset_label: {group_name: (mean, std_str)}}
    gp_indices: list of (group_name, col_index) 指定每组的 GP-SMOTE 列
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split(" & ")
        if parts[-1].endswith(" \\\\"):
            parts[-1] = parts[-1][:-3]

        d_label = parts[0].strip()
        d_match = re.match(r"D(\d+)", d_label)
        if not d_match:
            continue

        dataset = f"D{int(d_match.group(1))}"

        values = {}
        for group_name, col_idx in gp_indices:
            if col_idx < len(parts):
                mean, std = parse_cell(parts[col_idx])
                values[group_name] = (mean, std)
        result[dataset] = values
    return result


def main():
    out_dir = os.path.dirname(__file__)

    # table1: 单列 F-measure，GP-SMOTE 在第 7 列（0-indexed）
    table1 = parse_table(
        os.path.join(out_dir, "table1.txt"),
        [("F-measure", 7)]
    )

    # table2: 单列 AUC，GP-SMOTE 在第 7 列
    table2 = parse_table(
        os.path.join(out_dir, "table2.txt"),
        [("AUC", 7)]
    )

    # table3: 双列 F-measure + AUC
    # F-measure GP-SMOTE 在第 4 列，AUC GP-SMOTE 在第 8 列
    table3 = parse_table(
        os.path.join(out_dir, "table3.txt"),
        [("F-measure", 4), ("AUC", 8)]
    )

    # 对比
    rows = []
    all_datasets = sorted(table3.keys(), key=lambda x: int(x[1:]))

    for ds in all_datasets:
        # F-measure: table1 vs table3
        fm1 = table1.get(ds, {}).get("F-measure")
        fm3 = table3.get(ds, {}).get("F-measure")

        if fm1 and fm3:
            mean_match = (fm1[0] == fm3[0])
            std_match = (fm1[1] == fm3[1])
            fm_status = "OK" if (mean_match and std_match) else (
                f"均值{'同' if mean_match else '异'}, 标准差{'同' if std_match else '异'}"
            )
        else:
            fm_status = "数据缺失"

        # AUC: table2 vs table3
        auc2 = table2.get(ds, {}).get("AUC")
        auc3 = table3.get(ds, {}).get("AUC")

        if auc2 and auc3:
            mean_match = (auc2[0] == auc3[0])
            std_match = (auc2[1] == auc3[1])
            auc_status = "OK" if (mean_match and std_match) else (
                f"均值{'同' if mean_match else '异'}, 标准差{'同' if std_match else '异'}"
            )
        else:
            auc_status = "数据缺失"

        rows.append({
            "Dataset": ds,
            "T1_F1_mean": fm1[0] if fm1 else "",
            "T1_F1_std": fm1[1] if fm1 else "",
            "T3_F1_mean": fm3[0] if fm3 else "",
            "T3_F1_std": fm3[1] if fm3 else "",
            "F1_Status": fm_status,
            "T2_AUC_mean": auc2[0] if auc2 else "",
            "T2_AUC_std": auc2[1] if auc2 else "",
            "T3_AUC_mean": auc3[0] if auc3 else "",
            "T3_AUC_std": auc3[1] if auc3 else "",
            "AUC_Status": auc_status,
        })

    # 统计
    f1_ok = sum(1 for r in rows if r["F1_Status"] == "OK")
    f1_fail = sum(1 for r in rows if r["F1_Status"] != "OK" and r["F1_Status"] != "数据缺失")
    auc_ok = sum(1 for r in rows if r["AUC_Status"] == "OK")
    auc_fail = sum(1 for r in rows if r["AUC_Status"] != "OK" and r["AUC_Status"] != "数据缺失")

    # 写 CSV
    output_path = os.path.join(out_dir, "gp_smote_table_comparison.csv")
    fieldnames = ["Dataset",
                  "T1_F1_mean", "T1_F1_std", "T3_F1_mean", "T3_F1_std", "F1_Status",
                  "T2_AUC_mean", "T2_AUC_std", "T3_AUC_mean", "T3_AUC_std", "AUC_Status"]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 汇总
    print(f"F-measure 一致: {f1_ok}/{len(rows)}, 不一致: {f1_fail}")
    print(f"AUC 一致:       {auc_ok}/{len(rows)}, 不一致: {auc_fail}")

    if f1_fail > 0:
        print("\nF-measure 不一致详情:")
        for r in rows:
            if r["F1_Status"] != "OK" and r["F1_Status"] != "数据缺失":
                print(f"  {r['Dataset']}: {r['F1_Status']}")
                print(f"    T1: {r['T1_F1_mean']}({r['T1_F1_std']})  T3: {r['T3_F1_mean']}({r['T3_F1_std']})")

    if auc_fail > 0:
        print("\nAUC 不一致详情:")
        for r in rows:
            if r["AUC_Status"] != "OK" and r["AUC_Status"] != "数据缺失":
                print(f"  {r['Dataset']}: {r['AUC_Status']}")
                print(f"    T2: {r['T2_AUC_mean']}({r['T2_AUC_std']})  T3: {r['T3_AUC_mean']}({r['T3_AUC_std']})")

    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
