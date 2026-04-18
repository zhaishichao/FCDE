import pandas as pd
import os

# 合并结果所在目录
file_root = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\runtime\\"

# ====== 1. 读取多个CSV ======
file_paths = [
    "mean_dg.csv",
    "mean_bs.csv",
    "mean_ds.csv",
    # 以后可以继续加
]

method_names = [
    "DG-SMOTE",
    "MTGP-SMOTE",
    "Blind-SMOTE",
    "GP-SMOTE",
]

dfs = []
for path, name in zip(file_paths, method_names):
    path = os.path.join(file_root, path)
    df = pd.read_csv(path)
    df = df.rename(columns={"time": name})
    dfs.append(df)

# ====== 2. 合并 ======
df_merged = dfs[0]
for df in dfs[1:]:
    df_merged = pd.merge(df_merged, df, on="数据集")

# ====== 3. 按原顺序编号 D1, D2... ======
df_merged = df_merged.reset_index(drop=True)
df_merged["ID"] = ["D{}".format(i + 1) for i in range(len(df_merged))]

# 如果你只想显示 D1 而不显示原dataset名字：
df_merged_display = df_merged[["ID"] + method_names]

# ====== 4. 保留两位小数 ======
for col in method_names:
    df_merged_display[col] = df_merged_display[col].astype(float).round(2)

# ====== 5. 计算平均 ======
avg_vals = [round(df_merged_display[col].mean(), 2) for col in method_names]

# ====== 6. 生成LaTeX ======
latex = []
latex.append("\\begin{table}[htbp]")
latex.append("\\centering")
latex.append("\\caption{Average running time (seconds) over 30 runs}")
latex.append("\\label{table:time}")
latex.append("\\setlength{\\tabcolsep}{3mm}")
latex.append("\\begin{tabular}{" + "c" * (len(method_names) + 1) + "}")
latex.append("\\toprule")

# 表头
header = ["\\textbf{Dataset}"] + [f"\\textbf{{{m}}}" for m in method_names]
latex.append(" & ".join(header) + " \\\\")
latex.append("\\midrule")

# 数据行
for _, row in df_merged_display.iterrows():
    values = [row["ID"]]

    min_val = min([row[m] for m in method_names])

    for m in method_names:
        val = row[m]
        values.append(f"{val:.2f}")

    latex.append(" & ".join(values) + " \\\\")

latex.append("\\midrule")

# 平均行
min_avg = min(avg_vals)
avg_formatted = []
for val in avg_vals:
    if val == min_avg:
        avg_formatted.append(f"\\textbf{{{val:.2f}}}")
    else:
        avg_formatted.append(f"{val:.2f}")

latex.append("Average & " + " & ".join(avg_formatted) + " \\\\")

latex.append("\\bottomrule")
latex.append("\\end{tabular}")
latex.append("\\end{table}")

print("\n".join(latex))
