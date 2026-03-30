import os
import pandas as pd

# 分类器列表
classifiers = ["dt", "knn", "svm"]

# 要比较的指标
metrics = ["F-measure", "G-mean", "AUC"]

# 合并结果所在目录
merged_root = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\最终对比\\"
# 输出目录
output_root = "C:\\Users\\zsc\\Desktop\\FCDE实验结果汇总\\最终对比\\"
os.makedirs(output_root, exist_ok=True)

for clf in classifiers:
    clf_dir = os.path.join(merged_root, clf)

    # 读取 DG-SMOTE 和 GP-SMOTE 数据
    dg_file = os.path.join(clf_dir, "DG-SMOTE.csv")
    gp_file = os.path.join(clf_dir, "GP-SMOTE.csv")

    if not (os.path.exists(dg_file) and os.path.exists(gp_file)):
        print(f"Warning: Missing files for classifier {clf}")
        continue

    df_dg = pd.read_csv(dg_file)
    df_gp = pd.read_csv(gp_file)

    # 假设第一列为数据集名，如果没有请修改列名
    dataset_col = df_dg.columns[0]

    comparison_df = pd.DataFrame()
    comparison_df["数据集"] = df_dg[dataset_col]

    # 对每个指标进行比较
    for metric in metrics:
        if metric in df_dg.columns and metric in df_gp.columns:
            comparison_df[metric] = df_gp[metric] > df_dg[metric]
        else:
            print(f"Warning: Metric {metric} not found in CSVs for {clf}")
            comparison_df[metric] = False  # 默认 False

    # 保存对比结果
    output_file = os.path.join(output_root, f"{clf}_comparison.csv")
    comparison_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved comparison CSV for {clf}: {output_file}")